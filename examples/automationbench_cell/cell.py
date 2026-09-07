"""Run one Claude Code cell over AutomationBench under protocol v2, and read what it scored.

The cell this reruns served the same benchmark, in the same order, to the same model at the same
effort, and told the agent its score the moment each task ended. What differs is the serving
contract: the agent asks for work with ``pull`` rather than being handed it, every call names the
attempt it belongs to, the terminal is intercepted and sealed by the stream rather than by the
harness, and the score reaches the agent as a payload the stream released against the attempt
instead of as a field on the tool result that ended it. That difference is the measurement, so
everything else here is pinned to what the earlier cell ran.

Two commands. ``run`` composes the generation, serves it over stdio, and launches ``claude``
against it, keeping the transcript and the durable history side by side in one directory.
``table`` reads that directory back: the rows ``shogym results`` prints, and beside them the join
that says which benchmark task each attempt was and how much work went into it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Dict, List, Optional

from examples.automationbench_cell import pinned
from examples.automationbench_cell import table as read_back
from examples.automationbench_cell.serve import ROSTER_FILE, SERVER, release_for, roster

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: The model and effort the cell this one reruns was served to. Both are overridable, because a
#: smoke run proving the pipe should not cost what a cell costs, and the earlier harness took the
#: same two as overrides for the same reason.
MODEL = "claude-opus-5"
EFFORT = "xhigh"

#: The user turn. Everything else the agent is told is the system prompt, which is where the
#: earlier cell put it so that it survives compaction.
KICKOFF = "Begin."

#: The regime the earlier cell ran, read off its broker rather than its documentation: the score
#: and the environment's numbers reached the agent inside the result of the call that ended each
#: task, on every task, with no path that could withhold one. Under this protocol the equivalent
#: is the honest payload released at each seal, which is what ``immediate`` is.
CELL_ONE_SCHEDULE = "immediate"

#: What the run directory holds. The durable history and the sealed grades live under ``grades``,
#: which is a sibling of the directory the agent works in rather than a child of it.
GRADES = "grades"
SELF = "self"
HOME = "home"
CONFIG = "cfg"
TRANSCRIPT = "stream.jsonl"
STDERR = "stream.err.txt"
RUN_FILE = "run.json"

#: What a launch record says when the launch did not finish what it started. This launcher
#: records no status at all, which is a launch presenting its run as finished; one that knows
#: better says so with this, and a read holds a finished run to checks it lets an unfinished one
#: report as unavailable.
INCOMPLETE = "incomplete"


def new_run_dir(runs: Path, *, schedule: str) -> Path:
    """A fresh directory for one cell.

    Fresh per launch for the same reason the quickstart's is: the generation writes its manifest
    once and refuses a directory that already holds one. The stamp names whole seconds, so the
    token is what keeps two launches inside one second apart.
    """
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    return runs / f"cell-{schedule}-{stamp}-{token_hex(3)}"


def mcp_config(run_dir: Path, *, tasks: str, domain: str, schedule: str) -> Path:
    """Write the config Claude Code spawns the server from, and return its path.

    The server is spawned over stdio exactly as the quickstart's is, and what it is told is the
    roster, the domain and the schedule. It goes beside the run directory rather than inside the
    directory the agent works in, which is where the earlier cell kept its own.
    """
    directory = run_dir / CONFIG
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    SERVER: {
                        "command": "uv",
                        "args": [
                            "run",
                            "--project",
                            str(REPO),
                            "python",
                            str(HERE / "serve.py"),
                        ],
                        "env": {
                            "SHOGYM_CELL_TASKS": tasks,
                            "SHOGYM_CELL_DOMAIN": domain,
                            "SHOGYM_CELL_SCHEDULE": schedule,
                            "SHOGYM_CELL_RUN_DIR": str(run_dir / GRADES),
                        },
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def claude_argv(
    config: Path, *, model: str, effort: str, system_prompt: str, session_id: str
) -> List[str]:
    """The command the earlier cell launched, with this protocol's prompt in it.

    Every flag is the one that cell passed. ``--strict-mcp-config`` keeps the operator's own MCP
    servers out of the run, and nothing is denied: that cell ran its rollout arm with the agent's
    own tools left in place, web included, so a rerun that took them away would be comparing two
    things at once.
    """
    return [
        "claude",
        "-p",
        KICKOFF,
        "--model",
        model,
        "--effort",
        effort,
        "--mcp-config",
        str(config),
        "--strict-mcp-config",
        "--permission-mode",
        "bypassPermissions",
        "--forward-subagent-text",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--append-system-prompt",
        system_prompt,
        "--session-id",
        session_id,
    ]


def write_run_file(run_dir: Path, record: Dict[str, object]) -> Path:
    """Write the run's own record of itself, and return the path."""
    path = run_dir / RUN_FILE
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
    allow_cli_drift: bool = False,
) -> int:
    """Serve the roster and run the agent against it, and return what the agent exited with.

    The agent works in a directory of its own holding the one file the earlier cell's agent found
    in its own, with a fresh Claude Code home beside it, which is the shape that cell got from a
    fresh container. What the process is handed is built rather than inherited, and what it
    resolved to is written down: the argv, the environment, a digest of each directory the agent
    started from, and the CLI build, so that a launch nobody could pin is still a launch somebody
    can read back.

    The grades are kept in a sibling directory rather than under the one the agent works in. That
    is a weaker boundary than a mount the agent has no path to, and it is the honest description
    of what a host launch can offer: an agent running under bypassPermissions can read the whole
    filesystem, so what keeps this run's record straight is that nothing asks the agent about it.
    """
    positions = roster(tasks)
    # These three are read here for their refusals. A misspelled roster, an unknown schedule or a
    # CLI that is not the recorded build are all mistakes to hear about now rather than from a
    # server the agent has already been launched at, or from a comparison months later.
    release_for(schedule)
    cli_version = pinned.resolve_cli_version()
    pinned.check_cli_version(cli_version, allow_drift=allow_cli_drift)

    work = run_dir / SELF
    pinned.seed_workdir(work)
    home = run_dir / HOME
    home.mkdir(parents=True, exist_ok=True)
    config = mcp_config(run_dir, tasks=tasks, domain=domain, schedule=schedule)
    session_id = str(uuid.uuid4())
    system_prompt = (HERE / "PROMPT.txt").read_text(encoding="utf-8").strip()
    argv = claude_argv(
        config, model=model, effort=effort, system_prompt=system_prompt, session_id=session_id
    )
    environment = pinned.child_environment(os.environ, config_dir=home)
    carried = pinned.credential_name(os.environ)
    record: Dict[str, object] = {
        "model": model,
        "effort": effort,
        "schedule": schedule,
        "domain": domain,
        "tasks": tasks,
        "task_count": len(positions),
        "session_id": session_id,
        "server": SERVER,
        "started": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        "argv": argv,
        "cwd": str(work),
        # The environment as the child was handed it, never as the operator's shell had it, and
        # with the credential named rather than copied: which name authenticated the run is part
        # of the launch, and what it was worth is not.
        "environment": pinned.redacted(environment),
        "credential": carried,
        "digests": {
            "config": pinned.digest_tree(run_dir / CONFIG),
            "work": pinned.digest_tree(work),
            "home": pinned.digest_tree(home),
        },
        "cli_version": cli_version,
        "cli_version_recorded": pinned.CLI_VERSION,
    }
    # Written before the launch, so a run that dies mid-flight still says what it started as.
    # The session is what the transcript is found under afterwards.
    write_run_file(run_dir, record)
    if carried is None:
        # The agent is given a fresh Claude Code home, so nothing the operator's own home holds
        # reaches it. Whatever authenticates this run is then outside both, and unrecorded.
        print(f"[cell] no {' or '.join(pinned.CREDENTIALS)} in the environment")
    print(f"[cell] {len(positions)} tasks, {schedule}, {model}/{effort} -> {run_dir}")
    with (run_dir / TRANSCRIPT).open("wb") as out, (run_dir / STDERR).open("wb") as err:
        finished = subprocess.run(
            argv, cwd=work, env=environment, stdout=out, stderr=err, check=False
        )
    # The tool surface is the half of the launch that only exists once the agent has started, so
    # it is read out of the transcript's first line and compared to the recorded one here.
    init = pinned.init_event(run_dir / TRANSCRIPT)
    drift = pinned.surface_drift(init)
    record["exit_code"] = finished.returncode
    record["init"] = init
    record["drift"] = drift
    write_run_file(run_dir, record)
    print(f"[cell] agent exited {finished.returncode}; transcript {run_dir / TRANSCRIPT}")
    for line in pinned.drift_report(drift):
        print(line)
    return finished.returncode


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
    print(f"\n{transcript.pulls} pulls, {transcript.unserved} calls to the agent's own tools")
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
        default="cell-one",
        help="the roster: `cell-one`, `cell-one:20`, `0-19`, or `4,0,2` (default: cell-one)",
    )
    run.add_argument("--domain", default="public", help="automationbench domain (default: public)")
    run.add_argument(
        "--schedule",
        default=CELL_ONE_SCHEDULE,
        help=(
            "what the agent is told: `immediate` is the honest score at every seal, which is "
            f"the regime the earlier cell ran; `never` tells it nothing (default: "
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
            allow_cli_drift=args.allow_cli_drift,
        )
    return asyncio.run(table(Path(args.run_dir).expanduser()))


if __name__ == "__main__":
    sys.exit(main())
