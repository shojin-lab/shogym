"""What the earlier cell's agent started from, beside the command line it was started with.

The argv is the visible half of a launch and it is pinned in ``cell.py``. This is the other half:
the file the working directory holds, the environment the process is handed, the CLI build that
reads both, and the tool surface that build reports on the transcript's first line. None of those
appear in the command, every one of them can change the agent, and a rerun that let them drift
would be comparing more than the serving contract.

So each is written down here with the value the recorded run had, and a launch does what it can
with each. The working directory is seeded with the file that cell seeded. The child environment
is built from an allowlist rather than inherited, so an operator's shell cannot reach into the
run. The CLI build is resolved before anything is spawned and refused when it is not the pinned
one. The tool surface can only be read afterwards, so it is read out of the transcript and
compared, and what differs is written into the run's own record.
"""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: The CLI build the recorded run reported on its first line. The agent is the model and the
#: harness together, so a different build is a different agent: its system prompt, its built-in
#: tools and its compaction are the CLI's and not the model's.
CLI_VERSION = "2.1.220"

#: The project instructions that run's working directory held, byte for byte. It is one line and
#: it says nothing, which is the point: an empty directory and a directory holding this are two
#: different prompts, because Claude Code reads the file into the system prompt when it is there.
PROJECT_FILE = "CLAUDE.md"
PROJECT_INSTRUCTIONS = "# self\n"

#: What that run set in the agent's environment, and the whole of it. The container it ran in
#: supplied the rest, which is why this list is short rather than trimmed.
PINNED_ENVIRONMENT: Dict[str, str] = {"IS_SANDBOX": "1", "ENABLE_TOOL_SEARCH": "true"}

#: The credential names a launch here may carry through, in the order it looks for one. That run
#: passed a token in by name and had nothing else to authenticate with, and neither has this: the
#: Claude Code home is a fresh directory, so whatever the operator's own home holds is not in it.
CREDENTIALS: Tuple[str, ...] = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")

#: What the operating system is allowed to contribute, and nothing else passes. The agent's own
#: environment is not a place to put experiment settings: this repo reads a source override and a
#: cache location out of the ambient environment, and an operator's shell holding either would
#: quietly serve tasks from somewhere other than the pinned benchmark while the run's record
#: still described the standard one.
INHERITED: Tuple[str, ...] = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TERM",
    "TMPDIR",
    "TZ",
    "USER",
)

#: The model-visible surface that build reported: the tools it offered, the subagents it could
#: start, and the skills it could load. A rerun cannot pin these, because they belong to the
#: build and to whatever the account has enabled. It can say what they were, which is what turns
#: an unexplained behaviour change into a difference somebody can point at.
CLI_TOOLS: Tuple[str, ...] = (
    "Task",
    "Bash",
    "CronCreate",
    "CronDelete",
    "CronList",
    "DesignSync",
    "Edit",
    "EnterWorktree",
    "ExitWorktree",
    "Monitor",
    "NotebookEdit",
    "PushNotification",
    "Read",
    "RemoteTrigger",
    "ReportFindings",
    "ScheduleWakeup",
    "SendMessage",
    "Skill",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "ToolSearch",
    "WebFetch",
    "WebSearch",
    "Workflow",
    "Write",
)
CLI_AGENTS: Tuple[str, ...] = ("claude", "Explore", "general-purpose", "Plan", "statusline-setup")
CLI_SKILLS: Tuple[str, ...] = (
    "deep-research",
    "design-sync",
    "dataviz",
    "update-config",
    "verify",
    "debug",
    "code-review",
    "simplify",
    "batch",
    "fewer-permission-prompts",
    "doctor",
    "loop",
    "schedule",
    "claude-api",
    "run",
    "run-skill-generator",
)

#: What a redacted environment says in place of a credential. The run's record has to say which
#: name carried the credential, because that is part of how the agent authenticated, and it must
#: never say what the credential was.
REDACTED = "<redacted>"


def credential_name(ambient: Mapping[str, str]) -> Optional[str]:
    """Return the credential name this environment carries, or nothing if it carries none."""
    return next((name for name in CREDENTIALS if ambient.get(name)), None)


def child_environment(ambient: Mapping[str, str], *, config_dir: Path) -> Dict[str, str]:
    """Return the environment the agent is launched with, built rather than inherited.

    An MCP server spawned over stdio is a child of the agent's process, so this environment is
    the server's too. Building it from a list means the two processes are handed the operating
    system, one credential and this cell's own settings, and a variable nobody wrote down here
    reaches neither of them.
    """
    built = {name: ambient[name] for name in INHERITED if name in ambient}
    carried = credential_name(ambient)
    if carried is not None:
        built[carried] = ambient[carried]
    built.update(PINNED_ENVIRONMENT)
    # The Claude Code home this run gives the agent. Memory and skills it writes land there, so a
    # cell starts from nothing and its self-edits stay with the run rather than with whoever
    # launched it.
    built["CLAUDE_CONFIG_DIR"] = str(config_dir)
    return built


def redacted(environment: Mapping[str, str]) -> Dict[str, str]:
    """Return the environment as a record may keep it, with any credential value removed."""
    return {name: REDACTED if name in CREDENTIALS else value for name, value in environment.items()}


def seed_workdir(work: Path) -> Path:
    """Create the directory the agent works in, holding the file that cell's agent found there."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / PROJECT_FILE
    path.write_text(PROJECT_INSTRUCTIONS, encoding="utf-8")
    return path


def digest_tree(root: Path) -> str:
    """Return one digest over everything under ``root``: each path, and each file's bytes.

    A run says which directories the agent was given, and this is what says what was in them.
    Two runs whose digests agree started the agent from the same files, and two that disagree
    have somewhere to look.
    """
    digest = sha256()
    root = Path(root)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def resolve_cli_version(executable: str = "claude") -> str:
    """Return the version the CLI on this host reports, asked before anything is spawned."""
    try:
        answer = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as failure:
        raise ValueError(
            f"{executable!r} could not be asked for its version, so what would run is unknown: "
            f"{failure}"
        ) from failure
    reported = answer.stdout.strip().split()
    if not reported:
        raise ValueError(f"{executable!r} reported no version, so what would run is unknown")
    return reported[0]


def check_cli_version(resolved: str, *, allow_drift: bool) -> None:
    """Refuse a CLI that is not the pinned build, unless the operator has said to allow it."""
    if resolved == CLI_VERSION or allow_drift:
        return
    raise ValueError(
        f"this cell was recorded on Claude Code {CLI_VERSION} and this host has {resolved}, "
        f"which is a different agent and not a different serving contract. Install the pinned "
        f"build, or pass --allow-cli-drift to run anyway and have the difference recorded"
    )


def init_event(transcript: Path) -> Optional[Dict[str, Any]]:
    """Return the first line of the agent's transcript, which is what its harness started as.

    That line is the only place the run says which CLI build actually served it and what surface
    that build gave the model. It is read leniently, because a transcript from a run that died is
    still the record of how that run started.
    """
    path = Path(transcript)
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("subtype") == "init":
            return event
    return None


def surface_drift(init: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return how this run's agent surface differed from the recorded one, or nothing if it did not.

    What is compared is what the model could see: the build, the working directory it was given,
    the tools it was offered, the subagents it could start and the skills it could load. Each
    difference is reported as what is missing and what is new, because a tool the recorded agent
    had and this one does not is a different story from a tool this one was given.
    """
    if init is None:
        return {"init": "the transcript holds no init line, so what served this run is unknown"}
    drift: Dict[str, Any] = {}
    version = init.get("claude_code_version")
    if version != CLI_VERSION:
        drift["claude_code_version"] = {"recorded": CLI_VERSION, "resolved": version}
    for name, recorded, key in (
        ("tools", CLI_TOOLS, "tools"),
        ("agents", CLI_AGENTS, "agents"),
        ("skills", CLI_SKILLS, "skills"),
    ):
        reported = init.get(key)
        found = {str(item) for item in reported} if isinstance(reported, list) else set()
        missing = sorted(set(recorded) - found)
        added = sorted(found - set(recorded))
        if missing or added:
            drift[name] = {"missing": missing, "added": added}
    return drift


def drift_report(drift: Mapping[str, Any]) -> List[str]:
    """Return the lines a launch prints about its own drift, one to a difference."""
    return [
        f"[cell] launch drift in {name}: {json.dumps(value, sort_keys=True)}"
        for name, value in sorted(drift.items())
    ]


__all__ = [
    "CLI_AGENTS",
    "CLI_SKILLS",
    "CLI_TOOLS",
    "CLI_VERSION",
    "CREDENTIALS",
    "INHERITED",
    "PINNED_ENVIRONMENT",
    "PROJECT_FILE",
    "PROJECT_INSTRUCTIONS",
    "REDACTED",
    "check_cli_version",
    "child_environment",
    "credential_name",
    "digest_tree",
    "drift_report",
    "init_event",
    "redacted",
    "resolve_cli_version",
    "seed_workdir",
    "surface_drift",
]
