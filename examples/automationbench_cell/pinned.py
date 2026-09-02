"""What the earlier cell's agent started from, beside the command line it was started with.

The argv is the visible half of a launch and it is pinned in ``cell.py``. This is the other half:
the file the working directory holds, the environment the process is handed, the CLI build that
reads both, and the tool surface that build reports on the transcript's first line. None of those
appear in the command, every one of them can change the agent, and a rerun that let them drift
would be comparing more than the serving contract.

So each is written down here with the value the recorded run had, and a launch does what it can
with each. The working directory is seeded with the file that cell seeded. The environment the
agent's container is given is built rather than inherited, so an operator's shell cannot reach
into the run. The CLI build is resolved before anything is spawned and refused when it is not the
pinned one. The tool surface can only be read afterwards, so it is read out of the transcript and
compared, and what differs is written into the run's own record.
"""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: The CLI build the recorded run reported on its first line. The agent is the model and the
#: harness together, so a different build is a different agent: its system prompt, its built-in
#: tools and its compaction are the CLI's and not the model's.
CLI_VERSION = "2.1.220"

#: Where the CLI is installed from, and what the image it is installed into is built on. The base
#: is named by digest rather than by tag because a tag moves: the shell, the Node and the OS
#: packages the model reaches through Bash all belong to that image, and a rerun months from now
#: on a moved tag would differ from this one in every one of them.
AGENT_BASE = (
    "node:22-bookworm-slim@"
    "sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5"
)
CLI_PACKAGE = "@anthropic-ai/claude-code"
CLI_REGISTRY = "https://registry.npmjs.org"

#: The moment of the Debian archive the agent's OS packages come from, and the exact version of
#: each one. A package name on its own resolves against whatever the live repository is serving on
#: the day of the build, so an image rebuilt next month would hold a different shell, a different
#: ``curl`` and a different ``python3`` under an identity that said nothing had moved. The archive
#: is immutable at a timestamp, which makes the resolution a fact rather than a date; the versions
#: are named as well, so a snapshot that stopped answering is a build that fails rather than one
#: that quietly resolves somewhere else. Both are handed to the build and both are part of what a
#: launch compares, because this is the surface the model reaches through Bash.
APT_SNAPSHOT = "20260801T000000Z"
APT_PACKAGES: Tuple[str, ...] = (
    "ca-certificates=20230311+deb12u1",
    "curl=7.88.1-10+deb12u15",
    "git=1:2.39.5-0+deb12u3",
    "jq=1.6-2.1+deb12u2",
    "procps=2:4.0.2-3",
    "python3=3.11.2-1+b1",
    "python3-pip=23.0.1+dfsg-1",
    "python3-venv=3.11.2-1+b1",
    "ripgrep=13.0.0-4+b2",
)

#: What the agent's image is built from, and what a launch checks the image on the host against.
#: The earlier cell recorded no image identity and its image is gone, so this is not equality with
#: that one: it pins what this cell builds, and the difference from that cell is stated in the
#: README rather than claimed away. ``dockerfile`` is a digest of the file beside this one, so an
#: edit to it that nobody recorded here is a refusal rather than a silent rebuild.
AGENT_IMAGE_BUILD: Dict[str, str] = {
    "apt_packages": " ".join(APT_PACKAGES),
    "apt_snapshot": APT_SNAPSHOT,
    "base": AGENT_BASE,
    "cli_package": CLI_PACKAGE,
    "cli_registry": CLI_REGISTRY,
    "cli_version": CLI_VERSION,
    "dockerfile": "sha256:8ce3f5fe2f727b2aced8f3dbc751c9e5638b77bf3190927f054d6583e96a07ab",
}

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


def check_credential(ambient: Mapping[str, str]) -> str:
    """Return the credential name this environment carries, and refuse if it carries none.

    The agent's Claude Code home is fresh, so the environment is the only thing that can
    authenticate it. A launch without a credential would build two images, serve a roster and
    record an agent that exited at once, which is a mistake to hear about before any of that.
    """
    carried = credential_name(ambient)
    if carried is None:
        raise ValueError(
            f"no {' or '.join(CREDENTIALS)} in the environment: the agent's Claude Code home is "
            "fresh, so nothing else can authenticate the run"
        )
    return carried


def agent_environment(ambient: Mapping[str, str]) -> Dict[str, str]:
    """Return the environment the agent's container is given, built rather than inherited.

    It is short because the image supplies the rest: the path, the home and the locale belong to
    the container, and what a launch adds is this cell's two settings and the name of whatever
    authenticates the run. A variable nobody wrote down here reaches neither the agent nor
    anything it starts, which is more than tidiness: this repo reads a benchmark source override
    and a cache location out of the ambient environment, and an operator's shell holding either
    would be running a different benchmark under a record that still named this one.
    """
    built = dict(PINNED_ENVIRONMENT)
    carried = credential_name(ambient)
    if carried is not None:
        built[carried] = ambient[carried]
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


def resolve_cli_version(command: Sequence[str] = ("claude", "--version")) -> str:
    """Return the version the CLI reports, asked before anything is spawned.

    What is asked is the build that will serve the run, which is the one inside the image the
    agent starts in rather than whichever one is on the launching host's path.
    """
    argv = list(command)
    try:
        answer = subprocess.run(argv, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as failure:
        raise ValueError(
            f"{argv[0]!r} could not be asked for its version, so what would run is unknown: "
            f"{failure}"
        ) from failure
    reported = answer.stdout.strip().split()
    if not reported:
        raise ValueError(f"{argv[0]!r} reported no version, so what would run is unknown")
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


def check_image_build(resolved: Mapping[str, str], *, allow_drift: bool) -> None:
    """Refuse an agent image built from other inputs, unless the operator has said to allow it.

    An image identity is not an image: the tag says which build was asked for and the identity
    says what it was made of, and only the second can be compared with a rerun months later. So
    what is checked is the base, the package, the registry, the version and the recipe, and the
    refusal names whichever of them moved.
    """
    differing = sorted(
        name
        for name in set(AGENT_IMAGE_BUILD) | set(resolved)
        if AGENT_IMAGE_BUILD.get(name) != resolved.get(name)
    )
    if not differing or allow_drift:
        return
    moved = ", ".join(
        f"{name}: recorded {AGENT_IMAGE_BUILD.get(name)!r}, resolved {resolved.get(name)!r}"
        for name in differing
    )
    raise ValueError(
        f"the agent's image is built from inputs this cell did not record, which is a different "
        f"agent and not a different serving contract ({moved}). Restore the recorded inputs, or "
        f"pass --allow-image-drift to run anyway and have the difference recorded"
    )


def init_event(transcript: Path) -> Optional[Dict[str, Any]]:
    """Return the first line of the agent's transcript, which is what its harness started as.

    That line is the only place the run says which CLI build actually served it and what surface
    that build gave the model. It is read leniently, because a transcript from a run that died is
    still the record of how that run started, and a line at a time, because a session that worked
    a whole roster writes more of them than a launch has any reason to hold at once.
    """
    path = Path(transcript)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
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
    "AGENT_BASE",
    "AGENT_IMAGE_BUILD",
    "APT_PACKAGES",
    "APT_SNAPSHOT",
    "CLI_AGENTS",
    "CLI_PACKAGE",
    "CLI_REGISTRY",
    "CLI_SKILLS",
    "CLI_TOOLS",
    "CLI_VERSION",
    "CREDENTIALS",
    "PINNED_ENVIRONMENT",
    "PROJECT_FILE",
    "PROJECT_INSTRUCTIONS",
    "REDACTED",
    "agent_environment",
    "check_cli_version",
    "check_image_build",
    "check_credential",
    "credential_name",
    "digest_tree",
    "drift_report",
    "init_event",
    "redacted",
    "resolve_cli_version",
    "seed_workdir",
    "surface_drift",
]
