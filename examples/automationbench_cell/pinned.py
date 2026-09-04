"""What the recorded run's agent started from, beside the command line it was started with.

The argv is the visible half of a launch and it is pinned in ``cell.py``. This is the other half:
what the working directory holds, the environment the process is handed, the CLI build that reads
both, and the tool surface that build reports on the transcript's first line. None of those appear
in the command, every one of them can change the agent, and a rerun that let them drift would be
comparing more than the serving contract.

So each is written down here with the value the recorded run had, and a launch does what it can
with each. The working directory is made empty, because that run's was. The environment the
agent's container is given is built rather than inherited, so an operator's shell cannot reach
into the run. The CLI build is resolved before anything is spawned and refused when it is not the
pinned one. The tool surface can only be read afterwards, so it is read out of the transcript and
compared, and what differs is written into the run's own record.

The run this cell is pinned to is named here as well, by its own identifier and by the digests of
the things a later reader would otherwise have to take on trust: the prompt it served and the
split its roster was drawn from. A cell pinned to some other run is then a different set of
constants rather than a directory somebody has to remember the provenance of.
"""

from __future__ import annotations

import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: The run this cell reruns, and what identifies it. The transcript, the manifest and the results
#: of that run are the ground truth every constant here was read off, so the run is named rather
#: than described: a reader with the record in front of them can check any of this, and a reader
#: without it can at least tell which run was meant. ``RECORDED_PROMPT_SHA256`` is the digest of
#: the standing instruction that run served, which ``PROMPT.txt`` is the two protocol substitutions
#: away from, and ``RECORDED_SPLIT_DIGEST`` names the split its roster was drawn from.
RECORDED_RUN = "automationbench-claude_code-claude-opus-5-20260819T011123Z"
RECORDED_PROMPT_SHA256 = "610c05901efb0f4717a30bf9373bd089113af5ee5918a7ff6c654d33e6f6c175"
RECORDED_SPLIT_DIGEST = "f8a5e70dfff2ea07efeaa897ec26cddd6813387efcb9403c6a4f2de1ee7c1536"

#: The benchmark that run was over, and the checkout that served it. This cell serves whatever
#: revision its own adapter pins, which is not this one: the task text and the scoring assertions
#: moved between them, so two runs can match in every launch fact here and still be over different
#: tasks. The launch record names both.
RECORDED_BENCHMARK_REVISION = "a321764ace3cfbe42289e6a13abef2f0f4f56fad"
RECORDED_SHOGYM_REVISION = "9eb9edb88087af9a08520482a2d1de5831870944"

#: The CLI build the recorded run reported on its first line. The agent is the model and the
#: harness together, so a different build is a different agent: its system prompt, its built-in
#: tools and its compaction are the CLI's and not the model's.
CLI_VERSION = "2.1.226"

#: Where the CLI is installed from, and what the image it is installed into is built on. The base
#: is named by digest rather than by tag because a tag moves: the shell, the Node and the OS
#: packages the model reaches through Bash all belong to that image, and a rerun months from now
#: on a moved tag would differ from this one in every one of them. It is a later base than the
#: recorded image's, since that one was built on the tag and what the tag pointed at then cannot
#: be recovered; what this fixes is that every rerun from here starts from one pinned index.
AGENT_BASE = (
    "node:22-bookworm-slim@"
    "sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5"
)

#: The package the recorded image installed, and the registry this one installs it from. The two
#: are not alike in provenance: the package is that image's own, and the registry is this cell's
#: choice, because that image passed none and took whatever npm defaulted to that day. It is named
#: so the default cannot move under a rerun, and it is a difference from the recorded build rather
#: than a reconstruction of it.
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
#:
#: The day is the recorded run's own, and it is a bound rather than a reconstruction: that image
#: was built from an unpinned archive on a day nobody wrote down, and the image itself is gone, so
#: what can be said is that it existed by the day the run started. The package list is that run's,
#: ``fd-find`` and its ``fd`` symlink included, resolved at this snapshot.
APT_SNAPSHOT = "20260819T000000Z"
APT_PACKAGES: Tuple[str, ...] = (
    "ca-certificates=20250419~deb12u1",
    "curl=7.88.1-10+deb12u15",
    "fd-find=8.6.0-3",
    "git=1:2.39.5-0+deb12u3",
    "jq=1.6-2.1+deb12u2",
    "procps=2:4.0.2-3",
    "python3=3.11.2-1+b1",
    "python3-pip=23.0.1+dfsg-1",
    "python3-venv=3.11.2-1+b1",
    "ripgrep=13.0.0-4+b2",
)

#: What the agent's image is built from, and what a launch checks the image on the host against.
#: The recorded run kept its image's digest and nothing it was built from, and the image itself is
#: gone from this machine, so this is not equality with that one: it pins what this cell builds,
#: and the launch record says so rather than claiming equality. ``dockerfile``
#: is a digest of the file beside this one, so an edit to it that nobody recorded here is a
#: refusal rather than a silent rebuild.
AGENT_IMAGE_BUILD: Dict[str, str] = {
    "apt_packages": " ".join(APT_PACKAGES),
    "apt_snapshot": APT_SNAPSHOT,
    "base": AGENT_BASE,
    "cli_package": CLI_PACKAGE,
    "cli_registry": CLI_REGISTRY,
    "cli_version": CLI_VERSION,
    "dockerfile": "sha256:c171430ab4c3a622fd682d9844d858c041bb25e8807822a25815aaefb9564953",
}

#: What ``digest_tree`` answers for a directory holding no file. The recorded run's working
#: directory was empty when the agent started and empty when it stopped, and Claude Code reads a
#: project instruction file in that directory into the system prompt, so a file seeded there would
#: be a second prompt riding on top of the standing one. The digest is named rather than described
#: because it is what the run record carries: an empty directory is a fact a reader can check.
EMPTY_TREE = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

#: What that run set in the agent's environment, and the whole of it. The container it ran in
#: supplied the rest, which is why this list is short rather than trimmed. ``NODE_OPTIONS`` is
#: empty on purpose and in both places, the image and the launch, as that run had it: whatever an
#: inherited one would have added is options the Node the CLI runs on was not started with there.
PINNED_ENVIRONMENT: Dict[str, str] = {"IS_SANDBOX": "1", "NODE_OPTIONS": ""}

#: The credential a launch here may carry through. That run authenticated in subscription mode
#: with a token passed in by name, and this one does the same: the Claude Code home is a fresh
#: directory, so whatever the operator's own home holds is not in it.
CREDENTIALS: Tuple[str, ...] = ("CLAUDE_CODE_OAUTH_TOKEN",)

#: A credential this cell will not run on, and refuses rather than falls back to. An API key is a
#: different billing arm from the subscription the recorded run used, and an arm is the sort of
#: thing a comparison is about, so a launch that found one and used it would have changed the
#: measurement quietly. It is named here so a record never keeps its value either.
REFUSED_CREDENTIALS: Tuple[str, ...] = ("ANTHROPIC_API_KEY",)

#: The model-visible surface that build reported: the tools it offered, the subagents it could
#: start, and the skills it could load. A rerun cannot pin these, because they belong to the
#: build and to whatever the account has enabled. It can say what they were, which is what turns
#: an unexplained behaviour change into a difference somebody can point at.
#:
#: These are the built-in tools alone, in the order that run's first line listed them. What the
#: run's own server offered is the other half of the same array and is compared separately, since
#: those names are this cell's to decide and the built-ins are not.
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
    "ListAgents",
    "ListMcpResourcesTool",
    "Monitor",
    "NotebookEdit",
    "PushNotification",
    "Read",
    "ReadMcpResourceDirTool",
    "ReadMcpResourceTool",
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

#: How a harness names a tool it was given by an MCP server, which is what tells the two halves of
#: the reported surface apart.
SERVED_MARK = "mcp__"

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

    A key this cell does not run on is refused with the reason rather than passed over, because
    the operator who exported it meant it to be used and would otherwise be told only that nothing
    authenticated the run.
    """
    carried = credential_name(ambient)
    if carried is not None:
        return carried
    refused = [name for name in REFUSED_CREDENTIALS if ambient.get(name)]
    if refused:
        raise ValueError(
            f"the environment carries {' and '.join(refused)} and no {CREDENTIALS[0]}. The "
            f"recorded run authenticated on a subscription, and an API key is a different billing "
            f"arm, which is the sort of thing this cell exists to hold still: export a "
            f"{CREDENTIALS[0]} rather than have the run change arm quietly"
        )
    raise ValueError(
        f"no {CREDENTIALS[0]} in the environment: the agent's Claude Code home is fresh, so "
        "nothing else can authenticate the run"
    )


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
    """Return the environment as a record may keep it, with any credential value removed.

    Both the credential this cell runs on and the one it refuses are covered, because what a
    record must never hold is a secret and not a secret of the approved sort.
    """
    secret = set(CREDENTIALS) | set(REFUSED_CREDENTIALS)
    return {name: REDACTED if name in secret else value for name, value in environment.items()}


def empty_workdir(work: Path) -> Path:
    """Create the directory the agent works in, empty, as the recorded run's was.

    Empty is the pin. Claude Code reads a project instruction file in the working directory into
    the system prompt, so a run that seeded one there would be serving a second standing
    instruction beside the one the launch passes, and the two runs would differ in the prompt as
    well as in the serving.
    """
    work.mkdir(parents=True, exist_ok=True)
    return work


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


def image_drift(resolved: Mapping[str, str]) -> Dict[str, Dict[str, Optional[str]]]:
    """Return which of the image's recorded inputs this build differs in, and how.

    It is the answer the refusal is made of, and it is the answer a run that was allowed to drift
    has to keep: an operator who passes ``--allow-image-drift`` is told the difference is recorded,
    and a record holding only what this host resolved could not say which of it was unexpected.
    """
    return {
        name: {"recorded": AGENT_IMAGE_BUILD.get(name), "resolved": resolved.get(name)}
        for name in sorted(set(AGENT_IMAGE_BUILD) | set(resolved))
        if AGENT_IMAGE_BUILD.get(name) != resolved.get(name)
    }


def check_image_build(resolved: Mapping[str, str], *, allow_drift: bool) -> None:
    """Refuse an agent image built from other inputs, unless the operator has said to allow it.

    An image identity is not an image: the tag says which build was asked for and the identity
    says what it was made of, and only the second can be compared with a rerun months later. So
    what is checked is the base, the package, the registry, the version and the recipe, and the
    refusal names whichever of them moved.
    """
    differing = image_drift(resolved)
    if not differing or allow_drift:
        return
    moved = ", ".join(
        f"{name}: recorded {value['recorded']!r}, resolved {value['resolved']!r}"
        for name, value in differing.items()
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


def _reported(init: Mapping[str, Any], key: str) -> List[str]:
    """Return one of the init line's name lists, or nothing where the line carries no such list."""
    listed = init.get(key)
    return [str(item) for item in listed] if isinstance(listed, list) else []


def _surface_drift(recorded: Sequence[str], found: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Return how one reported list differs from the recorded one, or nothing where it does not.

    Membership is reported first and as two lists, because a name the recorded agent had and this
    one does not is a different story from a name this one was given.

    A list whose membership matches is still compared in full. These are ordered arrays and the
    model reads them in the order they are written, so a build that offered the same tools in
    another order offered a different prompt prefix, and a name listed twice is a surface nobody
    recorded either. Both used to pass as a faithful surface, because the comparison was over
    sets: the whole list is reported in that case, since the difference is the sequence and
    naming a name would not show it.
    """
    missing = sorted(set(recorded) - set(found))
    added = sorted(set(found) - set(recorded))
    if missing or added:
        return {"missing": missing, "added": added}
    if list(found) != list(recorded):
        return {"recorded_order": list(recorded), "reported_order": list(found)}
    return None


def surface_drift(init: Optional[Dict[str, Any]], *, served: Sequence[str]) -> Dict[str, Any]:
    """Return how this run's agent surface differed from the recorded one, or nothing if it did not.

    What is compared is what the model could see: the build, the tools it was offered, the
    subagents it could start and the skills it could load.

    The reported tool array holds two surfaces and they are compared as two. The built-ins belong
    to the CLI build and are the recorded ones; the names an MCP server contributed belong to this
    cell, which serves a different control tool under a protocol that serves no separate abort, so
    a faithful run differs there by construction. Compared as one array every good run reported
    drift, which is the same false alarm the missing built-ins used to raise from the other side.
    """
    if init is None:
        return {"init": "the transcript holds no init line, so what served this run is unknown"}
    drift: Dict[str, Any] = {}
    version = init.get("claude_code_version")
    if version != CLI_VERSION:
        drift["claude_code_version"] = {"recorded": CLI_VERSION, "resolved": version}
    tools = _reported(init, "tools")
    for name, recorded, found in (
        ("tools", CLI_TOOLS, [tool for tool in tools if not tool.startswith(SERVED_MARK)]),
        ("served", tuple(served), [tool for tool in tools if tool.startswith(SERVED_MARK)]),
        ("agents", CLI_AGENTS, _reported(init, "agents")),
        ("skills", CLI_SKILLS, _reported(init, "skills")),
    ):
        difference = _surface_drift(recorded, found)
        if difference is not None:
            drift[name] = difference
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
    "EMPTY_TREE",
    "PINNED_ENVIRONMENT",
    "RECORDED_BENCHMARK_REVISION",
    "RECORDED_PROMPT_SHA256",
    "RECORDED_RUN",
    "RECORDED_SHOGYM_REVISION",
    "RECORDED_SPLIT_DIGEST",
    "REDACTED",
    "REFUSED_CREDENTIALS",
    "SERVED_MARK",
    "agent_environment",
    "check_cli_version",
    "check_image_build",
    "check_credential",
    "credential_name",
    "digest_tree",
    "drift_report",
    "empty_workdir",
    "image_drift",
    "init_event",
    "redacted",
    "resolve_cli_version",
    "surface_drift",
]
