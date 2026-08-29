"""The paired launch this port documents, run the way the README says to run it.

Offline and upstream-free. What is checked here is the *shape of the launch*, which is a property
of the commands and the config file rather than of an episode: whether the two arms hand the agent
the same process, and whether the config those commands name resolves the server from the
directory they are run in. The episodes themselves are driven in ``test_appworld_served.py``.

Both failures behind this module were failures of documentation that no test could have caught,
because no test ran what the README said. The commands used to prefix the *agent's own* process
with `SHOGYM_FEEDBACK`, the identity and the task list, so an agent with its built-in tools could
read its own assignment out of its environment, and the record it could then find under the
working directory holds the receipt the placebo arm exists to withhold. And they named a
`serve.py` by a relative path that resolves from no single directory: from the repository root the
config argument was valid and the server path was not, and from the quickstart directory it was
the other way around.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
README = REPO / "src" / "shogym" / "envs" / "appworld" / "README.md"

#: The members of the identity and the treatment the paired configuration is made of. Every one of
#: them belongs to the server, and none of them may appear in the agent's own process.
CONFIGURED = (
    "SHOGYM_ENV",
    "SHOGYM_TASKS",
    "SHOGYM_FEEDBACK",
    "SHOGYM_IDENTITY",
    "SHOGYM_DEADLINE",
    "SHOGYM_IN_FLIGHT",
    "SHOGYM_RUNS",
)

#: What the smoke test below serves instead of `appworld`. The question it asks is whether the
#: documented command resolves the server from the directory the agent is launched in, which is a
#: question about the argv and not about the env: this one needs no extra, no key and no download,
#: so the check stays offline. The appworld config's own contents are checked above, and appworld
#: episodes are served in `test_appworld_served.py`.
OFFLINE_ENV = "wordle_v1"


def _paired_launch() -> str:
    """The shell the README documents for the pair, lifted out of it verbatim.

    Every ``bash`` block under the paired-arms heading, in order, which is one block that writes
    the arm's configuration and one that launches the two agents. Read line by line rather than
    with one expression over the whole section, because the blocks themselves hold shell comments
    that a heading pattern matches."""
    lines = README.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("### The paired arms"))
    shell: List[str] = []
    fence = ""
    for line in lines[start + 1 :]:
        if line.startswith("```"):
            fence = "" if fence else line[3:].strip()
            continue
        if not fence and re.match(r"^#{1,3} ", line):
            break
        if fence == "bash":
            shell.append(line)
    documented = "\n".join(shell)
    # Every line of it is run below, so a third block under this heading is a third thing this
    # would execute. Two launches and no more is the shape the section documents.
    assert documented.count("claude -p") == 2, "the paired arms are two launches of shell"
    return documented


def _shim(recording: Path) -> Path:
    """A ``claude`` that records the process it was launched as, instead of being one.

    It writes down the whole of what the agent's side of the launch is: its arguments, its
    environment, the directory it was started in, and the config file it was pointed at, which is
    the server's side and is what the two arms are allowed to differ in."""
    shim = recording.parent / "bin" / "claude"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"recording = Path({str(recording)!r})\n"
        "config = Path(sys.argv[sys.argv.index('--mcp-config') + 1])\n"
        "launch = {\n"
        "    'argv': sys.argv[1:],\n"
        "    'env': dict(os.environ),\n"
        "    'cwd': os.getcwd(),\n"
        "    'config': json.loads(config.read_text()),\n"
        "}\n"
        "seen = len(list(recording.iterdir()))\n"
        "(recording / f'{seen}.json').write_text(json.dumps(launch))\n"
    )
    shim.chmod(0o755)
    return shim


def _launches(tmp_path: Path) -> List[Dict[str, Any]]:
    """Run the documented shell, and return what each ``claude`` in it was launched as.

    ``HOME`` is the temporary tree, because the documented pair directory is under it, and the
    inherited environment is stripped of everything this port reads so that the launch is the
    README's rather than the shell this suite happens to be running in."""
    recording = tmp_path / "launches"
    recording.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir(parents=True)
    shim = _shim(recording)
    inherited = {name: value for name, value in os.environ.items() if not name.startswith("SHOGYM_")}
    # The documented shell, and one no-op after it. `bash -c` execs its *last* command in place
    # instead of forking one more process for it, which leaves the second arm's `SHLVL` one lower
    # than the first's: a difference the shell makes between two launches that are the same
    # command, and one an operator typing them at a prompt never sees.
    subprocess.run(
        ["bash", "-c", _paired_launch() + "\n:\n"],
        # The repository root, which is where the one relative line in the block reads from. Every
        # path the launch itself uses is absolute, which is the property the smoke test checks.
        cwd=str(REPO),
        env={**inherited, "HOME": str(home), "PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}"},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return [
        json.loads(written.read_text())
        for written in sorted(recording.iterdir(), key=lambda path: int(path.stem))
    ]


def documented_arms(tmp_path: Path) -> Dict[str, Dict[str, str]]:
    """The MCP child's environment the README's own commands write, one entry per arm.

    Exported because the episodes those arms produce are served in ``test_appworld_served.py``,
    and a launch test that checks the configuration while the episode test invents its own is two
    documents rather than one."""
    return {
        launch["config"]["mcpServers"]["shogym"]["env"]["SHOGYM_FEEDBACK"]: launch["config"][
            "mcpServers"
        ]["shogym"]["env"]
        for launch in _launches(tmp_path)
    }


def test_the_two_arms_launch_the_agent_as_the_same_process(tmp_path: Path) -> None:
    """The treatment and the control have to be one process apart, and they used to be two.

    The commands set `SHOGYM_FEEDBACK`, the task list and the identity on the outer `claude`
    process, which put the assignment, the run's fingerprint and the pulse in the agent's own
    environment. The quickstart says in as many words that Claude Code keeps its built-in tools
    unless a deny list is added, and the paired commands added none, so the agent could read its
    arm before it was treated and could reach the record, which holds both payloads on every row
    whichever arm answered the terminal.

    So the arm is the server's configuration now, and this compares the two launches on the agent's
    side of that line: the arguments, the environment and the working directory, byte for byte."""
    launches = _launches(tmp_path)
    assert len(launches) == 2, "one launch per arm"
    treatment, control = launches

    assert treatment["argv"] == control["argv"]
    assert treatment["env"] == control["env"]
    assert treatment["cwd"] == control["cwd"]

    for launch in launches:
        # Not merely equal to each other: neither carries any of it at all.
        assert set(CONFIGURED).isdisjoint(launch["env"])
        agent_side = json.dumps([launch["argv"], launch["env"]])
        for named in ("information", "placebo", "SHOGYM_"):
            assert named not in agent_side, f"the agent's process names {named!r}"

    # And the built-ins that would read the config, the corpus or the record are taken away.
    denied = set(treatment["argv"][treatment["argv"].index("--disallowedTools") + 1].split(","))
    assert {"Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch"} <= denied


def test_the_arm_and_the_record_reach_the_server_and_nothing_else(tmp_path: Path) -> None:
    """The other side of the same line: what the two arms are allowed to differ in.

    The regime, the queue, the identity, the deadline and the capacity travel in the MCP config's
    `env` block, which is the child's environment and nobody else's, and the two arms' blocks
    differ in the regime alone, because every other member is what makes two runs one measurement.
    The provenance directory goes outside the directory the agent is launched into, since it names
    the regime in `claim.json` before the first dispense and keeps every payload the env published
    afterwards."""
    launches = _launches(tmp_path)
    arms = [launch["config"]["mcpServers"]["shogym"]["env"] for launch in launches]

    assert [arm["SHOGYM_FEEDBACK"] for arm in arms] == ["information", "placebo"]
    assert arms[0] == {**arms[1], "SHOGYM_FEEDBACK": "information"}, "one difference, and it is the arm"
    for arm in arms:
        assert set(CONFIGURED) <= set(arm)
        assert arm["SHOGYM_ENV"] == "appworld"

    working = Path(launches[0]["cwd"]).resolve()
    records = Path(arms[0]["SHOGYM_RUNS"]).resolve()
    assert records.is_absolute() and not records.is_relative_to(working)


async def test_the_documented_config_starts_the_server_from_any_directory(tmp_path: Path) -> None:
    """The config the commands name, spawned as an MCP server, from a directory that is neither.

    The commands used to be written for two directories at once. They passed
    `examples/claude_code/.mcp.json` and `examples/claude_code/PROMPT.txt`, which resolve from the
    repository root, and that config spawned `uv run python serve.py`, which resolves from the
    directory the agent was launched in and named nothing at the root. Claude Code resolves a
    relative command argument in an MCP config against the launch directory rather than against
    the config file's own directory, so one of the two was always wrong. The launch test that
    existed asserted the JSON argument and then bypassed it by importing the module, which is the
    one thing that cannot fail on a path.

    So this starts the documented command itself, from a temporary directory that is not the
    repository, not the quickstart and not the pair, and speaks MCP to what comes up."""
    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    server = _launches(tmp_path)[0]["config"]["mcpServers"]["shogym"]
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    # The config's block is added to the environment the harness already has, which is how Claude
    # Code spawns it. The two overrides keep this offline and keep the record in the temporary
    # tree (see `OFFLINE_ENV`).
    environment = {
        **os.environ,
        **server["env"],
        "SHOGYM_ENV": OFFLINE_ENV,
        "SHOGYM_RUNS": str(tmp_path / "runs"),
    }
    transport = StdioTransport(
        server["command"], server["args"], env=environment, cwd=str(elsewhere)
    )
    async with Client(transport) as client:
        published = {tool.name for tool in await client.list_tools()}
    assert "get_task" in published, "the queue's own tool, so the stream really came up"
