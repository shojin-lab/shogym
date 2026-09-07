"""What the agent can read of who is serving it, which is meant to be nothing.

An agent works a long session against one server, and the name that server is known by is the
front of every tool name it reads. A name that identified the platform would be a search key: an
agent with general egress that wanted to do better could look up the platform, its graders or its
source, and one recorded run named its own memory files after the platform, which is what an
agent does with a word it thinks it is learning about. So the name is neutral, and this is where
that is checked.

It is checked by building the surface rather than by reading the source. Every string swept here
comes off a served server: the name the server calls itself, every tool name, every tool
description, every schema, the environment's instructions, the bodies the plain server publishes,
and the text of a refusal the transport actually answered a call with. A platform name that came
back into any of those fails this whether it was written into a constant, appended by a wrapper or
carried in from an environment.

The word list is the platform, the two repositories it lives in, and every environment class in
this repository, derived from the package rather than typed out, because a list typed out is a
list that goes stale the day a class is renamed.

Four things are deliberately left, each because closing it is a change of a different kind, and
each stated here rather than quietly left off the list.

The environment's canonicalization version names the platform, and the generation declares it in
the acknowledgement that answers every filing, so a model that ends a task reads it. It is also
inside the document a generation hashes, so moving it refuses the resume of every run recorded
before the move. It wants its own change, with its own statement about resumes, made where the
environments declare it.

The AutomationBench ``done`` tool's description names the benchmark. A tool description is inside
that same hashed document, so it wants its own change on exactly the same terms.

The registered name of an environment is not on the word list at all. An environment that names
itself in a tool description leaks on the same terms as the two above, and the plain server
publishes the name in its task contract: the registry's word for an environment built through the
registry, which is how every measured run builds one, and the class's own name for one
constructed directly. That second case is a limit of the plain serving path rather than something
covered here.

The plain server's metadata sidecar carries two keys named after the platform. Nothing swept here
covers them: they are a wire contract between that server and a client that reads metadata, the
stream gateway sends no metadata at all, and no measured run reads them.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

pytest.importorskip("temporalio")

from fastmcp import Client, FastMCP  # noqa: E402

from examples.automationbench_cell import cell as launcher  # noqa: E402
from examples.automationbench_cell import pinned  # noqa: E402
from examples.automationbench_cell import sandbox  # noqa: E402
from examples.automationbench_cell import serve as cell  # noqa: E402
from shogym.core import Env  # noqa: E402
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    CANONICALIZATION_VERSION,
    PULL_TOOL,
    StreamGateway,
    _refusal,
    build_gateway_server,
    served_manifest,
    stream_start,
    terminal_manifest,
)
from shogym.serve.protocol_v2.kernel.messages import configuration_hash  # noqa: E402
from shogym.serve.protocol_v2.records import (  # noqa: E402
    PROTOCOL_ERROR_CODES,
    SealAck,
    visible_bytes,
)
from shogym.serve.server import SERVED_NAME, TASK_RESOURCE, build_server  # noqa: E402

from tests._fixtures.receipts_bundle import verified_bundle  # noqa: E402
from tests._fixtures.upstream_gate import gate  # noqa: E402

# The env most tests here serve over. It needs no extra, no key and no download, so the sweep
# runs offline, and what it is does not matter: the strings under audit belong to the serving
# layer and reach the model whatever is behind them.
TEST_ENV = "wordle_v1"

#: One self-contained AutomationBench task, which is how that environment is served without its
#: domain datasets. What it asks for does not matter: nothing here files anything and nothing is
#: scored, and what is under test is the text the generation hands the agent before any work.
OFFLINE_TASK: Dict[str, Any] = {
    "prompt": [{"role": "user", "content": "Do nothing."}],
    "info": {"zapier_tools": [], "initial_state": {}, "assertions": []},
}


def env_classes() -> Tuple[str, ...]:
    """The name of every environment class this repository defines.

    Derived rather than listed. A list written out here would have been right on the day it was
    written and wrong the day a class was renamed or added, and a sweep whose word list has gone
    stale passes over exactly the name it was meant to catch. Importing the package is what
    defines them, and the module a class was defined in is what says it is this repository's
    environment rather than a fixture some other test file registered.
    """
    import shogym.envs  # noqa: F401  importing the package defines and registers every one

    found: Dict[str, None] = {}
    pending: List[type] = list(Env.__subclasses__())
    while pending:
        subclass = pending.pop()
        pending.extend(subclass.__subclasses__())
        if subclass.__module__.startswith("shogym.envs"):
            found[subclass.__name__] = None
    return tuple(sorted(found))


#: What a string the model can read may not carry: the platform, the two repositories it lives
#: in, and the class name of every environment served through it. A word here is a word an agent
#: could search for, which is the whole of why the list is this list.
FORBIDDEN: Tuple[str, ...] = ("shogym", "hgym", "shobench") + env_classes()


def carried(text: str) -> List[str]:
    """Every forbidden word in one string, matched however it was capitalized."""
    lowered = text.lower()
    return [word for word in FORBIDDEN if word.lower() in lowered]


def strings(where: str, value: Any) -> Iterator[Tuple[str, str]]:
    """Every string anywhere in a structure, with the path it was found at.

    Keys as well as values: a name is as readable as the thing it names, and the server key in a
    harness config is exactly a key.
    """
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield f"{where}<key>", key
            yield from strings(f"{where}.{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from strings(f"{where}[{index}]", item)


def offences(where: str, value: Any) -> List[str]:
    """One line per forbidden word in a structure, empty when there are none."""
    return [
        f"{path} carries {word!r}: {' '.join(text.split())[:120]}"
        for path, text in strings(where, value)
        for word in carried(text)
    ]


def gateway_over(episode: ServedEpisode, name: Optional[str] = None) -> FastMCP:
    """The stream gateway's own server, over a real episode, as a launcher builds it.

    Built from the episode rather than from constants, which is what makes the tests below an
    audit rather than a reading of the source: the descriptions are assembled from a control
    tool, a wrapper note and an environment's own manifest, and only the assembled text says
    what the model gets. No stream stands behind it, so the generation composed here is never
    worked; the one call below that would reach a stream is refused before it gets there.
    """
    spec = episode.describe()
    terminal = terminal_manifest(spec)
    gateway = StreamGateway(
        None,  # type: ignore[arg-type]  # listing tools and refusing a call reach no stream
        episode,
        spec,
        terminal,
        initial_cursor="0" * 32,
        generation=stream_start(
            spec,
            terminal,
            claim_hash=sha256(b"a claim").hexdigest(),
            evaluation_only=True,
            info=True,
        ),
    )
    return build_gateway_server(gateway, name=name)


async def gateway_surface(
    name: Optional[str] = None,
    env_name: str = TEST_ENV,
    env_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything a model connected to a stream gateway can read."""
    episode = await ServedEpisode.start(
        env_name, task=0, ends_on_horizon=False, env_config=env_config
    )
    try:
        server = gateway_over(episode, name)
        async with Client(server) as client:
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "schema": tool.inputSchema,
                }
                for tool in await client.list_tools()
            ]
        return {
            "server_name": server.name,
            "tools": tools,
            "instructions": episode.describe().instructions,
        }
    finally:
        await episode.close()


async def plain_surface(env_name: str = TEST_ENV) -> Dict[str, Any]:
    """Everything a model connected to the plain server can read, bodies included.

    The bodies are the point. A listing says what the tools are called; the task contract is a
    document this server publishes twice, once as the ``describe`` tool and once as a resource,
    and it is the one place this path hands the model a whole document rather than a name.
    """
    episode = await ServedEpisode.start(env_name, task=0)
    try:
        server = build_server(episode)
        async with Client(server) as client:
            listed = await client.list_tools()
            resources = [str(found.uri) for found in await client.list_resources()]
            described = (await client.call_tool("describe", {})).data
            published = json.loads((await client.read_resource(TASK_RESOURCE))[0].text)
        return {
            "server_name": server.name,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "schema": tool.inputSchema,
                }
                for tool in listed
            ],
            "resources": resources,
            "describe": described,
            "task": published,
        }
    finally:
        await episode.close()


async def test_the_gateway_serves_itself_under_a_name_that_says_nothing() -> None:
    """The default is one word, and it is not built out of the environment behind it.

    A harness puts this word in front of every tool name it hands the model, so a default taken
    from the episode would tell the model which environment it was in on every call it read.
    """
    surface = await gateway_surface()
    assert surface["server_name"] == SERVED_NAME == "stream"
    assert [tool["name"] for tool in surface["tools"]] == [PULL_TOOL, "info", "terminate", "guess"]


async def test_a_launcher_that_wants_another_name_serves_under_it() -> None:
    """The name is the launcher's to set, and setting it changes nothing else.

    A run that serves under its own word is a run whose agent read that word, which is why the
    launcher records what it passed. What it may not change is the surface underneath: the same
    tools, in the same order, under the same descriptions.
    """
    default = await gateway_surface()
    custom = await gateway_surface(name="a-different-word")
    assert custom["server_name"] == "a-different-word"
    assert custom["tools"] == default["tools"]


async def test_the_plain_server_serves_itself_and_its_task_resource_under_that_name() -> None:
    """The other serving path, which publishes a resource as well as tools.

    A URI is text a model can read, so the scheme is the served name too: a resource addressed at
    the platform would say in an address what the tool names no longer say.
    """
    surface = await plain_surface()
    assert surface["server_name"] == SERVED_NAME
    assert TASK_RESOURCE == "stream://task"
    assert surface["resources"] == [TASK_RESOURCE]


async def test_no_platform_name_reaches_the_model_through_the_gateway() -> None:
    """The audit itself: every served string, swept for every forbidden word.

    This is the test that fails when a name comes back. It reads the surface rather than the
    source, so it catches a name however it arrived: written into a control tool's description,
    appended by the wrapper, carried in from an environment's own manifest, or put there by a
    launcher that named its server after the platform.
    """
    for name in (None, "a-different-word"):
        assert offences("served", await gateway_surface(name)) == []


async def test_no_platform_name_reaches_the_model_from_the_receipts_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The same sweep over the environment whose first pilot has not run yet.

    An environment brings its own instructions, its own terminal and its own tool descriptions
    into the served surface, so the audit has to be over more than one of them: a name written
    into an environment reaches the model as surely as one written into the gateway. What is
    swept is the environment as it is served, over a bundle built here rather than a stub.
    """
    bundle = verified_bundle(tmp_path_factory.mktemp("bundles"), size=1)
    surface = await gateway_surface(env_name="receipts_v1", env_config={"bundle": str(bundle)})
    assert [tool["name"] for tool in surface["tools"]] == [PULL_TOOL, "info", "submit_filing"]
    assert offences("receipts", surface) == []


async def test_no_platform_name_reaches_the_model_from_the_automationbench_environment() -> None:
    """The same sweep over the environment the measured cell is actually over.

    It is the one whose descriptions a model has already read two hundred tasks of, so a sweep
    that covered the other two and inferred this one would be inferring the answer for the only
    environment the question has been asked about. It is served from the one task written out
    above rather than from the domain datasets, so no task dataset is fetched and nothing is
    scored; the pinned upstream source itself must already be provisioned, and the import gate
    skips this test on a machine that has neither it nor the network to fetch it.

    The benchmark's own name is not on the word list, and the ``done`` tool's description does
    name it. That is the leak this module states and does not close, because the description is
    inside the document a generation hashes.
    """
    gate("shogym.envs.automationbench.adapter", package="automationbench", extra="automationbench")
    surface = await gateway_surface(
        env_name=cell.ENV, env_config={"tasks": [OFFLINE_TASK], "max_steps": 50}
    )
    assert [tool["name"] for tool in surface["tools"]] == [
        PULL_TOOL,
        "info",
        "api_search",
        "api_fetch",
        "base64_encode",
        "done",
    ]
    assert offences("automationbench", surface) == []


async def test_no_platform_name_reaches_the_model_through_the_plain_server() -> None:
    """The same sweep over the other serving path, bodies and resource included.

    What this path adds is a document: the task contract, published as the ``describe`` tool and
    again as a resource, and swept here as the bodies those two answer with rather than as their
    names. The contract carries the environment's own name, which is the registry's word for an
    environment built the way every measured run builds one. An environment constructed directly
    carries its class name there instead, which is a limit of this path and is stated in this
    module rather than swept.
    """
    assert offences("plain", await plain_surface()) == []


async def test_no_platform_name_reaches_the_model_in_a_refusal() -> None:
    """A refusal is text the model reads, and it is swept as the transport answered it.

    The call below is a real call to a real served tool, carrying a native object that tool's own
    schema refuses, and what is swept is the content the client was handed back. So a name
    appended where a refusal is built is a name this test sees: nothing here re-encodes the
    record for itself and then checks its own encoding.

    One code is reachable without a stream behind the transport, so the other fourteen are swept
    through the same function that built the one above. The whole body is checked rather than the
    code alone, because a refusal that grew an explanatory sentence naming the platform would
    carry the name there.
    """
    episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
    try:
        async with Client(gateway_over(episode)) as client:
            refused = await client.call_tool(
                "guess",
                {"attempt_id": "0" * 32, "arguments": {"word": 5}},
                raise_on_error=False,
            )
    finally:
        await episode.close()
    assert refused.is_error
    answered = [getattr(item, "text", "") for item in refused.content]
    assert offences("refusal", answered) == []
    assert json.loads(answered[0])["kind"] == "protocol_error"
    bodies = {code: str(_refusal(code)) for code in sorted(PROTOCOL_ERROR_CODES)}
    assert len(bodies) == 15
    assert offences("refusals", bodies) == []


def test_no_platform_name_reaches_the_model_through_what_the_launcher_writes(
    tmp_path: Path,
) -> None:
    """What a launch hands the agent, taken from the writers rather than rebuilt here.

    The config file is read back as the bytes that were written, because the key is the front of
    every tool name and the URL names the container that answers on the private network. The
    command is the one the launcher builds, standing instruction and opening turn inside it,
    because a process can read its own arguments. The environment is the one the container is
    given. A dictionary assembled in this test would be a check on this test.
    """
    url = sandbox.gateway_url(sandbox.names("a-token")[1])
    written = launcher.mcp_config(tmp_path, url=url)
    prompt = (Path(launcher.HERE) / "PROMPT.txt").read_text(encoding="utf-8")
    argv = launcher.claude_argv(
        Path(sandbox.CONFIG_MOUNT) / written.name,
        model=launcher.MODEL,
        effort=launcher.EFFORT,
        system_prompt=prompt,
        session_id="0" * 32,
    )
    handed = {
        "config": written.read_text(encoding="utf-8"),
        "config_path": str(Path(sandbox.CONFIG_MOUNT) / written.name),
        "argv": argv,
        "kickoff": launcher.KICKOFF,
        "environment": pinned.agent_environment({}),
        "served_tools": list(cell.SERVED_TOOLS),
    }
    assert offences("cell", handed) == []
    # And the file really is the one under audit: the key, the endpoint and the prompt are in it.
    assert list(json.loads(handed["config"])["mcpServers"]) == [SERVED_NAME]
    assert url in handed["config"] and prompt in argv and launcher.KICKOFF in argv
    assert cell.SERVER == SERVED_NAME
    assert cell.SERVED_PREFIX == f"mcp__{SERVED_NAME}__"
    assert url == f"http://{SERVED_NAME}-a-token:{sandbox.SERVER_PORT}/mcp"


def test_the_cell_serves_every_tool_under_the_one_name(tmp_path: Path) -> None:
    """The file the agent reads and the names the launch pins agree, down to the prefix.

    What the launch then writes into its own record is checked where the record is written, over
    a launch that produced one.
    """
    config = json.loads(launcher.mcp_config(tmp_path, url="http://somewhere:1/mcp").read_text())
    assert list(config["mcpServers"]) == [cell.SERVER]
    assert all(name.startswith(cell.SERVED_PREFIX) for name in cell.SERVED_TOOLS)


def test_the_word_list_names_every_environment_class_in_this_repository() -> None:
    """The sweep is only as good as its word list, so the list is checked before it is used.

    The class an audited environment is implemented by is the one that matters, and it is not
    always the class it derives from. A list naming a plausible word instead of the real ones
    would sweep every surface here and find nothing, whatever those surfaces held.
    """
    classes = env_classes()
    assert {"WordleV1Env", "WordleV1Default", "AutomationBenchEnv", "ReceiptsV1Env"} <= set(classes)
    assert carried("WordleV1Default") == ["WordleV1Default"]
    assert carried("WordleV1Env") == ["WordleV1Env"]
    assert carried("the wordlev1default was here") == ["WordleV1Default"]
    # The registered name of an environment is not on the list, deliberately.
    assert carried("wordle_v1") == [] and carried(cell.ENV) == []


async def test_the_configuration_document_is_hashed_and_one_of_its_versions_is_served() -> None:
    """Why two platform-named version strings stay, and which of them the model reads.

    Both go into the document a generation hashes. That document is what a resume is held to, so
    moving either string moves the hash and refuses every run recorded before the move, over a
    rule that had not changed.

    There the two part. The wrapper version is identity and nothing else: it is in the document
    and in nothing the model can read. The canonicalization version is in the document and also
    in the acknowledgement the stream answers a filing with, which is a record the model reads at
    the end of every task it finishes. So it is a served platform name, it is not fixed here, and
    it is named as a leak rather than left off the list. It wants its own change, with its own
    statement about resumes, made where the environments declare it.
    """
    episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
    try:
        spec = episode.describe()
        terminal = terminal_manifest(spec)
        document = served_manifest(spec, terminal, info=True)
        generation = stream_start(
            spec, terminal, claim_hash=sha256(b"a claim").hexdigest(), evaluation_only=True
        )
    finally:
        await episode.close()
    assert carried(document["wrapper_version"]) and carried(document["canonicalization_version"])
    # The wrapper version is in nothing the model can read.
    readable = [text for _, text in strings("served", await gateway_surface())]
    assert not any(document["wrapper_version"] in text for text in readable)
    # The canonicalization version is in the acknowledgement, which is bytes the model is handed.
    acknowledgement = visible_bytes(
        SealAck(
            message_id="0" * 32,
            attempt_id="1" * 32,
            submission_digest="d" * 64,
            canonicalization_version=generation.canonicalization_version,
        )
    ).decode("utf-8")
    assert generation.canonicalization_version == CANONICALIZATION_VERSION
    assert carried(acknowledgement) == ["shogym"]
    # And moving it is a changed generation, which is why it is not moved here.
    moved = replace(generation, canonicalization_version="stream.gateway.1")
    assert configuration_hash(moved) != configuration_hash(generation)
