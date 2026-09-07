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

The two strings this sweep named as left have moved, and a third it had not been looking at moved
with them. The two named were the canonicalization version each environment declares and the
AutomationBench ``done`` tool's description. The third was a sentence in YC-Bench's ``submit``
description saying the platform sealed the episode, and this file did not see it because it swept
three environments and YC-Bench was not one of them. That is the reason for the shape of what is
here now: every environment this file can serve is swept rather than the three it started with,
the acknowledgement that answers a filing is read off a real seal rather than assembled here, and
the refusal the cut costs has a test of its own, per environment.

What this file claims is what it sweeps, and three things sit outside that. Each is stated here
rather than quietly left off the list.

Two environments are not swept, because neither can be served where this file runs.
``frontier_bench`` builds its task's image when a session begins, so its surface needs a
container runtime. ``tau2`` reads task data provisioned from upstream into a cache and has no
injectable stand-in for it. Both are served through the same gateway as the rest, so a name
written into either would reach a model exactly as one written into the others would, and
nothing here would see it.

The registered name of an environment is not on the word list at all. An environment names itself
in its own task text, which is content rather than a fact about who is serving it, and the plain
server publishes the name in its task contract: the registry's word for an environment built
through the registry, which is how every measured run builds one, and the class's own name for one
constructed directly. That second case is a limit of the plain serving path rather than something
covered here. What a task's own text says is content on the same terms, and it is swept for the
tasks served here rather than established for every task an environment could load.

The plain server's metadata sidecar carries two keys named after the platform. Nothing swept here
covers them: they are a wire contract between that server and a client that reads metadata, the
stream gateway sends no metadata at all, and no measured run reads them.
"""

from __future__ import annotations

import json
import pkgutil
from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

pytest.importorskip("temporalio")

from fastmcp import Client, FastMCP  # noqa: E402

from examples.automationbench_cell import cell as launcher  # noqa: E402
from examples.automationbench_cell import pinned  # noqa: E402
from examples.automationbench_cell import sandbox  # noqa: E402
from examples.automationbench_cell import serve as cell  # noqa: E402
from shogym.core import Env  # noqa: E402
from shogym.envs.browsecomp_plus.searcher import InMemorySearcher  # noqa: E402
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import canonical_json  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    CANONICALIZATION_VERSION,
    PULL_TOOL,
    StreamGateway,
    _configuration_hash,
    _refusal,
    build_gateway_server,
    environment_terminal,
    open_gateway,
    served_manifest,
    stream_start,
    terminal_manifest,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    STREAM_TASK_QUEUE,
    OfferedMessage,
    durable_client,
    resume_run_directory,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel.messages import configuration_hash  # noqa: E402
from shogym.serve.protocol_v2.records import (  # noqa: E402
    PROTOCOL_ERROR_CODES,
    SealAck,
    Task,
    visible_bytes,
)
from shogym.serve.protocol_v2.rundir import (  # noqa: E402
    ResumeRefused,
    create_run_directory,
)
from shogym.serve.server import SERVED_NAME, TASK_RESOURCE, build_server  # noqa: E402

from tests._fixtures.receipts_bundle import verified_bundle  # noqa: E402
from tests._fixtures.upstream_gate import gate  # noqa: E402

# The env most tests here serve over. It needs no extra, no key and no download, so the sweep
# runs offline, and what it is does not matter: the strings under audit belong to the serving
# layer and reach the model whatever is behind them.
TEST_ENV = "wordle_v1"

#: One self-contained AutomationBench task, which is how that environment is served without its
#: domain datasets. What it asks for does not matter: what is under test is the text the
#: generation hands the agent, and the number a filing over it earns is nobody's measurement.
OFFLINE_TASK: Dict[str, Any] = {
    "prompt": [{"role": "user", "content": "Do nothing."}],
    "info": {"zapier_tools": [], "initial_state": {}, "assertions": []},
}

#: One question, for the environment that answers questions. It is written here rather than
#: loaded, so this sweep needs no gated dataset and no network.
OFFLINE_QUESTION: Dict[str, Any] = {
    "id": "q_capital",
    "question": "What is the capital of France?",
    "answer": "Paris",
    "answer_type": "exactMatch",
}

#: One query and the two documents that answer it, for the environment that searches. Written
#: here for the reason the question above is, and the searcher is the in-memory one so no index
#: is built and no corpus is fetched.
OFFLINE_QUERY: Dict[str, Any] = {
    "query_id": "q_capital",
    "query": "What is the capital of France?",
    "answer": "Paris",
    "qrel_gold": ["1"],
    "qrel_evidence": ["1"],
}
OFFLINE_CORPUS: Dict[str, str] = {
    "1": "Paris is the capital of France.",
    "9": "Lyon is not.",
}


class UncalledJudge:
    """The judge of an environment nobody grades here, which is every one of them.

    Both question-answering environments build a judge when a session begins, and the default
    one wants a key. This one is injected in its place so the surface can be served offline,
    and it says so rather than answering: a sweep that reached a judge would be measuring an
    answer instead of reading the text an agent is handed.
    """

    def __call__(self, **_: Any) -> Any:
        raise AssertionError("nothing swept here is graded, so no judge is asked anything")


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

    The benchmark's own name is not on the word list, because an environment's name is content.
    What is checked instead is the one sentence that used to say more than the name: the tool an
    agent files with said its scoring was the benchmark's own rubric, which told a model it was
    being scored by a benchmark and gave it the word to look the benchmark up by. The sentence
    says what the tool does now, and this is where it is held to that.
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
    [done] = [tool for tool in surface["tools"] if tool["name"] == "done"]
    assert "rubric" in done["description"] and "benchmark" not in done["description"].lower()
    assert "automationbench" not in done["description"].lower()


async def test_no_platform_name_reaches_the_model_from_the_yc_bench_environment() -> None:
    """The same sweep over the environment whose terminal said who was sealing it.

    Its ``submit`` description said the platform sealed the episode, and ``submit`` is the tool a
    run over this environment is filed with, so the word was in the tool list of every session
    over it. That is a fact about who is serving rather than about the company being run, and it
    is the one this cut moved last. The sentence names the harness now.

    The upstream source is provisioned at runtime rather than installed, so the import gate skips
    this test on a machine that has neither it nor the network to fetch it. The sim is seeded in
    process from the task's own seed, so serving the surface needs nothing else.
    """
    gate("shogym.envs.yc_bench.adapter", package="yc_bench", extra="yc_bench")
    surface = await gateway_surface(env_name="yc_bench")
    assert [tool["name"] for tool in surface["tools"]] == [
        PULL_TOOL,
        "info",
        "run_command",
        "submit",
    ]
    assert offences("yc_bench", surface) == []
    [submit] = [tool for tool in surface["tools"] if tool["name"] == "submit"]
    assert "The harness seals the episode" in " ".join(submit["description"].split())


async def test_no_platform_name_reaches_the_model_from_the_hle_environment() -> None:
    """The same sweep over an environment that brings no terminal of its own.

    The three above each seal and grade themselves, and an environment that does neither is
    served by the same gateway with the kernel's stand-ins behind it. Its surface is assembled
    the same way, so a name in it would reach a model the same way, and the sweep says so rather
    than inferring it from the environments that do.

    It is served from the one question written out above, so the gated dataset is neither
    fetched nor needed, and the judge is the stand-in that refuses to be asked.
    """
    pytest.importorskip("datasets", reason="hle extra not installed")
    surface = await gateway_surface(
        env_name="hle",
        env_config={"tasks": [OFFLINE_QUESTION], "judge": UncalledJudge()},
    )
    assert [tool["name"] for tool in surface["tools"]] == [PULL_TOOL, "info", "submit_answer"]
    assert offences("hle", surface) == []


async def test_no_platform_name_reaches_the_model_from_the_browsecomp_plus_environment() -> None:
    """The same sweep over the other environment the kernel's stand-ins seal for.

    This one serves retrieval tools as well as a terminal, which is the surface an environment
    with more than one thing to say puts in front of a model. Everything it needs is injected:
    the query, the corpus, the searcher and the judge, so no index is built and no corpus is
    downloaded.
    """
    surface = await gateway_surface(
        env_name="browsecomp_plus",
        env_config={
            "tasks": [OFFLINE_QUERY],
            "searcher": InMemorySearcher(OFFLINE_CORPUS),
            "judge": UncalledJudge(),
        },
    )
    assert [tool["name"] for tool in surface["tools"]] == [
        PULL_TOOL,
        "info",
        "search",
        "get_document",
        "submit_answer",
    ]
    assert offences("browsecomp_plus", surface) == []


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

    One code is reachable without a stream behind the transport, so the other fifteen are swept
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
    assert len(bodies) == 16
    # The newest of them is the one a delivery refuses with when the bytes it depends on are gone.
    # It is a bare token and this is where that is swept like every other refusal the model reads.
    assert "evidence_unavailable" in bodies
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


# ----- the version a filing is acknowledged under -----

ATTEMPT = "0" * 31 + "1"
TASK_ID = "0" * 31 + "2"
ACK_ID = "0" * 31 + "3"
CURSOR = "0" * 32

#: The canonicalization versions this repository declared before the cut. They are kept because
#: the configuration hash of every run recorded under them was taken over them, so they are what
#: such a resume presents and what it is now refused for.
RETIRED_VERSIONS: Dict[str, str] = {
    "automationbench": "shogym.automationbench.1",
    "gateway": "shogym.gateway.1",
    "receipts": "shogym.receipts.1",
    "wordle": "shogym.wordle.1",
    "yc_bench": "shogym.yc_bench.1",
}


def declared_versions() -> Dict[str, str]:
    """The canonicalization version of every environment here, and of the gateway's stand-in.

    Derived rather than listed, for the reason the word list is: a version written out here would
    be right on the day it was written and silent the day an environment was added under a name
    that named the platform again. An environment whose attempts end the plain way declares none,
    and there is nothing to sweep for it. That is the only import failure passed over: an
    environment whose port is there and will not load is raised rather than quietly skipped, or a
    machine missing one optional package would sweep an empty list and report no leak.
    """
    envs = import_module("shogym.envs")
    found: Dict[str, str] = {"gateway": CANONICALIZATION_VERSION}
    for module in pkgutil.iter_modules(envs.__path__):
        if not module.ispkg:
            continue
        port_name = f"shogym.envs.{module.name}.protocol_v2"
        try:
            port = import_module(port_name)
        except ModuleNotFoundError as missing:
            if missing.name != port_name:
                raise
            continue
        version = getattr(port, "CANONICALIZATION_VERSION", None)
        if isinstance(version, str):
            found[module.name] = version
    return found


class OneAttemptStream:
    """A stream that offers one task, grants the calls that work it, and answers one filing.

    It stands in for the durable stream so the sweep below stays offline, and what it is faithful
    about is the transport: the bytes a pull, a call and a filing are answered with are encoded,
    routed and handed over by the real gateway. It is not faithful about the seal. The
    acknowledgement is minted here from the version it was built with rather than by the kernel,
    so a sweep through it says what a record carrying that version reads like and says nothing
    about which version the kernel would have put there. That second question is the real stream's
    to answer, and it is asked of a real stream further down.
    """

    def __init__(self, version: str) -> None:
        self.version = version
        self.cursor = CURSOR
        self.attempts: Dict[str, str] = {}
        self.environment_calls: Dict[str, int] = {}
        self.held: Optional[str] = None

    async def pull(self, request: Any) -> OfferedMessage:
        task = Task(message_id=TASK_ID, attempt_id=ATTEMPT, body="Guess the word.")
        return OfferedMessage(
            message_id=TASK_ID,
            kind="task",
            visible_text=visible_bytes(task).decode("utf-8"),
            attempt_id=ATTEMPT,
        )

    async def seal(self, request: Any) -> OfferedMessage:
        ack = SealAck(
            message_id=ACK_ID,
            attempt_id=request.metadata.attempt_id,
            submission_digest="d" * 64,
            canonicalization_version=self.version,
        )
        return OfferedMessage(
            message_id=ACK_ID,
            kind="seal_ack",
            visible_text=visible_bytes(ack).decode("utf-8"),
            attempt_id=ack.attempt_id,
        )

    async def present(self, message: Any, **blobs: Any) -> Any:
        return SimpleNamespace(cursor=message.message_id)

    async def commit_presentation(self, commit: Any) -> Any:
        self.cursor = commit.message_id
        if commit.task_start_checkpoint_blob is not None:
            self.attempts[ATTEMPT] = "active"
            self.environment_calls[ATTEMPT] = 0
        if commit.message_id == ACK_ID:
            self.attempts[ATTEMPT] = "ack_presented"
        return SimpleNamespace(cursor=commit.message_id)

    async def begin_environment_call(self, call: Any) -> Any:
        self.held = call.call_id
        spent = self.environment_calls.get(call.attempt_id, 0)
        self.environment_calls[call.attempt_id] = spent + 1
        return SimpleNamespace(call_id=call.call_id, attempt_id=call.attempt_id, held=True)

    async def end_environment_call(self, call: Any) -> Any:
        held, self.held = self.held == call.call_id, None
        return SimpleNamespace(call_id=call.call_id, attempt_id=call.attempt_id, held=held)

    async def stream_state(self) -> Any:
        return SimpleNamespace(
            cursor=self.cursor,
            generation_state="open",
            attempts=dict(self.attempts),
            environment_calls=dict(self.environment_calls),
            stream_state_sha256="0" * 64,
            pending_message_id=None,
            pending_kind=None,
        )


async def worked_and_filed(version: str) -> List[str]:
    """Every text a model is handed across one pull, one call and one filing.

    The generation is composed under ``version`` and the sequence is driven through a client,
    so what comes back is what the transport really answered rather than a record this file
    encoded for itself.
    """
    episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
    try:
        spec = episode.describe()
        terminal = terminal_manifest(spec)
        gateway = StreamGateway(
            OneAttemptStream(version),  # type: ignore[arg-type]  # the seal reaches no service
            episode,
            spec,
            terminal,
            initial_cursor=CURSOR,
            generation=stream_start(
                spec,
                terminal,
                claim_hash=sha256(b"a claim").hexdigest(),
                evaluation_only=True,
                canonicalization_version=version,
            ),
        )
        async with Client(build_gateway_server(gateway)) as client:
            answers = [
                await client.call_tool(PULL_TOOL, {}),
                await client.call_tool(
                    "guess", {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
                ),
                await client.call_tool("terminate", {"attempt_id": ATTEMPT, "arguments": {}}),
            ]
    finally:
        await episode.close()
    return [getattr(item, "text", "") for answer in answers for item in answer.content]


def test_no_canonicalization_version_declared_here_names_the_platform() -> None:
    """The versions themselves, swept as every other served string is.

    The environment's own name stays: it is content, the same word its task text uses. What is
    gone is the platform in front of it, and the number behind it is untouched, because the
    number is what says which capture rule a recorded submission was taken under.
    """
    declared = declared_versions()
    assert set(RETIRED_VERSIONS) <= set(declared)
    assert offences("versions", declared) == []
    # Each one is the version it replaced with the platform taken off the front and nothing else
    # touched, which is what keeping the number means.
    assert all(carried(retired) == ["shogym"] for retired in RETIRED_VERSIONS.values())
    assert all(
        declared[name] == retired.removeprefix("shogym.")
        for name, retired in RETIRED_VERSIONS.items()
    )


async def test_no_platform_name_reaches_the_model_in_what_the_transport_hands_over() -> None:
    """The bytes of a pull, a call and a filing, swept as the transport encoded them.

    This is the surface the tool listings above cannot reach. A listing is what a session opens
    with; a record is what it works and closes a task with, and the version is in the last of
    them because the digest of a submission means nothing without the rule it was taken under.
    So the sequence is driven rather than described: the task comes back from a pull, a guess is
    played through the wrapper, the attempt is filed, and every text handed over on the way is
    swept for every forbidden word.

    Each version any environment here declares is carried through it, because a record is text
    and the version is in that text. Which version the kernel puts there is a different question
    and this cannot answer it: the stream behind this is a double and the acknowledgement is
    minted from the version it was built with. The real stream answers that below, one case per
    environment, each served the way production serves it: under the version that environment
    declares and over the Activities that environment answered with.
    """
    for version in sorted(set(declared_versions().values())):
        handed = await worked_and_filed(version)
        assert offences(f"filed under {version}", handed) == []
        acknowledgement = json.loads(handed[-1])
        assert acknowledgement["kind"] == "seal_ack"
        assert acknowledgement["canonicalization_version"] == version


@dataclass(frozen=True)
class Filing:
    """One environment, the version it declares, and the call that ends an attempt over it.

    A row is that environment served the way production serves it: the gateway asks it how the
    attempts of a generation over it end, and what it answers is both the version the generation
    declares and the Activities the Worker registers. So the table is a table of environments,
    the sweep below is one case per row, and no row pairs a version with an environment that
    does not declare it.

    That pairing is what production makes rather than the only pairing the code can be made to
    hold. A seal through the shared grading port refuses a request whose version is not the one
    the environment captures under, and so does the receipts port, so ``automationbench``,
    ``yc_bench`` and ``receipts`` reject a mismatched pairing at the seal. The wordle port does
    not compare the request's version at all, so a caller that handed the gateway a doctored
    descriptor over a wordle episode would be answered rather than refused. Nothing here does
    that, and the pairing each row makes is the environment's own.

    ``port`` is the module that declares the version, which is the module whose constant has to
    be put back to reconstruct what a build from before the cut was answered with.
    """

    port: str
    version: str
    env_name: str
    terminal: str
    arguments: Dict[str, Any]


#: Every version any environment here declares, with the environment that declares it and the
#: filing that reaches its seal. ``browsecomp_plus`` brings no terminal of its own, so it is the
#: one served under the gateway's stand-in, which is what that constant is for.
FILINGS: Tuple[Filing, ...] = (
    Filing("gateway", "gateway.1", "browsecomp_plus", "submit_answer", {"answer": "Paris"}),
    Filing("wordle", "wordle.1", TEST_ENV, "terminate", {}),
    Filing(
        "receipts", "receipts.1", "receipts_v1", "submit_filing", {"filing": "record-1,Nothing"}
    ),
    Filing("automationbench", "automationbench.1", cell.ENV, "done", {}),
    Filing("yc_bench", "yc_bench.1", "yc_bench", "submit", {}),
)


def offline_config(env_name: str, room: Path) -> Optional[Dict[str, Any]]:
    """What each environment above needs to be served with nothing fetched."""
    if env_name == "receipts_v1":
        return {"bundle": str(verified_bundle(room, size=1))}
    if env_name == cell.ENV:
        return {"tasks": [OFFLINE_TASK], "max_steps": 50}
    if env_name == "browsecomp_plus":
        return {
            "tasks": [OFFLINE_QUERY],
            "searcher": InMemorySearcher(OFFLINE_CORPUS),
            "judge": UncalledJudge(),
        }
    return None


def gated(env_name: str) -> None:
    """Skip where an environment's upstream is provisioned at runtime and is not here."""
    if env_name == cell.ENV:
        gate(
            "shogym.envs.automationbench.adapter",
            package="automationbench",
            extra="automationbench",
        )
    if env_name == "yc_bench":
        gate("shogym.envs.yc_bench.adapter", package="yc_bench", extra="yc_bench")


async def pulled_and_filed(
    client: Any, episode: ServedEpisode, environment: Any, filing: Filing
) -> List[str]:
    """Every text a model is handed across one pull and one filing, over a real stream."""
    spec = episode.describe()
    terminal = terminal_manifest(spec)
    gateway = await open_gateway(
        client,
        episode,
        environment=environment,
        start=stream_start(
            spec,
            terminal,
            claim_hash=sha256(b"a claim").hexdigest(),
            evaluation_only=True,
            grade=environment.grade,
        ),
    )
    try:
        async with Client(build_gateway_server(gateway)) as served:
            answers = [await served.call_tool(PULL_TOOL, {})]
            attempt = json.loads(answers[0].content[0].text)["attempt_id"]
            answers.append(
                await served.call_tool(
                    filing.terminal, {"attempt_id": attempt, "arguments": filing.arguments}
                )
            )
    finally:
        await gateway.aclose()
    return [getattr(item, "text", "") for answer in answers for item in answer.content]


def test_every_version_declared_here_has_a_case_in_the_table_below() -> None:
    """The table covers every declared version, under the module that declares it.

    The versions are gathered from the package, so an environment added later brings one with
    it and this fails until the table has a case for it. That is what makes the sweep below grow
    with the repository instead of covering the environments that happened to exist when it was
    written, which is the mistake this module has already made once.

    The module is checked with the version, because the reconstruction below puts that module's
    retired constant back and a row naming the wrong one would reconstruct the wrong thing.
    """
    assert {filing.port: filing.version for filing in FILINGS} == declared_versions()
    assert {filing.port for filing in FILINGS} == set(RETIRED_VERSIONS)


@pytest.mark.network
@pytest.mark.parametrize("filing", FILINGS, ids=[filing.version for filing in FILINGS])
async def test_no_platform_name_reaches_the_model_in_a_real_acknowledgement(
    filing: Filing, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The acknowledgement the kernel really mints, over each environment under its own version.

    The sweep above drives the transport and mints the acknowledgement in the double, so it says
    nothing about where the version in one comes from. Here nothing is minted by this file: a
    real stream is started, the Activities the environment answered with are registered with the
    Worker that serves it, and the seal the filing runs is the transaction the acknowledgement
    comes out of. So a kernel that answered with a name of its own rather than with the version
    the generation declared fails this, and the whole record is swept as well as the one field,
    because an acknowledgement that grew a sentence naming the platform would carry the name
    there.

    One case per environment, because that is the pairing production makes: the environment is
    asked how its attempts end, and the version it answers with is the version the generation
    declares and the one the acknowledgement carries. The stand-in's version is what an
    environment that brings no terminal of its own is served under, which is why the row for it
    is such an environment. Whether a doctored pairing would be refused is a different question
    and this does not ask it; the table above says which ports compare the version at the seal.
    ``automationbench`` and ``yc_bench`` are provisioned from upstream at runtime, so their cases
    skip on a machine that has neither the source nor the network to fetch it; the other three
    need nothing fetched.
    """
    gated(filing.env_name)
    room = tmp_path_factory.mktemp("filed")
    episode = await ServedEpisode.start(
        filing.env_name,
        task=0,
        ends_on_horizon=False,
        env_config=offline_config(filing.env_name, room),
    )
    running = False
    try:
        async with durable_client() as client:
            running = True
            environment = environment_terminal(episode)
            # The environment answers with its own version, and this is the case for that one.
            assert environment.canonicalization_version == filing.version
            async with stream_worker(client, activities=environment.activities):
                handed = await pulled_and_filed(client, episode, environment, filing)
    except Exception as error:  # noqa: BLE001 - re-raised below unless the service never came up
        if running:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")
    finally:
        await episode.close()

    assert offences(f"filed over {filing.env_name}", handed) == []
    acknowledgement = json.loads(handed[-1])
    assert acknowledgement["kind"] == "seal_ack"
    assert acknowledgement["canonicalization_version"] == filing.version


async def test_the_one_platform_named_version_left_is_in_nothing_the_model_can_read() -> None:
    """The wrapper version stays, and this is the difference that lets it.

    It is in the document a generation hashes and in nothing else: not a tool name, not a
    description, not a schema, not an instruction, and not a record the stream hands over. A name
    a model cannot read is not a name it can look anything up by, so this one is identity and the
    cut had no reason to reach it.
    """
    episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
    try:
        document = served_manifest(episode.describe(), terminal_manifest(episode.describe()))
    finally:
        await episode.close()
    assert carried(document["wrapper_version"]) == ["shogym"]
    assert carried(document["canonicalization_version"]) == []
    readable = [text for _, text in strings("served", await gateway_surface())]
    assert not any(document["wrapper_version"] in text for text in readable)


# ----- what the cut costs -----

#: The gateway's version as every served manifest carried it before the cut. It is in the
#: manifest whatever version the environment declares, so it is in the hash of every generation
#: this gateway composes and putting it back is part of reconstructing any pre-cut composition.
PARENT_GATEWAY_VERSION = "shogym.gateway.1"

#: The two terminal descriptions the cut moved, each under the tool it is in, with the wording
#: it carried before beside the wording it carries now. A description is inside the same hashed
#: document the version is, so a reconstruction that left one of them at today's wording would
#: hash a manifest no build ever served.
DONE_SENTENCE = "the sealed final state, against the task's own rubric;"
PARENT_DONE_SENTENCE = "the sealed final state (AutomationBench's own rubric);"
SUBMIT_SENTENCE = "The harness seals the"
PARENT_SUBMIT_SENTENCE = "shogym seals the"
MOVED_SENTENCES: Dict[str, Tuple[str, str, str]] = {
    cell.ENV: ("done", DONE_SENTENCE, PARENT_DONE_SENTENCE),
    "yc_bench": ("submit", SUBMIT_SENTENCE, PARENT_SUBMIT_SENTENCE),
}


@contextmanager
def the_retired_version(port: str) -> Iterator[None]:
    """Put one module's retired canonicalization version back, and take it out again after.

    An environment's configuration digest is taken over the version it declares as well as over
    what that environment is configured as, so the digest a pre-cut build folded into its hash
    cannot be reached by rewriting a field: it has to be computed again with the retired
    constant in place. The constant is read where the digest is taken rather than captured at
    import, so restoring it here and asking the environment again is the computation a pre-cut
    build did.
    """
    module = import_module(
        "shogym.serve.protocol_v2.gateway"
        if port == "gateway"
        else f"shogym.envs.{port}.protocol_v2"
    )
    today = module.CANONICALIZATION_VERSION
    module.CANONICALIZATION_VERSION = RETIRED_VERSIONS[port]
    try:
        yield
    finally:
        module.CANONICALIZATION_VERSION = today


def asked_before_the_cut(episode: ServedEpisode, port: str) -> Any:
    """What this environment answered a build from before the cut when the gateway asked it.

    The same call production makes, made with the retired constant in place, so the version and
    the configuration digest that come back are both the pre-cut ones rather than one pre-cut
    field beside a digest taken over today's.
    """
    with the_retired_version(port):
        return environment_terminal(episode)


def before_the_cut(description: str) -> str:
    """One tool description with whichever sentence the cut moved in it put back."""
    for _, sentence, parent in MOVED_SENTENCES.values():
        description = description.replace(sentence, parent)
    return description


def parent_configuration(spec: Any, terminal: Any, environment: Any) -> str:
    """The inner hash a build from before the cut took over this environment's served manifest.

    The manifest is the one production serves, with the strings the cut moved put back where
    they were and nothing else touched: the gateway's version, which every manifest carries
    whatever the environment declares, and the terminal sentence of each environment whose
    wording moved.

    ``environment`` is the answer a pre-cut build got when it asked this environment, so the
    digest folded in below is that build's rather than today's. For the environments whose own
    version moved that is the answer taken with the retired constant in place, and for one whose
    version did not move it is today's answer, because nothing about it moved.
    """
    manifest = served_manifest(spec, terminal, environment.horizon_ending)
    parent = {
        **manifest,
        "canonicalization_version": PARENT_GATEWAY_VERSION,
        "tools": [
            {**tool, "description": before_the_cut(tool["description"])}
            for tool in manifest["tools"]
        ],
    }
    if environment.configuration_digest is not None:
        parent = {**parent, "environment": environment.configuration_digest}
    return sha256(canonical_json(parent)).hexdigest()


def composed_over(spec: Any, terminal: Any, environment: Any) -> Any:
    """The generation this gateway starts over one environment, as ``open_gateway`` completes it.

    The environment is asked what its attempts are sealed under and what it is configured as,
    and both go into what the generation is: the version it declares, and the hash taken over
    the manifest it serves under the ending that environment gives a spent budget.
    """
    composed = stream_start(
        spec,
        terminal,
        claim_hash=sha256(b"a claim").hexdigest(),
        evaluation_only=True,
        grade=environment.grade,
    )
    return replace(
        composed,
        canonicalization_version=environment.canonicalization_version,
        configuration_hash=_configuration_hash(
            spec, terminal, environment.configuration_digest, environment.horizon_ending
        ),
    )


def refuses_the_parent(root: Path, parent: Any, current: Any) -> Any:
    """Record ``parent`` as a run directory and take it over as ``current``, which refuses."""
    create_run_directory(
        root,
        workflow_id=f"stream/{root.name}/1",
        task_queue=STREAM_TASK_QUEUE,
        configuration_hash=configuration_hash(parent),
    )
    # No client is reached: the directory is read and the hash compared before anything is
    # claimed, which is what makes this a refusal rather than a takeover that fails.
    return resume_run_directory(None, root, start=current)  # type: ignore[arg-type]


@pytest.mark.parametrize("filing", FILINGS, ids=[filing.version for filing in FILINGS])
async def test_a_run_recorded_before_the_cut_is_refused_on_resume(
    filing: Filing, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """What the cut costs, over each environment, written down as the refusal it is.

    A run recorded before the cut committed the hash a pre-cut build derived, and both layers of
    that hash moved. The outer one is over the generation, which carries the version that
    environment declares. The inner one is over the served manifest, which carried the gateway's
    version, the terminal description of the two environments whose wording moved, and the
    environment's own configuration digest, which is itself taken over the version that
    environment declares.

    So the environment is asked again with its retired constant in place, and what it answers is
    the version and the digest a pre-cut build was answered with. The manifest is hashed with
    the sentences put back around that digest, the parent generation carries both layers, and a
    process composed today is refused by the directory that composition recorded.
    """
    gated(filing.env_name)
    retired = RETIRED_VERSIONS[filing.port]
    room = tmp_path_factory.mktemp("recorded")
    episode = await ServedEpisode.start(
        filing.env_name,
        task=0,
        ends_on_horizon=False,
        env_config=offline_config(filing.env_name, room),
    )
    try:
        spec = episode.describe()
        terminal = terminal_manifest(spec)
        environment = environment_terminal(episode)
        before = asked_before_the_cut(episode, filing.port)
        current = composed_over(spec, terminal, environment)
        parent = replace(
            current,
            canonicalization_version=before.canonicalization_version,
            configuration_hash=parent_configuration(spec, terminal, before),
        )
        moved = MOVED_SENTENCES.get(filing.env_name)
        if moved is not None:
            # The environments whose manifest carried a moved sentence as well. The
            # reconstruction above put it back, so this is where it is checked there was one to
            # put back: a sentence that had already moved on would replace nothing.
            name, sentence, _ = moved
            [tool] = [
                tool
                for tool in served_manifest(spec, terminal, environment.horizon_ending)["tools"]
                if tool["name"] == name
            ]
            assert sentence in tool["description"]
    finally:
        await episode.close()

    assert before.canonicalization_version == retired
    assert current.canonicalization_version == filing.version
    # The digest an environment that has one publishes moved with its version, which is the
    # input the outer field cannot reach and the reason the environment is asked twice.
    assert (before.configuration_digest is None) == (environment.configuration_digest is None)
    if environment.configuration_digest is not None:
        assert before.configuration_digest != environment.configuration_digest
    assert parent.canonicalization_version != current.canonicalization_version
    assert parent.configuration_hash != current.configuration_hash
    assert configuration_hash(parent) != configuration_hash(current)
    with pytest.raises(ResumeRefused) as refused:
        await refuses_the_parent(tmp_path / filing.env_name, parent, current)
    assert refused.value.code == "configuration_mismatch"
    assert configuration_hash(parent) in str(refused.value)


async def test_a_generation_whose_own_version_did_not_move_is_refused_too(
    tmp_path: Path,
) -> None:
    """The boundary of the cut, which is every generation this gateway composes.

    An environment that brings its own terminal declares its own version, and one declared
    outside this repository did not move. Its generation still hashes the gateway's version,
    because the served manifest carries that constant whatever the environment declares, so the
    configuration hash of such a generation moved with everything else. A run it recorded is
    refused on resume over a string its own environment never said, and that is what makes this
    one cut across every composition rather than a cut over four environments.

    Nothing of this environment's own is put back, because nothing of its own moved: a digest
    taken over a version this cut did not touch is the digest a pre-cut build folded in, so
    today's answer is the answer that build was given.
    """
    episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
    try:
        spec = episode.describe()
        terminal = terminal_manifest(spec)
        own = environment_terminal(episode)._replace(
            canonicalization_version="world.1", configuration_digest="a digest of its own"
        )
        current = composed_over(spec, terminal, own)
        parent = replace(current, configuration_hash=parent_configuration(spec, terminal, own))
    finally:
        await episode.close()

    assert parent.canonicalization_version == current.canonicalization_version == "world.1"
    assert carried("world.1") == []
    # Only the manifest moved, and it is enough.
    assert parent.configuration_hash != current.configuration_hash
    with pytest.raises(ResumeRefused) as refused:
        await refuses_the_parent(tmp_path / "own-terminal", parent, current)
    assert refused.value.code == "configuration_mismatch"
    assert configuration_hash(parent) in str(refused.value)
