"""The two domains this cell runs across, and the boundary between them.

The agent runs in one container and the measurement runs in another. The agent's container holds
the directory it works in and the Claude Code home it writes to, and nothing else this run
created: no run directory, no repository, no benchmark cache, no grade. The server's container
holds all four, and publishes one thing, the gateway's MCP endpoint, on a private network the two
containers share.

That is the boundary, and it is made of mounts rather than of an allow list. An agent under
``bypassPermissions`` can read the filesystem it is running on, so the question a rerun has to
answer is not whether the agent was told to leave the grades alone but whether the grades were
ever in front of it. Here they are on the far side of a container, reachable only through the
protocol, which is where the cell this one reruns kept them.

The durable service the gateway runs on starts inside the server's container and binds that
container's own loopback, so a network the two containers share still carries nothing but the
endpoint. The agent keeps the internet, because the cell this one reruns did.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import subprocess
import tarfile
import tempfile
import time
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from examples.automationbench_cell import pinned

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: What the two images are called. The agent's carries the CLI build in its tag, because the build
#: is the half of the agent no flag can pin and a tag is where a rebuild becomes visible.
AGENT_IMAGE = "shogym-cell-agent"
SERVER_IMAGE = "shogym-cell-server"

#: The label each image carries, holding a digest of everything it was built from. A tag says
#: which build was asked for and says nothing about what answered: an image built from an earlier
#: checkout keeps the tag it was given, and a run that reused it would serve the old grader under
#: a record naming the new one. So the identity is written into the image at build time, read back
#: before reuse, and kept in the run's own record.
BUILD_LABEL = "shogym.cell.build"

#: What the server's image copies out of this repository, which is the other half of what it is
#: built from. The package is the gateway, the grader and the durable history, so a change to any
#: of it is a change to the measurement and not to the example that starts it. The lock is in the
#: list because it decides the distributions: this image installs from it rather than from the
#: ranges beside it, so the same source at another lock is another server.
SERVER_SOURCE = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "NOTICE",
    "src",
    "examples/automationbench_cell",
)

#: What this repository generates rather than keeps, and what is therefore never part of what an
#: image is built from. A run directory is this cell's own output and the default place a probe
#: and a launch write, so the tree the server's image copies holds one as soon as the first
#: command in the README has been followed; bytecode and tool caches appear beside any source that
#: has been imported or linted. Left in, they would reach the source digest and the image both, and
#: two builds of one source would be two different measurements, the second carrying the first
#: one's transcripts and grades. The patterns are docker's: one that matches a directory leaves out
#: everything under it, and one beginning ``**/`` matches at any depth.
GENERATED = (
    "examples/*/runs",
    "**/__pycache__",
    "**/*.py[cod]",
    "**/*.egg-info",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.venv",
    "**/shogym_logs",
    "**/.DS_Store",
)

#: Where each recipe sits in the context its own build is handed. The agent's context is the
#: recipe alone, because that image copies nothing out of this repository; the server's is the
#: source its identity is a digest of, and its recipe is one of those files.
AGENT_DOCKERFILE = "agent.Dockerfile"
SERVER_DOCKERFILE = (HERE / "server.Dockerfile").relative_to(REPO).as_posix()

#: The lock the server's dependencies are installed from, and the exporter that turns it into the
#: pinned, hashed requirements the image installs. The exporter's own version is part of the build
#: because it is what reads the lock.
SERVER_LOCK = "uv.lock"
UV_VERSION = "0.11.20"

#: The environment the agent's image and docker itself supply, by name. ``PWD`` is docker's, from
#: the working directory the container is started in, ``NODE_OPTIONS`` is this cell's recipe
#: setting it empty as the recorded image did, and the rest are the base image's. The launch adds
#: this cell's own two and the credential's name and nothing else, so any name in a container that
#: is not one of these is something the agent was handed that nobody wrote down. The list is fixed
#: because the base image is pinned by digest and the recipe beside this file is digested.
IMAGE_ENVIRONMENT = (
    "HOME",
    "HOSTNAME",
    "NODE_OPTIONS",
    "NODE_VERSION",
    "PATH",
    "PWD",
    "YARN_VERSION",
)

#: Where the agent works, and where the two directories it keeps are mounted. ``/work`` is the
#: path the recorded run's agent worked in, and the home is the whole of ``/root`` as that run
#: mounted it: the CLI writes its memory under ``.claude`` and writes elsewhere in the home too,
#: and a mount of the subtree alone would let the rest die with the container.
WORK = "/work"
AGENT_HOME = "/root"
CONFIG_MOUNT = "/cfg"

#: What the file naming the endpoint is called inside that mount, which is the name the recorded
#: run gave it. The flag names the file outright and no directory is searched, so nothing here
#: depends on the spelling; it is still a name the agent's own shell finds in the one directory it
#: was given, and a launch fact is adopted rather than defaulted.
MCP_CONFIG = "claude.mcp.json"

#: Where the server's own three live. The run directory is the generation's history, its blobs and
#: its sealed grades; the cache holds the pinned benchmark source and the durable service's binary.
GRADES_MOUNT = "/grades"
CACHE_MOUNT = "/cache"

#: What the benchmark source is called on both sides of the cache: the directory the host keeps it
#: in, and the name under the cache mount that the server reads it at and the fetch writes it to.
SOURCE_CACHE = "automationbench"

#: The port the gateway listens on inside the server's container. Nothing publishes it to the
#: host: it is reachable on the private network and there only.
SERVER_PORT = 9000


def default_cache() -> Path:
    """Where the host keeps the benchmark source and the service binary this cell reuses.

    Resolved rather than read out of the ambient environment. This repo honours ``SHOGYM_CACHE``,
    and a cell that honoured it too would let an operator's shell decide which benchmark the run
    served while the run's own record still named the standard one. A launch takes the path as an
    argument and writes down the one it resolved.
    """
    return Path.home() / ".cache" / "shogym"


def cache_mounts(cache: Path) -> List[Tuple[Path, str, str]]:
    """The two directories the server's container reuses from the host, and how each is bound.

    The benchmark source is read only: it is the task definitions, the answers and the scoring
    assertions, and the run has no business writing to them. The service binary is a download the
    server would otherwise repeat every run, and it is kept apart from the host's own copy because
    the two are built for different operating systems.
    """
    return [
        (cache / SOURCE_CACHE, f"{CACHE_MOUNT}/{SOURCE_CACHE}", "ro"),
        (cache / "cell-server-temporal", f"{CACHE_MOUNT}/temporal", "rw"),
    ]


def agent_mounts(run_dir: Path, *, self_dir: str, home_dir: str, config_dir: str) -> List[
    Tuple[Path, str, str]
]:
    """Everything the agent's container can see of this host, which is three directories.

    Two of them are the agent's own and survive the run: the directory it works in and the home
    its memory, its skills and everything else the CLI keeps land in. The third is the file naming
    the endpoint, mounted read only and outside the working directory so that it never becomes
    part of the self a later run would start from.
    """
    return [
        (run_dir / self_dir, WORK, "rw"),
        (run_dir / home_dir, AGENT_HOME, "rw"),
        (run_dir / config_dir, CONFIG_MOUNT, "ro"),
    ]


def server_mounts(run_dir: Path, *, grades_dir: str, cache: Path) -> List[Tuple[Path, str, str]]:
    """Everything the server's container can see, which is the run and the cache."""
    return [(run_dir / grades_dir, GRADES_MOUNT, "rw"), *cache_mounts(cache)]


def _mount_flags(mounts: Sequence[Tuple[Path, str, str]]) -> List[str]:
    flags: List[str] = []
    for source, target, mode in mounts:
        flags += ["-v", f"{source}:{target}:{mode}"]
    return flags


def mount_record(mounts: Sequence[Tuple[Path, str, str]]) -> List[str]:
    """The mounts as a record keeps them: what was bound where, and whether it could be written."""
    return [f"{source}:{target}:{mode}" for source, target, mode in mounts]


def gateway_url(server: str) -> str:
    """The endpoint the agent's config names, which is the container and not the host.

    The path is the one the server publishes, without the trailing slash the server answers with
    a redirect: an MCP client that does not follow one connects to nothing, and a run whose agent
    never connects still exits reporting success.
    """
    return f"http://{server}:{SERVER_PORT}/mcp"


def names(token: str) -> Tuple[str, str, str]:
    """The network and the two containers one run owns, kept apart from any other run's.

    A network per run is what keeps two cells on one host out of each other's endpoint: a server
    is reachable by name, and a name only resolves on the network it was started on.
    """
    return f"shogym-cell-{token}", f"shogym-cell-{token}-server", f"shogym-cell-{token}-agent"


# --- docker -----------------------------------------------------------------------------------


def _docker(args: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], **kwargs)  # type: ignore[arg-type]


def docker_available() -> bool:
    """Whether this host has a docker daemon to run the two domains on."""
    try:
        return _docker(["info"], capture_output=True).returncode == 0
    except OSError:
        return False


def image_id(image: str) -> Optional[str]:
    """The identity of the image on this host, or nothing when there is no such image."""
    done = _docker(["image", "inspect", "-f", "{{.Id}}", image], capture_output=True, text=True)
    return done.stdout.strip() or None if done.returncode == 0 else None


def image_build(image: str) -> Optional[str]:
    """What the image on this host was built from, or nothing when it cannot say.

    An image built before this label existed answers nothing, which is the same answer as no
    image at all: neither can be shown to be the one a comparison wants, so both are rebuilt.
    """
    done = _docker(
        ["image", "inspect", "-f", f'{{{{index .Config.Labels "{BUILD_LABEL}"}}}}', image],
        capture_output=True,
        text=True,
    )
    identity = done.stdout.strip()
    return identity if done.returncode == 0 and identity and identity != "<no value>" else None


def file_digest(path: Path) -> str:
    """Return a digest of one file's bytes, in the shape the recorded inputs name them."""
    return f"sha256:{sha256(Path(path).read_bytes()).hexdigest()}"


def _base_image(dockerfile: Path) -> str:
    """The image a Dockerfile is built on, read out of the file rather than described twice."""
    for line in Path(dockerfile).read_text(encoding="utf-8").splitlines():
        if line.strip().upper().startswith("FROM "):
            return line.split(None, 1)[1].strip()
    raise ValueError(f"{dockerfile} names no base image, so what it builds on is unknown")


def agent_build_inputs() -> Dict[str, str]:
    """What the agent's image is built from, as a launch resolves it here and now.

    The recipe, the base, the OS packages, the package, the registry and the version: everything
    that decides what the model reaches through Bash and which harness reads its prompt. The base
    is read off the file that builds the image and the rest are what the build is handed, so what
    this returns is resolved rather than recorded. It is what ``pinned.AGENT_IMAGE_BUILD`` is
    compared with, so a rerun on another host either matches the recorded inputs or says which of
    them moved.

    The OS packages are here because they are installed rather than inherited. A base pinned by
    digest fixes the image the build starts from and fixes nothing about the apt resolution on top
    of it, so a shell, a ``curl`` and a ``python3`` chosen by whatever the repository served that
    day used to be outside every identity this cell kept.
    """
    return {
        "apt_packages": " ".join(pinned.APT_PACKAGES),
        "apt_snapshot": pinned.APT_SNAPSHOT,
        "base": _base_image(HERE / "agent.Dockerfile"),
        "cli_package": pinned.CLI_PACKAGE,
        "cli_registry": pinned.CLI_REGISTRY,
        "cli_version": pinned.CLI_VERSION,
        "dockerfile": file_digest(HERE / "agent.Dockerfile"),
    }


def _matches(parts: Sequence[str], pattern: str) -> bool:
    """Whether a path, split into its parts, is under something this pattern names."""
    tokens = pattern.split("/")
    if tokens[0] == "**":
        rest = "/".join(tokens[1:])
        return any(_matches(parts[at:], rest) for at in range(len(parts)))
    if len(tokens) > len(parts):
        return False
    return all(fnmatchcase(part, token) for part, token in zip(parts, tokens))


def generated(relative: str) -> bool:
    """Whether this path in the repository is one of the things the repository generates."""
    parts = PurePosixPath(relative).parts
    return any(_matches(parts, pattern) for pattern in GENERATED)


def source_files(names: Optional[Sequence[str]] = None) -> List[Tuple[Path, str]]:
    """Every file the server's image is built from: where it is here, and where it is in the build.

    One list answers both halves of the identity question, which is why it is a list rather than
    two walks. It is what the source digest is taken over and it is the whole of the context the
    build is handed, so a file that would change the image is a file that changes the digest, and
    a file left out of the digest is one the build never sees. The two used to be a tree walk and
    a directory handed to docker, and what fell between them was everything this repository
    generates: an image built after a probe carried that probe's transcripts and grades under a
    label that said it was built from the source beside it.
    """
    found: List[Tuple[Path, str]] = []
    for name in SERVER_SOURCE if names is None else names:
        path = REPO / name
        if not path.is_dir():
            found.append((path, name))
            continue
        for item in sorted(entry for entry in path.rglob("*") if entry.is_file()):
            relative = item.relative_to(REPO).as_posix()
            if not generated(relative):
                found.append((item, relative))
    return found


def _as_built(entry: tarfile.TarInfo) -> tarfile.TarInfo:
    """Strip an entry down to what an image is built from: where the file is, and its bytes.

    Who owned it and when it was last written are this checkout's facts rather than the source's,
    and they are what would otherwise make two builds of identical files two different contexts.
    """
    entry.uid = entry.gid = 0
    entry.uname = entry.gname = ""
    entry.mtime = 0
    return entry


def write_context(archive: Path, files: Sequence[Tuple[Path, str]]) -> Path:
    """Write the build context as an archive holding those files and nothing else.

    A directory handed to docker is a context whose contents nobody enumerated. This is the
    enumeration: what goes in is what ``source_files`` returned, so what the build can copy is
    what the identity was computed over rather than whatever else the directory happened to hold.

    The archive is written in the older format on purpose. Docker decides whether what it is being
    handed is a context or a recipe by reading the first entry out of the first block, and the
    modern format writes an extended header there that does not fit in one: a context in it is
    read as a Dockerfile and the build refuses before it starts.
    """
    with tarfile.open(archive, "w", format=tarfile.GNU_FORMAT) as bundle:
        for path, name in files:
            bundle.add(path, arcname=name, filter=_as_built)
    return archive


def server_context(archive: Path) -> Path:
    """Take the snapshot of this repository that the server's image is built from and named by.

    One archive answers for both. It is what the build is handed and it is what the recorded
    source digest is a digest of, so there is one read of the tree rather than two: a file saved
    between a digest and a stream would otherwise label one context with the identity of another.
    """
    return write_context(archive, source_files())


def server_build_inputs(context: Path) -> Dict[str, str]:
    """What the server's image is built from, which is this repository as it stands.

    The source is a digest of the archive the build is handed, so it names those files, their
    bytes and the mode and kind of each entry, and not a summary taken separately from them.
    An image built from an earlier checkout does not answer to it. Nothing pins it to a recorded
    value: the measurement lives in this package, and the value of the check is that a run never
    serves a build older than the source beside it.

    The lock is named on its own as well as counted in the source, because it is what decides the
    distributions this image installs. The ranges in ``pyproject.toml`` say which versions are
    admissible and the lock says which ones were chosen, and only the second is what runs.
    """
    return {
        "base": _base_image(HERE / "server.Dockerfile"),
        "dockerfile": file_digest(HERE / "server.Dockerfile"),
        "lock": file_digest(REPO / SERVER_LOCK),
        "source": file_digest(context),
        "uv_version": UV_VERSION,
    }


def build_identity(inputs: Mapping[str, str]) -> str:
    """One digest over everything an image was built from, which is what the label holds."""
    return sha256(json.dumps(dict(inputs), sort_keys=True).encode("utf-8")).hexdigest()


def build_images(*, agent: str, server: str, rebuild: bool = False) -> Dict[str, Dict[str, str]]:
    """Build whichever image this host does not already hold at the inputs it is built from.

    A tag is not an identity. The old rule was to skip the build whenever the tag existed, which
    left an image from an earlier checkout serving under the name of this one. So the inputs are
    digested, written into the image as a label, and read back before reuse: an image that cannot
    say what it was made of, or that says something else, is rebuilt.

    The build arguments carry the pins, so neither image resolves anything of its own: the
    recorded CLI version and the registry it comes from, the archive moment and the exact versions
    the OS packages are installed at, and the exporter that turns this repository's lock into what
    the server installs. Each image is then built from the inputs the comparison is against rather
    than from whichever ones npm, Debian and PyPI were serving on the day it was built.

    What comes back is what each image was built from, which is what a launch checks against the
    recorded inputs and what the run's own record keeps. The label is a digest of exactly that, so
    it is not returned twice.

    Each build is handed an archive of the files its identity names rather than a directory, and
    for the server it is handed the very archive the identity is a digest of. The context is taken
    once, before anything is compared, so the bytes that decide whether to build are the bytes the
    build gets. The agent's image copies nothing out of this repository; the server's copies the
    source it is named by.
    """
    with tempfile.TemporaryDirectory() as scratch:
        source = server_context(Path(scratch) / "server.tar")
        recipes = (
            (
                agent,
                agent_build_inputs(),
                AGENT_DOCKERFILE,
                write_context(
                    Path(scratch) / "agent.tar", [(HERE / "agent.Dockerfile", AGENT_DOCKERFILE)]
                ),
                [
                    f"APT_SNAPSHOT={pinned.APT_SNAPSHOT}",
                    f"APT_PACKAGES={' '.join(pinned.APT_PACKAGES)}",
                    f"CLAUDE_CODE_PACKAGE={pinned.CLI_PACKAGE}",
                    f"CLAUDE_CODE_VERSION={pinned.CLI_VERSION}",
                    f"CLAUDE_CODE_REGISTRY={pinned.CLI_REGISTRY}",
                ],
            ),
            (
                server,
                server_build_inputs(source),
                SERVER_DOCKERFILE,
                source,
                [f"UV_VERSION={UV_VERSION}"],
            ),
        )
        built: Dict[str, Dict[str, str]] = {}
        for image, resolved, dockerfile, context, arguments in recipes:
            identity = build_identity(resolved)
            if rebuild or image_build(image) != identity:
                print(f"[cell] building {image}", flush=True)
                argument_flags = [flag for value in arguments for flag in ("--build-arg", value)]
                with context.open("rb") as stream:
                    _docker(
                        [
                            "build",
                            "-f",
                            dockerfile,
                            *argument_flags,
                            "--label",
                            f"{BUILD_LABEL}={identity}",
                            "-t",
                            image,
                            "-",
                        ],
                        check=True,
                        stdin=stream,
                    )
            built[image] = resolved
    return built


#: What fills the benchmark cache, run in the server's own image. The import is what provisions:
#: the commit the source comes from and the archive it is fetched out of are named in the adapter
#: and nowhere else, so a cell that named them again could pin one benchmark while the server
#: loaded another.
PROVISION = "import shogym.envs.automationbench.adapter"


def provisioner_name(server: str) -> str:
    """What the fetch runs under, which is this run's own name and a container a launch can take.

    A signal reaches the launcher rather than the daemon: the docker client dies and the container
    goes on holding and writing the cache. So the fetch is named after the run that started it,
    the way the two long-lived containers are, and a launch that is stopped has a name to remove.
    """
    return f"{server}-provision"


def provision_argv(*, image: str, cache: Path, name: str) -> List[str]:
    """The one-shot container that fills the benchmark cache before the server is given it.

    The server has the source mounted read only, because the task definitions, the answers and the
    scoring assertions are not a run's to write. A cache that has never been filled cannot be
    filled from behind that mount, so a first launch on a clean host died there with the loader's
    own read-only error and served nothing. The fetch happens here instead, in a container that
    has this one directory writable and holds nothing else of the run: no grades, no history and
    no network of this run's. What is long lived stays read only.
    """
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "-v",
        f"{cache / SOURCE_CACHE}:{CACHE_MOUNT}/{SOURCE_CACHE}:rw",
        "-e",
        f"SHOGYM_CACHE={CACHE_MOUNT}",
        image,
        "python",
        "-c",
        PROVISION,
    ]


def provision_source(image: str, *, cache: Path, name: str) -> None:
    """Fill the benchmark cache, and refuse the launch when it cannot be filled.

    It is idempotent: a cache that already holds the pinned source is a container that reads it
    and exits. A failure is raised here rather than left to the server, because the server would
    meet it as a read-only filesystem inside a container whose logs a launcher only saves at
    teardown.
    """
    # The command names docker itself, as the two launch commands beside it do, and this is the
    # one of the three that is run from here rather than handed to the launcher.
    argv = provision_argv(image=image, cache=cache, name=name)[1:]
    done = _docker(argv, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(
            f"the benchmark source could not be provisioned into {cache}, so the server would "
            f"have nothing to serve: {(done.stderr or done.stdout).strip()}"
        )


def cli_version_command(agent: str) -> List[str]:
    """How the CLI build is asked for its version, which is inside the image that will run it."""
    return ["docker", "run", "--rm", "--entrypoint", "claude", agent, "--version"]


def create_network(network: str) -> None:
    _docker(["network", "rm", network], capture_output=True)
    _docker(["network", "create", network], check=True, capture_output=True)


def _still_here(listing: Sequence[str], name: str) -> Optional[str]:
    """Whether docker still lists something under this exact name, or why it cannot be asked."""
    try:
        done = _docker(
            [*listing, "--filter", f"name=^{name}$", "-q"], capture_output=True, text=True
        )
    except OSError as failure:
        return f"docker could not be asked whether {name} is still here: {failure}"
    if done.returncode != 0:
        return f"docker could not be asked whether {name} is still here: {done.stderr.strip()}"
    return f"{name} is still here" if done.stdout.strip() else None


def remove_network(network: str) -> Optional[str]:
    """Take the network down, and return why it is still up or nothing when it is gone."""
    try:
        _docker(["network", "rm", network], capture_output=True)
    except OSError:
        pass
    return _still_here(["network", "ls"], network)


def remove_container(name: str) -> Optional[str]:
    """Stop and remove the container, and return why it is still running or nothing when it is not.

    The result is read back rather than taken from the removal's own exit code, because the code
    is nonzero both for a container that was already gone, which is the ordinary case for one
    started with ``--rm``, and for one that would not die. Those are opposite facts, and only the
    listing tells them apart.
    """
    try:
        _docker(["rm", "-f", name], capture_output=True)
    except OSError:
        pass
    return _still_here(["ps", "-a"], name)


def server_argv(
    *,
    image: str,
    name: str,
    network: str,
    mounts: Sequence[Tuple[Path, str, str]],
    environment: Mapping[str, str],
) -> List[str]:
    """The command that starts the measurement's domain.

    No port is published. The endpoint is reachable by container name on the private network, so
    a run serves the agent it started and nothing else on the host can reach the generation.
    """
    argv = ["docker", "run", "-d", "--name", name, "--network", network]
    argv += _mount_flags(mounts)
    for key, value in sorted(environment.items()):
        argv += ["-e", f"{key}={value}"]
    argv.append(image)
    return argv


def server_environment(
    *, tasks: str, domain: str, schedule: str
) -> Dict[str, str]:
    """What the server is told, which is built here and inherited from nowhere.

    The container supplies the operating system and this supplies the run, so no variable an
    operator's shell holds reaches the process that loads the benchmark and commits the scores.
    """
    return {
        "SHOGYM_CELL_TASKS": tasks,
        "SHOGYM_CELL_DOMAIN": domain,
        "SHOGYM_CELL_SCHEDULE": schedule,
        "SHOGYM_CELL_RUN_DIR": GRADES_MOUNT,
        "SHOGYM_CELL_HOST": "0.0.0.0",
        "SHOGYM_CELL_PORT": str(SERVER_PORT),
        "SHOGYM_CACHE": CACHE_MOUNT,
    }


def agent_argv(
    *,
    image: str,
    name: str,
    network: str,
    mounts: Sequence[Tuple[Path, str, str]],
    environment: Mapping[str, str],
    credential: Optional[str],
    command: Sequence[str],
) -> List[str]:
    """The command that starts the agent's domain, and the whole of what it is given.

    The credential is passed by name rather than by value, so what authenticated the run is in the
    launcher's environment and never in an argument list a record or a process table would keep.
    """
    argv = ["docker", "run", "--rm", "--name", name, "--network", network]
    for key, value in sorted(environment.items()):
        argv += ["-e", f"{key}={value}"]
    if credential is not None:
        argv += ["-e", credential]
    argv += _mount_flags(mounts)
    argv += ["-w", WORK, image, *command]
    return argv


def wait_for_gateway(server: str, *, tries: int = 900, settle: float = 6.0) -> None:
    """Wait until the gateway is listening inside the server's container, or say why it is not.

    The socket opens before the endpoint is ready to answer a handshake, so a settle follows the
    first connection: an agent whose first MCP call races an unready server exits with nothing.
    The wait is long because a cold run downloads the durable service and reads the benchmark
    before it serves anything.
    """
    probe = "import socket; socket.create_connection(('127.0.0.1', %d), 1).close()" % SERVER_PORT
    for _ in range(tries):
        if _docker(["exec", server, "python", "-c", probe], capture_output=True).returncode == 0:
            time.sleep(settle)
            return
        running = _docker(
            ["inspect", "-f", "{{.State.Running}}", server], capture_output=True, text=True
        )
        if running.stdout.strip() != "true":
            logs = _docker(["logs", "--tail", "40", server], capture_output=True, text=True)
            raise RuntimeError(f"the server stopped before it served:\n{logs.stdout}{logs.stderr}")
        time.sleep(1.0)
    logs = _docker(["logs", "--tail", "40", server], capture_output=True, text=True)
    raise RuntimeError(f"the server never listened:\n{logs.stdout}{logs.stderr}")


def save_logs(server: str, path: Path) -> None:
    """Keep what the server said, which is the only place its side of the run is written down."""
    logs = _docker(["logs", server], capture_output=True, text=True)
    path.write_text(logs.stdout + logs.stderr, encoding="utf-8")


#: How the gateway's own access log names a request it answered at the endpoint, and answered
#: successfully. The path is matched exactly, because ``/mcp/`` is the redirect this cell's URL
#: was corrected away from and ``/mcpx`` is not this endpoint at all, and the status is matched
#: because a request the gateway refused is not one it served. The readiness check connects and
#: closes without speaking, so it leaves no line here at all.
_SERVED = re.compile(r'"(?:GET|POST|PUT|PATCH|DELETE) /mcp(?:\?[^"\s]*)? HTTP/[0-9.]+" 2\d\d')


def served_requests(log: str) -> int:
    """How many requests the server's log says the gateway answered at the endpoint.

    This is the server's side of whether the agent ever arrived, and it is only that side: the
    handshake and the tool listing are requests too, so a count above nought says something
    reached the measurement and never says a task came back. What was served is read from the
    agent's own transcript.
    """
    return len(_SERVED.findall(log))


def parse_listeners(table: str) -> List[str]:
    """Return the listening sockets a kernel TCP table names, each as its address and port.

    The table is hexadecimal, and each four bytes of the address are written least significant
    first, so this is a parse rather than a read. State ``0A`` is listening, and a socket in any
    other state is not something anybody could connect to.
    """
    listeners: List[str] = []
    for line in table.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[3] != "0A":
            continue
        host, _, port = fields[1].partition(":")
        packed = bytes.fromhex(host)
        if len(packed) == 4:
            address = socket.inet_ntop(socket.AF_INET, packed[::-1])
        else:
            address = socket.inet_ntop(
                socket.AF_INET6, b"".join(packed[at : at + 4][::-1] for at in range(0, 16, 4))
            )
        listeners.append(f"{address}:{int(port, 16)}")
    return listeners


def listening_sockets(server: str) -> List[str]:
    """What the server's container is listening on, read out of the container's own kernel table.

    This is the network half of the boundary stated as a fact rather than as an intention: the
    endpoint is the only listener bound to an address another container could route to, and the
    durable service the history lives in is bound to this container's loopback.

    A read that fails is refused rather than answered with an empty list. An empty list is what a
    container listening on nothing looks like, and a check that cannot tell that from a container
    it could not ask passes whenever the asking breaks.
    """
    done = _docker(
        ["exec", server, "sh", "-c", "cat /proc/net/tcp /proc/net/tcp6 2>/dev/null"],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        raise ValueError(
            f"{server}'s listeners could not be read, so what it is reachable on is unknown: "
            f"{(done.stderr or done.stdout).strip()}"
        )
    return sorted(set(parse_listeners(done.stdout)))


def network_namespace(container: str) -> str:
    """The network namespace a container's own PID 1 is in, named as the kernel names it.

    Two containers on a bridge are two namespaces, and a container started to share another's is
    one. Nothing else in a launch tells them apart: the same mounts, the same environment and the
    same resolver answer either way, so the identifier is read and compared rather than inferred
    from the flag that was meant to have been passed.

    A read that fails is refused rather than answered with a name that matches nothing, for the
    same reason the listener read is: a namespace nobody could ask about is not one anybody has
    shown to be separate.
    """
    done = _docker(
        ["exec", container, "readlink", "/proc/1/ns/net"], capture_output=True, text=True
    )
    identity = done.stdout.strip()
    if done.returncode != 0 or not identity:
        raise ValueError(
            f"{container}'s network namespace could not be read, so what it shares is unknown: "
            f"{(done.stderr or done.stdout).strip()}"
        )
    return identity


def loopback_listeners(listeners: Sequence[str]) -> List[str]:
    """Those of a container's listeners that are on its own loopback, address and port together.

    These are the ones the boundary rests on rather than the one it publishes: the durable service
    holding the history is here, and it is out of reach because a loopback address belongs to the
    namespace it is in. So they are what the agent's container is asked about, at the exact
    address each was bound to, and an answer there is a shared namespace rather than a shared
    network.
    """
    loopback: List[str] = []
    for entry in listeners:
        address, _, _ = entry.rpartition(":")
        try:
            if ipaddress.ip_address(address).is_loopback:
                loopback.append(entry)
        except ValueError:
            continue
    return loopback


def unexpected_listeners(listeners: Sequence[str]) -> List[str]:
    """Those of the server's listeners a container on its network could reach and should not.

    The address and the port are compared apart. The endpoint is one port, so a port that merely
    ends in the endpoint's digits is another service: ``19000`` is not ``9000``, and a suffix test
    would have waved it through. An address this cannot parse is not treated as loopback either,
    because a listener nobody can place is one nobody has shown to be unreachable.
    """
    unexpected: List[str] = []
    for entry in listeners:
        address, _, port = entry.rpartition(":")
        try:
            loopback = ipaddress.ip_address(address).is_loopback
        except ValueError:
            loopback = False
        if loopback or port == str(SERVER_PORT):
            continue
        unexpected.append(entry)
    return unexpected


def topology(
    *,
    network: str,
    server: str,
    agent: str,
    agent_image: str,
    server_image: str,
    agent_mount_list: Sequence[Tuple[Path, str, str]],
    server_mount_list: Sequence[Tuple[Path, str, str]],
    agent_environment: Mapping[str, str],
    server_env: Mapping[str, str],
    credential: Optional[str],
    images: Mapping[str, Mapping[str, str]],
) -> Dict[str, object]:
    """The boundary as the run's own record keeps it.

    Built from the same lists the two commands were built from, so a record that says the agent
    was given three directories is saying what the launch gave it rather than what this file once
    said it would. Each image is named three ways, because none of the three answers for the
    others: the tag says what was asked for, the id says which image on this host answered, and
    the build says what that image was made of, which is the only one of the three a rerun on
    another host can be compared against.
    """
    return {
        "kind": "two-domain",
        "network": network,
        "gateway_url": gateway_url(server),
        "agent": {
            "container": agent,
            "image": agent_image,
            "image_id": image_id(agent_image),
            "build": dict(images.get(agent_image, {})),
            "workdir": WORK,
            "mounts": mount_record(agent_mount_list),
            "environment": pinned.redacted(dict(agent_environment)),
            "credential": credential,
        },
        "server": {
            "container": server,
            "image": server_image,
            "image_id": image_id(server_image),
            "build": dict(images.get(server_image, {})),
            "mounts": mount_record(server_mount_list),
            "environment": dict(server_env),
        },
        "network_policy": (
            "the two containers share a private network and no port is published to the host; the "
            "gateway is the only listener bound to an address the agent's container can route to, "
            "and the durable service holding the history is bound to the server's own loopback; "
            "the agent keeps general egress, as the cell this one reruns did"
        ),
    }


# --- the probe --------------------------------------------------------------------------------

#: What is asked, from inside a container started exactly as the agent's is, of the things the
#: agent must not be able to reach. Each line prints a verdict, and the last line is the count of
#: checks that failed, which is what the caller reads. Lines beginning ``note`` are observations
#: rather than verdicts: the agent keeps general egress, so what it can reach out there is
#: recorded as the retained egress it is and never counted as isolation.
#:
#: What the script is told arrives as arguments rather than as environment, because one of the
#: things it checks is the environment: a probe that added six variables of its own could not
#: then say what the agent's process was handed.
PROBE_SCRIPT = r"""
set -u
RUN_DIR=$1
REPO=$2
CACHE=$3
SERVER=$4
GATEWAY_ROOT=$5
GATEWAY_URL=$6
EXPECTED=$7
FROM_IMAGE=$8
SERVER_NAMESPACE=$9
SERVER_LOOPBACK=${10}
failed=0
verdict() {
  if [ "$1" = 0 ]; then echo "ok    $2"; else echo "FAIL  $2"; failed=$((failed + 1)); fi
}
note() { echo "note  $1"; }
absent() {
  if [ -e "$1" ]; then verdict 1 "$2 ($1 is here)"; else verdict 0 "$2"; fi
}
reaches() { curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$1" 2>/dev/null || true; }
# A TCP connection rather than an HTTP request, because most of what a port could be is not a web
# server: a curl at a durable service's own protocol ends in an error the same way a curl at a
# closed port does, and a check that read those alike would call every one of them refused.
connects() {
  python3 -c 'import socket, sys
probe = socket.socket(socket.AF_INET6 if ":" in sys.argv[1] else socket.AF_INET)
probe.settimeout(3)
sys.exit(0 if probe.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0 else 1)' "$1" "$2" 2>/dev/null
}
absent "$RUN_DIR" "the run directory has no path in this container"
absent /grades "the grades are not mounted"
absent "$REPO" "the repository has no path in this container"
absent "$CACHE" "the benchmark cache has no path in this container"
found=$(find / -xdev \( -name cell.json -o -name stream.sqlite -o -name 'run.json' \) 2>/dev/null)
deeper=$(find /work /root /cfg -name cell.json -o -name stream.sqlite 2>/dev/null)
if [ -z "$found$deeper" ]; then
  verdict 0 "no roster, history or run record anywhere on this filesystem"
else
  verdict 1 "a roster, history or run record is readable here: $found$deeper"
fi
bound=$(awk '{print $5}' /proc/self/mountinfo | grep -v -E '^(/proc|/sys|/dev)($|/)|^/etc/(resolv.conf|hostname|hosts)$|^/$' | sort -u | tr '\n' ' ')
if [ "$bound" = "/cfg /root /work " ]; then
  verdict 0 "the only host directories bound here are /work, /root and /cfg"
else
  verdict 1 "this container is bound to $bound"
fi
held=$(ls -A /cfg | tr '\n' ' ')
named=$(jq -r '.mcpServers[].url' /cfg/claude.mcp.json 2>/dev/null | tr '\n' ' ')
if [ "$held" = "claude.mcp.json " ] && [ "$named" = "$GATEWAY_URL " ]; then
  verdict 0 "/cfg holds the endpoint this run serves and nothing else"
else
  verdict 1 "/cfg holds $held naming $named"
fi
if touch /cfg/probe 2>/dev/null; then
  rm -f /cfg/probe
  verdict 1 "/cfg can be written, so the endpoint a later call reads is not fixed"
else
  verdict 0 "/cfg is read only"
fi
names=$(tr '\0' '\n' < /proc/1/environ | sed 's/=.*//' | sort | tr '\n' ' ')
missing=
extra=
for name in $EXPECTED; do
  case " $names " in *" $name "*) ;; *) missing="$missing $name" ;; esac
done
for name in $names; do
  case " $EXPECTED $FROM_IMAGE " in *" $name "*) ;; *) extra="$extra $name" ;; esac
done
if [ -z "$missing$extra" ]; then
  verdict 0 "this process was handed the launch's environment and nothing besides the image's"
else
  verdict 1 "the environment is missing$missing and carries unrecorded$extra"
fi
processes=$(ls -d /proc/[0-9]* | wc -l)
if [ "$$" = 1 ] && [ "$processes" -le 10 ]; then
  verdict 0 "the PID namespace is this container's own: it is PID 1 and sees $processes processes"
else
  verdict 1 "this shell is PID $$ and $processes processes are visible, so the host's are here"
fi
mine=$(readlink /proc/1/ns/net 2>/dev/null || true)
if [ -n "$mine" ] && [ -n "$SERVER_NAMESPACE" ] && [ "$mine" != "$SERVER_NAMESPACE" ]; then
  verdict 0 "the network namespace is this container's own: $mine, and the server is in $SERVER_NAMESPACE"
else
  verdict 1 "this container is in network namespace ${mine:-none} and the server in ${SERVER_NAMESPACE:-none}"
fi
caps=$(sed -n 's/^CapBnd:[[:space:]]*//p' /proc/self/status)
seccomp=$(sed -n 's/^Seccomp:[[:space:]]*//p' /proc/self/status)
if [ $((0x$caps & 0x200000)) = 0 ] && [ "$seccomp" != 0 ]; then
  verdict 0 "this container is unprivileged (CapBnd $caps) and seccomp filtered"
else
  verdict 1 "this container has CapBnd $caps and seccomp $seccomp"
fi
sockets=
for path in /var/run/docker.sock /run/docker.sock /var/run/containerd/containerd.sock; do
  if [ -e "$path" ]; then sockets="$sockets $path"; fi
done
if [ -z "$sockets" ]; then
  verdict 0 "no docker or containerd socket is mounted here"
else
  verdict 1 "a container runtime is reachable through$sockets"
fi
for endpoint in "$SERVER:2375" "$SERVER:2376" host.docker.internal:2375 host.docker.internal:2376; do
  code=$(reaches "http://$endpoint/version")
  if [ "$code" = 000 ]; then
    verdict 0 "no docker API answers on $endpoint"
  else
    verdict 1 "a docker API answered $code on $endpoint"
  fi
done
for endpoint in http://169.254.169.254/latest/meta-data/ http://169.254.169.254/metadata/instance http://metadata.google.internal/computeMetadata/v1/; do
  code=$(reaches "$endpoint")
  if [ "$code" = 000 ]; then
    verdict 0 "no instance metadata answers at $endpoint"
  else
    verdict 1 "instance metadata answered $code at $endpoint, so this host's credentials are here"
  fi
done
address=$(getent hosts "$SERVER" | awk '{print $1}' | head -1)
if [ -n "$address" ]; then
  verdict 0 "$SERVER resolves to $address, so this run's own network is what was measured"
else
  verdict 1 "$SERVER does not resolve, so nothing below was asked of this run's server"
fi
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$GATEWAY_ROOT/")
if [ "$code" = 404 ]; then
  verdict 0 "the gateway serves nothing at its root ($code)"
else
  verdict 1 "the gateway answered $code at its root"
fi
for path in cell.json stream.sqlite grades/cell.json blobs; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$GATEWAY_ROOT/$path")
  if [ "$code" = 404 ]; then
    verdict 0 "the gateway serves nothing at /$path ($code)"
  else
    verdict 1 "the gateway answered $code at /$path"
  fi
done
code=$(curl -s -L -o /dev/null -w '%{http_code}' --max-time 10 "$GATEWAY_URL")
if [ "$code" = 400 ] || [ "$code" = 405 ] || [ "$code" = 406 ]; then
  verdict 0 "the endpoint wants a handshake and answers no file read ($code)"
else
  verdict 1 "the endpoint answered $code to a plain read"
fi
for port in 7233 8233 8080 9001 5432; do
  if connects "$SERVER" "$port"; then
    verdict 1 "$SERVER:$port answered, so something beside the gateway is reachable"
  else
    verdict 0 "$SERVER:$port refuses, so the gateway is the only way in"
  fi
done
for socket in $SERVER_LOOPBACK; do
  if connects "${socket%:*}" "${socket##*:}"; then
    verdict 1 "$socket answers here, so the server's loopback is this container's loopback"
  else
    verdict 0 "$socket does not answer here, so it is on a loopback this container is not on"
  fi
done
host=$(getent hosts host.docker.internal | awk '{print $1}' | head -1)
note "general egress is retained, as the cell this one reruns retained it: the host answers to \
${host:-no name here} and the internet is reachable, so what these checks establish is that this \
run's roster, history and grades are not among the things out there"
echo "failed=$failed"
"""


def probe_command(
    *,
    run_dir: Path,
    cache: Path,
    server: str,
    environment: Sequence[str],
    server_namespace: str = "",
    server_loopback: Sequence[str] = (),
) -> List[str]:
    """The probe as a command, carrying what it checks as arguments rather than as environment.

    ``environment`` is the list of names the launch hands the agent's container, which the probe
    asks its own process about: the point of running it as the agent is that it can be asked what
    the agent would have been handed, and a probe told through the environment could not.

    ``server_namespace`` and ``server_loopback`` are the server's half of the network claim, read
    off the server before this runs. They are what the questions are asked against: the namespace
    to compare with its own, and the addresses whose answering here would mean the two containers
    are in one network stack rather than on one network. Told neither, the probe fails the checks
    it cannot make.
    """
    return [
        "bash",
        "-c",
        PROBE_SCRIPT,
        "probe",
        str(run_dir),
        str(REPO),
        str(cache),
        server,
        f"http://{server}:{SERVER_PORT}",
        gateway_url(server),
        " ".join(sorted(environment)),
        " ".join(IMAGE_ENVIRONMENT),
        server_namespace,
        " ".join(server_loopback),
    ]


def read_probe(output: str) -> int:
    """How many of the probe's checks failed, read off its last line."""
    for line in reversed(output.splitlines()):
        if line.startswith("failed="):
            return int(line.partition("=")[2])
    raise ValueError("the probe printed no verdict, so what it found is unknown")


__all__ = [
    "AGENT_DOCKERFILE",
    "AGENT_HOME",
    "AGENT_IMAGE",
    "BUILD_LABEL",
    "CACHE_MOUNT",
    "CONFIG_MOUNT",
    "GENERATED",
    "GRADES_MOUNT",
    "IMAGE_ENVIRONMENT",
    "MCP_CONFIG",
    "PROBE_SCRIPT",
    "PROVISION",
    "SERVER_DOCKERFILE",
    "SERVER_IMAGE",
    "SERVER_LOCK",
    "SERVER_PORT",
    "SERVER_SOURCE",
    "SOURCE_CACHE",
    "UV_VERSION",
    "WORK",
    "agent_argv",
    "agent_build_inputs",
    "agent_mounts",
    "build_identity",
    "build_images",
    "cache_mounts",
    "cli_version_command",
    "create_network",
    "default_cache",
    "docker_available",
    "file_digest",
    "gateway_url",
    "generated",
    "image_build",
    "image_id",
    "listening_sockets",
    "loopback_listeners",
    "mount_record",
    "names",
    "network_namespace",
    "parse_listeners",
    "probe_command",
    "provision_argv",
    "provision_source",
    "provisioner_name",
    "read_probe",
    "remove_container",
    "remove_network",
    "save_logs",
    "served_requests",
    "server_argv",
    "server_build_inputs",
    "server_context",
    "server_environment",
    "server_mounts",
    "source_files",
    "topology",
    "unexpected_listeners",
    "wait_for_gateway",
    "write_context",
]
