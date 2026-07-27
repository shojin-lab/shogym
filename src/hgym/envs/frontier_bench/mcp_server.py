"""In-process MCP server for ``frontier_bench`` — a task's Docker shell, served as tools.

Frontier-Bench (Harbor) separates the *agent* (which acts on the world only through a shell:
``exec`` + file up/download) from the *verifier* (which scores the container end-state). hgym
is the agent driver: this server builds+starts the task's environment container on
``begin_session`` and exposes its shell as MCP tools —

  - **``exec(command)``** — run a shell command in the task container (Harbor
    ``BaseEnvironment.exec``); returns ``{ok, exit_code, stdout, stderr}``.
  - **``read_file(path)`` / ``write_file(path, content)``** — file I/O into the container
    (Harbor ``download_file`` / ``upload_file``).
  - **``done()``** — the env's ``score`` terminal. The serve layer does not dispatch it as an
    ordinary tool: a ``done`` call *seals* the episode and the env's ``finalize`` hook runs the
    verdict via :func:`finalize_session` — collect the task's declared ``artifacts`` off the
    (still-live) container end-state, run the task's verifier (SEPARATE mode — a second
    container), read the 0/1 from ``/logs/verifier/reward.txt``, and return a
    :class:`~hgym.envs.frontier_bench.docker_backend.VerifierOutcome`. Because the seal makes it
    structurally one-shot, there is no marker/one-shot bookkeeping here — a post-seal ``done``
    (or any other call) is tombstoned by the serve layer and never re-runs the verifier.

Each episode gets its own container, keyed by ``_session_id`` (hgym injects it), torn down on
``end_session`` *after* ``finalize`` has committed the verdict. Importing this module needs
**no** Docker; only ``begin_session`` / ``finalize_session`` shell out. The tool *schemas* list
without Docker, so ``describe`` and the manifest probe stay offline.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from hgym.envs.frontier_bench import docker_backend as dk
from hgym.envs.frontier_bench.docker_backend import VerifierOutcome
from hgym.envs.frontier_bench.manifest import FrontierTask, load_task

EXEC_TOOL_NAME = "exec"
READ_FILE_TOOL_NAME = "read_file"
WRITE_FILE_TOOL_NAME = "write_file"
DONE_TOOL_NAME = "done"

DEFAULT_COMMAND_TIMEOUT_SECONDS = 600.0

server: FastMCP = FastMCP(name="frontier_bench")

# session_id -> _Session
_sessions: Dict[str, "_Session"] = {}
_lock = threading.RLock()


def _image_tags(task: FrontierTask) -> tuple[str, str]:
    """Deterministic, content-addressed image tags so repeated episodes reuse a cached build."""
    short = task.content_sha256[:12]
    return (
        f"hgym-frontier-env-{task.name}:{short}",
        f"hgym-frontier-verifier-{task.name}:{short}",
    )


def build_task_image(task_name: Optional[str] = None) -> str:
    """Build the task's environment image (if not already cached) and return its tag.

    This is the *same* build ``begin_session`` performs, factored out so it can be exercised
    up front (e.g. an example preflight) — a Dockerfile / base-image-pull / platform failure
    surfaces here rather than crashing the served server on the first tool call. Serving then
    reuses the content-addressed image from cache. Needs Docker.
    """
    task = load_task(task_name)
    env_tag, _ = _image_tags(task)
    if not dk.image_exists(env_tag):
        dk.build_image(
            context_dir=task.environment_dir,
            dockerfile=task.environment_dockerfile,
            tag=env_tag,
            timeout=task.build_timeout_sec,
        )
    return env_tag


class _Session:
    """One episode: a built+running task container plus its pinned task metadata."""

    def __init__(
        self,
        *,
        task: FrontierTask,
        command_timeout_seconds: float,
        keep_container: bool,
    ) -> None:
        self.task = task
        self.command_timeout_seconds = command_timeout_seconds
        self.keep_container = keep_container
        _, self._verifier_tag = _image_tags(task)
        # Build the agent image (cached by content-addressed tag) and start the container.
        self._env_tag = build_task_image(task.name)
        self.container = dk.Container(
            image=self._env_tag,
            workdir="/app",
            cpus=task.cpus,
            memory_mb=task.memory_mb,
        )
        self.container.start()

    # ----- shell surface -----

    def exec(self, command: str) -> Dict[str, Any]:
        r = self.container.exec(command, timeout=self.command_timeout_seconds)
        return {
            "ok": r.ok,
            "exit_code": r.exit_code,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }

    def read_file(self, path: str) -> Dict[str, Any]:
        r = self.container.read_file(path)
        return {
            "ok": r.ok,
            "exit_code": r.exit_code,
            "content": r.stdout if r.ok else "",
            "stderr": r.stderr,
        }

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        r = self.container.write_file(path, content)
        return {"ok": r.ok, "exit_code": r.exit_code, "stderr": r.stderr}

    # ----- terminal verdict -----

    def finalize(self) -> VerifierOutcome:
        """Run the task's verifier over the container end-state; return the recorded outcome.

        Invoked by the env's ``finalize`` hook on the *already-sealed* episode (via
        :func:`finalize_session`), while this container is still live. Returns the raw
        :class:`~hgym.envs.frontier_bench.docker_backend.VerifierOutcome` (reward + provenance);
        the env stamps it into core-owned terminal evidence, surfacing only the public-safe
        reward/success/artifact fields to the agent and keeping the verifier's stdout/exit-code
        in the private diagnostic.

        No one-shot bookkeeping lives here anymore: the seal makes ``done`` structurally
        one-shot (a post-seal call is tombstoned by the serve layer and never reaches here), so
        the old ``_graded`` guard — which protected the legacy re-grade path — is redundant.
        """
        return dk.run_separate_verifier(
            agent=self.container,
            tests_context_dir=self.task.tests_dir,
            tests_dockerfile=self.task.tests_dockerfile,
            verifier_image_tag=self._verifier_tag,
            artifacts=self.task.artifacts,
            build_timeout=self.task.build_timeout_sec,
            verifier_timeout=self.task.verifier_timeout_sec,
            keep_image=self.keep_container,
        )

    def run_oracle(self) -> Dict[str, Any]:
        """Upload the vendored ``solution/`` and run ``solve.sh`` (Harbor's ``oracle`` agent).

        Not an agent-facing tool — a session helper the Docker-gated sanity gate uses to
        confirm the task's own oracle scores 1 through this port's verifier.
        """
        self.container.exec("mkdir -p /solution", user="root")
        for item in sorted(self.task.solution_dir.iterdir()):
            self.container.copy_in(item, f"/solution/{item.name}")
        self.container.exec("chmod +x /solution/solve.sh", user="root")
        r = self.container.exec(
            "bash /solution/solve.sh",
            workdir="/app",
            user="root",
            timeout=self.task.verifier_timeout_sec,
        )
        return {"ok": r.ok, "exit_code": r.exit_code, "stdout": r.stdout, "stderr": r.stderr}

    def teardown(self) -> None:
        if not self.keep_container:
            self.container.stop()


# ----- session lifecycle (called in-process by the env) -----


def begin_session(
    session_id: str,
    *,
    task_name: Optional[str] = None,
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    keep_container: bool = False,
) -> None:
    """Build+start a fresh task container for this episode. Needs Docker."""
    task = load_task(task_name)
    session = _Session(
        task=task,
        command_timeout_seconds=command_timeout_seconds,
        keep_container=keep_container,
    )
    with _lock:
        old = _sessions.pop(session_id, None)
        _sessions[session_id] = session
    if old is not None:
        old.teardown()


def finalize_session(session_id: str) -> Optional[VerifierOutcome]:
    """Run the sealed episode's verifier over its container end-state; return the outcome.

    Called by the env's ``finalize`` hook (off the event loop, in a worker thread) after the
    serve layer has sealed the episode and BEFORE ``end_session`` tears the container down, so
    the container this reads is the agent's true final state. Returns ``None`` if there is no
    live session for ``session_id`` (a broken state the env fails closed on).
    """
    session = _session_for(session_id)
    if session is None:
        return None
    return session.finalize()


def end_session(session_id: str) -> None:
    """Tear down a finished episode's container. Idempotent."""
    with _lock:
        session = _sessions.pop(session_id, None)
    if session is not None:
        session.teardown()


def reset_state() -> None:
    """Drop all sessions (test hygiene)."""
    with _lock:
        ids = list(_sessions)
    for session_id in ids:
        end_session(session_id)


def _session_for(session_id: Optional[str]) -> Optional["_Session"]:
    if session_id is None:
        return None
    with _lock:
        return _sessions.get(session_id)


def _no_session_error(tool: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"session not initialized; env did not call begin_session ({tool})",
        }
    )


# ----- MCP tools -----


@server.tool
def exec(command: str, _session_id: str) -> str:
    """Run one shell command inside the task container and return its result.

    ``command`` is a full shell command line (e.g. ``"ls /app/inputs"``,
    ``"python3 /app/run.py"``). Returns JSON with ``ok``, ``exit_code``, ``stdout``,
    ``stderr``. The task's inputs live under ``/app/inputs``; write outputs under ``/app``.
    """
    session = _session_for(_session_id)
    if session is None:
        return _no_session_error("exec")
    return json.dumps(session.exec(command))


@server.tool
def read_file(path: str, _session_id: str) -> str:
    """Read a file from the task container. Returns JSON with ``ok`` and ``content``."""
    session = _session_for(_session_id)
    if session is None:
        return _no_session_error("read_file")
    return json.dumps(session.read_file(path))


@server.tool
def write_file(path: str, content: str, _session_id: str) -> str:
    """Write ``content`` to ``path`` inside the task container (parent dirs created)."""
    session = _session_for(_session_id)
    if session is None:
        return _no_session_error("write_file")
    return json.dumps(session.write_file(path, content))


@server.tool
def done(_session_id: str) -> str:
    """Finish the task: seal the episode and run the verifier over the container end-state.

    This is the env's ``score`` terminal. Calling it ends the episode: the serve layer seals the
    run and the env's ``finalize`` hook runs the task's verifier over the container's final state
    (collecting the declared artifacts, running the pytest checks in a separate container) and
    returns the recorded 0/1 reward and artifact-collection status. The verifier's own stdout /
    exit code are never returned, and ``done`` is structurally one-shot — after it, every further
    tool call is tombstoned, so a failing verdict cannot be inspected and re-graded. Make sure the
    required outputs are written before calling it.

    (The serve layer intercepts ``done`` as the score terminal and does not dispatch this handler
    on the sealed path; this body is a defensive inert fallback for any non-sealed invocation.)
    """
    return json.dumps(
        {"note": "done is the score terminal; the serve layer seals and finalizes it"}
    )


__all__ = [
    "server",
    "begin_session",
    "finalize_session",
    "end_session",
    "reset_state",
    "build_task_image",
    "EXEC_TOOL_NAME",
    "READ_FILE_TOOL_NAME",
    "WRITE_FILE_TOOL_NAME",
    "DONE_TOOL_NAME",
]
