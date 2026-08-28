# This file is run by an interpreter this port provisions, not by the one shogym runs under, and
# `appworld` cannot be installed beside shogym at all (it pins `pydantic<2`). Its imports are
# expected to be unresolved in the type-check environment.
# pyright: reportMissingImports=false
"""One episode's world, in an interpreter of its own, behind a loopback port.

Two reasons this is a process and not an object, and either would be enough on its own.

**AppWorld cannot share an interpreter with shogym.** It pins ``pydantic<2``; shogym's MCP layer
needs ``pydantic>=2.7``. There is no environment that satisfies both, so the world runs under an
interpreter this port provisions for it and the two talk over a socket. That is also why this file
imports nothing from shogym: it is run by path, under a Python that has never heard of shogym, and
its only imports are the standard library and ``appworld``.

**AppWorld cannot share an interpreter with itself.** It freezes the clock with a library that
patches ``datetime`` for the whole process, every app's model classes hold their database engine
on a class attribute, and building a second world calls ``close_all()`` on the first. Two episodes
in one process are one episode the other keeps unfreezing. So an episode gets a process, and the
process gets a port.

**The port is bound to loopback and gated by a token read from stdin at startup.** AppWorld's own
environment server publishes ``evaluate``, ``save_state`` and ``load_state`` with no
authentication, on every interface. This one answers nothing without the token. The token and the
corpus root arrive on stdin rather than on the command line, because the code an agent writes runs
inside this process and ``sys.argv`` is one attribute lookup away from it; stdin is read once,
before any world exists, and closed.

**What this process must not be able to tell an agent, it does not hold.** The world is built with
``load_ground_truth=False``, so the answers and the base task's checks are not objects in the
process that runs agent-authored code, and there is no evaluator here to call. Grading the base
task happens in a second, short-lived process that never runs agent code (:func:`grade`), reading
the end state upstream flushed to disk. The convention the filing is graded against is never sent
to either: there is no field in the protocol for one and no comparison in this file.

**And no lifecycle fact comes from this process either.** There is no seal, quiesce, read or close
command. The writer on this process's own socket is reachable from inside the interpreter that
executes agent-authored Python, so a reply saying "I have stopped", "nothing else is running" or
"the filing says X" is a reply the episode could have written. The host stops this process's group,
confirms it is gone by reading the process table itself, and grades the tree on disk. Nothing here
has to flush a final state, because upstream already did: ``AppWorld.execute`` ends with its own
save into the episode's output tree and ``initialize`` writes one before any block runs, so an
episode that ran N blocks is graded on the state after block N and one that ran none is graded on
its opening state. Work an agent's thread does after its last block is lost rather than scored,
which is the same rule the block budget already states.

**What it still cannot do is contain an agent.** The code an agent writes runs as this process,
with this process's filesystem and network. The environment is scrubbed to an allow-list and the
working directory is a scratch directory, so an inherited API key or a relative path to the run's
own records is not simply lying there, and the corpus this process is given carries no
``ground_truth`` directory at all. It is not a sandbox: an agent that goes looking can still read
whatever the user running it can read. The port's README says so in those words, and a run whose
scores have to survive an adversary needs a container around this process, not a longer allow-list
inside it.

Usage::

    python worker.py serve            # {"root": ..., "token": ..., "keepalive": ...} on stdin
    python worker.py grade            # {"root", "task_id", "experiment", "filing", "keepalive"}
    python worker.py install
    python worker.py unpack --bundle <bundle> --into <directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

#: The header every request must carry.
#:
#: **A gate on the socket, not a secret from the agent.** It is kept out of argv, the environment,
#: the tool schemas and the results, so nothing hands it over; but it lives in this process's own
#: handler state, and this is the process that runs agent-authored code. Code running here can
#: reach it, and no arrangement inside one interpreter changes that.
#:
#: What it is for is the boundary it can actually hold: another process on this machine cannot
#: drive this worker's world without it. Read it as a cross-process gate, and do not build
#: anything on it being unknown to the episode.
TOKEN_HEADER = "X-Shogym-Worker-Token"


#: Where the generator's state digest is written, beside the episode's databases. Beside rather
#: than inside them, because upstream's saver clears that directory on every save.
RNG_DIGEST_FILE = "rng.digest"


class Episode:
    """The one world this process serves."""

    def __init__(self) -> None:
        self.world: Any = None

    # ----- seeding, before any world exists -----

    def seed(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Write a task's database log with extra rows added, through the model layer.

        The rows are created rather than hand-written as SQL. A task's database file is a
        replayable statement log with a per-row hash column and full-text shadow tables behind it,
        and an insert that got either of those wrong would be a world that looks right and diffs
        wrong. Creating the rows through the models is the same code path AppWorld uses for its
        own state, so what lands in the log is correct by construction.

        The clock is frozen at the world's own datetime first, so every row's creation timestamp
        is the world's "today" and two builds of one task produce the same bytes."""
        from appworld.apps.api_lib import (
            save_local_dbs,
            set_local_date_and_time,
            unset_local_date_and_time,
        )
        from appworld.collections.models import ModelCollection

        moment = _parse_datetime(body["datetime"])
        freezer = set_local_date_and_time(moment)
        try:
            models = ModelCollection.load(
                to_db_home_path=f":memory:shogym-seed-{body['tag']}",
                from_db_home_path=body["from_dbs"],
                load_apps=["todoist"],
            )
            todoist = models.todoist
            user = todoist.User.find_one(email=body["supervisor_email"])
            if user is None:
                raise ValueError(f"no todoist account for {body['supervisor_email']!r}")
            project = todoist.Project.create(
                user_id=user.id,
                name=body["project"]["name"],
                description=body["project"]["description"],
            )
            project.save()
            todoist.ProjectCollaboratorLink.create(project_id=project.id, user_id=user.id).save()
            for order, name in enumerate(body["sections"]):
                todoist.Section.create(
                    project_id=project.id, name=name, order_index=order
                ).save()
            for order, request in enumerate(body["requests"]):
                todoist.Task.create(
                    project_id=project.id,
                    section_id=None,
                    user_id=user.id,
                    title=request["title"],
                    description=request["description"],
                    due_date=None,
                    priority=body["priority"],
                    duration=None,
                    duration_unit=None,
                    order_index=order,
                ).save()
            written = _save_one_log(save_local_dbs, models, body["into"])
        finally:
            unset_local_date_and_time(freezer)
        return {"rows": 2 + len(body["sections"]) + len(body["requests"]), "into": written}

    # ----- the episode -----

    def open(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Build the world and report what the agent is allowed to know about it.

        ``seed`` is what AppWorld seeds the global generator with when it builds the world's
        requester, and the caller chooses it from the task alone. The process's own generator is
        this worker's alone, so there is nobody to hand it back to: AppWorld saves databases and
        not generator state, so a world replayed from its databases alone agrees on its contents
        and disagrees on its next draw, and the digest recorded beside them is what makes that
        checkable."""
        from appworld import AppWorld

        self.world = AppWorld(
            task_id=body["task_id"],
            experiment_name=body["experiment"],
            random_seed=int(body["seed"]),
            # The answers are not loaded into the process that runs the agent's code. Grading the
            # base task reads the end state from disk in a process of its own, so nothing here
            # holds a check, an expected value or an evaluator to call.
            load_ground_truth=False,
        )
        task = self.world.task
        self._record_rng()
        return {
            "instruction": task.instruction,
            "supervisor": dict(task.supervisor),
            "datetime": task.datetime.isoformat(),
        }

    def execute(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Run one snippet of the agent's code against the world and return what it printed.

        **Upstream persists the world at the end of every execution, and that is what is graded.**
        ``AppWorld.execute`` ends with its own ``_save_state`` into the episode's output tree, and
        ``initialize`` writes one before any block runs. So the tree on disk is always the world as
        it stood when the last block finished, and nothing here has to be asked to produce a final
        one. That matters because this process runs agent-authored code: an answer from it saying
        "I have flushed" or "I have stopped" is an answer the agent could have written.

        The generator digest goes to the same tree for the same reason. It is a fact about this
        interpreter that no file could otherwise carry, and it is a diagnostic rather than a score,
        but a diagnostic read out of a reply is a diagnostic the episode can choose."""
        output = self.world.execute(str(body["code"]))
        self._record_rng()
        return {"output": output}

    def _record_rng(self) -> None:
        """Write the generator's state digest beside this episode's databases.

        Beside rather than inside: upstream's saver owns the ``dbs`` directory and clears it on
        every save, so a file written in there would last until the next block."""
        try:
            beside = os.path.dirname(self.world.output_db_home_path_on_disk)
            with open(os.path.join(beside, RNG_DIGEST_FILE), "w") as handle:
                handle.write(_digest(repr(random.getstate())))
        except Exception:
            # A diagnostic that cannot be written is a diagnostic that is missing, which the
            # grader reports as absent. It is not worth failing an episode over.
            pass


# ----- reading and writing the world -----


def _read_filing(models: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    """The earliest row titled ``body['title']`` in the named project, and its stored fields.

    Several such rows is not several answers: the earliest is read and the count is kept, so
    filing three logs and hoping one lands is visible in the record and worth no more than filing
    one. The label's colour counts only once the label is on that row, because creating a label
    and never attaching it is the same amount of work as attaching it and the instruction asks for
    both."""
    empty = {
        "filed": False,
        "rows": 0,
        "lines": [],
        "section": None,
        "color": None,
        "unit": None,
        "priority": None,
        "duration": None,
    }
    todoist = models.todoist
    user = todoist.User.find_one(email=body["supervisor_email"])
    if user is None:
        return empty
    project = todoist.Project.find_one(user_id=user.id, name=body["project"])
    if project is None:
        return empty
    rows = [t for t in todoist.Task.find_all(project_id=project.id) if t.title == body["title"]]
    rows.sort(key=lambda row: (row.created_at, row.id))
    label = todoist.Label.find_one(user_id=user.id, name=body["label"])
    color = label.color if label is not None else None
    if label is not None and rows:
        links = todoist.TaskLabelLink.find_all(task_id=rows[0].id)
        if not any(link.label_id == label.id for link in links):
            color = None
    if not rows:
        # A label with no row to carry it is a colour nobody chose for anything: the instruction
        # asks for the label *and* for it to be attached, and half of that is not a choice made.
        return empty
    first = rows[0]
    sections = {s.id: s.name for s in todoist.Section.find_all(project_id=project.id)}
    return {
        "filed": True,
        "rows": len(rows),
        "lines": (first.description or "").splitlines(),
        "section": sections.get(first.section_id),
        "color": color,
        "unit": first.duration_unit,
        "priority": first.priority,
        "duration": first.duration,
    }


def _save_one_log(save_local_dbs: Any, models: Any, into: str) -> str:
    """Save the todoist statement log to ``into``, leaving the rest of the directory alone.

    Upstream's saver owns a whole directory and clears it, so it writes to a scratch directory and
    the one file that matters is moved across. The other apps' logs beside ``into`` are the task's
    own and have to survive."""
    directory = os.path.dirname(into)
    scratch = os.path.join(directory, ".todoist.building")
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        save_local_dbs(
            from_db_home_path=models.todoist.SQLModel.db.home_path,
            to_db_home_path=scratch,
            format="changes",
            app_names=["todoist"],
            delete_if_exists=True,
            skip_mandatory_apps=True,
            save_model_hashes=False,
            vaccum=False,
        )
        os.replace(os.path.join(scratch, "todoist.jsonl"), into)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return into


class _ignoring:
    """Add names to every changed-model comparison's ignore list for the length of the body.

    A task's evaluation asserts which models a scenario changed, and a chore appended to every
    instruction changes more. Without this every scenario would fail an assertion about the chore
    rather than about the agent. The names are merged into whatever the task already ignores; a
    task that passes an explicit include list is left alone, because an include list already says
    exactly which models it is looking at."""

    def __init__(self, names: List[str]) -> None:
        self.names = names
        self.original: Any = None

    def __enter__(self) -> "_ignoring":
        from appworld.collections.models import ModelCollectionPair

        self.original = ModelCollectionPair._changed_model_names
        original, extra = self.original, self.names

        def with_extras(pair: Any, include: Any = None, ignore: Any = None) -> Any:
            if include:
                return original(pair, include=include, ignore=ignore)
            merged = list(ignore or [])
            merged.extend(name for name in extra if name not in merged)
            return original(pair, include=None, ignore=merged)

        ModelCollectionPair._changed_model_names = with_extras
        return self

    def __exit__(self, *exc: Any) -> None:
        from appworld.collections.models import ModelCollectionPair

        ModelCollectionPair._changed_model_names = self.original


def _test_data(task: Any) -> List[Dict[str, Any]]:
    """The task's own list of checks, in its own order."""
    ground_truth = task.ground_truth
    return list(ground_truth.test_data) if ground_truth is not None else []


def _parse_datetime(text: str) -> Any:
    from datetime import datetime

    return datetime.fromisoformat(text)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _directory_digest(directory: str) -> str:
    """A digest over every database log in ``directory``, file by file in a fixed order."""
    digest = hashlib.sha256()
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".jsonl"):
            continue
        digest.update(name.encode())
        with open(os.path.join(directory, name), "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


# ----- the server -----


def build_handler(episode: Episode, token: str, server: List[Any]) -> type:
    """The request handler: one method per protocol command, and a token on every one of them.

    **There are no lifecycle commands, and that is the point.** ``seal``, ``quiesce``, ``read``
    and ``close`` are gone: this is the process that runs agent-authored Python, so a reply from
    it saying "I have stopped", "nothing else is running" or "the filing says X" is a reply the
    episode could have written. The host stops this process's group, confirms it is gone by
    reading the process table itself, and grades the tree upstream already wrote (see
    :meth:`Episode.execute`). What is left here is the two commands that have to come from the
    world because only the world can answer them."""
    commands = {
        "/seed": episode.seed,
        "/open": episode.open,
        "/execute": episode.execute,
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 (the base class names it)
            if self.headers.get(TOKEN_HEADER) != token:
                # No detail, and the same answer for a missing token and a wrong one: a caller
                # that did not bring the token is not one this process owes an explanation.
                self._answer(403, {"error": "forbidden"})
                return
            command = commands.get(self.path)
            if command is None:
                self._answer(404, {"error": "no such command"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                self._answer(200, {"output": command(body)})
            except Exception as exc:  # the world's failures are answers, not crashes
                self._answer(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

        def _answer(self, status: int, payload: Dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            """Silence. Every line here would name the task and the port on the parent's stderr."""

    return Handler


def grade(body: Dict[str, Any]) -> Dict[str, Any]:
    """The base task's own checks, in the order the task lists them, in a process of its own.

    This is the only place ground truth is loaded, and it happens after the world is sealed, in a
    process that has never run a line the agent wrote. It reads the end state the serving worker
    flushed to disk, so nothing about the world has to be carried across.

    Checks are reported by position rather than by requirement text: a requirement is a paragraph
    of English naming the models and values it asserts on, so carrying one out of here would carry
    part of the task's answer with it. The position is the task's own order and is all a row
    identifier has to be.

    **The filing and the digests are read here too, and that is the change.** They used to be
    asked of the serving world over the protocol, which meant the process that runs agent-authored
    code was the process reporting what the episode had done. Now one process reads one stopped
    tree: the filing, the databases' digest, the generator digest and the evaluator's verdicts all
    come from the same bytes, so they cannot be two instants, and none of them can be composed by
    the episode."""
    os.environ["APPWORLD_ROOT"] = body["root"]
    from appworld.evaluator import evaluate_task
    from appworld.task import Task

    dbs = os.path.join(body["experiment"], "tasks", body["task_id"], "dbs")
    # Before the evaluator, and in this order for a mechanical reason rather than a preference:
    # AppWorld holds each app's database engine on a class attribute, so a collection loaded after
    # the evaluator has built its own reads the evaluator's world instead of the one it asked for.
    # Both read the same tree either way, because the process that could have changed it is gone;
    # what the order buys is that this one reads what it named.
    filing = _read_filing(_models_on_disk(dbs, body["task_id"]), body["filing"])
    world_digest = _directory_digest(dbs)
    rng_digest = _read_rng(os.path.dirname(dbs))
    with _ignoring(list(body.get("ignore") or ())):
        tracker = evaluate_task(
            task_id=body["task_id"],
            experiment_name=body["experiment"],
            suppress_errors=True,
            # Upstream writes `evaluation/report.md` beside the episode's output by default, and
            # that report quotes the requirement prose and the expected values behind it. It is a
            # grade, written into a tree, and a tree is a thing another episode can read. Nothing
            # here needs it: the verdicts come back through the protocol.
            save_report=False,
        )
    reported = tracker.to_dict(stats_only=False)
    passed = {entry["requirement"] for entry in reported["passes"]}
    failed = {entry["requirement"] for entry in reported["failures"]}
    checks = []
    task = Task.load(task_id=body["task_id"])
    for position, entry in enumerate(_test_data(task), start=1):
        requirement = entry["requirement"]
        verdict = True if requirement in passed else (False if requirement in failed else None)
        checks.append(["aw.%03d" % position, verdict])
    return {
        "checks": checks,
        # Read above rather than off a live world, and from the same tree the evaluator just read.
        # The process that could have changed these bytes was stopped and confirmed gone before
        # this one started, so what is scored and what is graded are one state by construction
        # rather than by two observations happening to agree.
        "filing": filing,
        "world_digest": world_digest,
        "rng_digest": rng_digest,
    }


def _read_rng(beside: str) -> str:
    """The generator digest the serving world wrote, or the empty string if it wrote none.

    Read from the tree rather than asked of the world, for the same reason everything else here
    is: the process that could answer is the process that runs the agent's code."""
    try:
        with open(os.path.join(beside, RNG_DIGEST_FILE)) as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _models_on_disk(dbs: str, tag: str) -> Any:
    """Load one task's flushed databases into an interpreter of their own.

    Through upstream's own model layer rather than by parsing the statement log: the log is a
    replayable format with a per-row hash column and full-text shadow tables behind it, and a
    reader that got either wrong would be reading a world that is not the one that was saved."""
    from appworld.collections.models import ModelCollection

    return ModelCollection.load(
        to_db_home_path=f":memory:shogym-read-{tag}",
        from_db_home_path=dbs,
        load_apps=["todoist"],
    )


def _handshake() -> Dict[str, Any]:
    """The one line of stdin a worker is started with, and the last thing it reads from it.

    On stdin rather than on ``sys.argv`` because agent-authored code runs in this process, and
    ``sys.argv`` survives for the life of it while a line of stdin does not."""
    line = sys.stdin.readline()
    sys.stdin.close()
    return json.loads(line)


def watch_parent(descriptor: Optional[int]) -> None:
    """Stop this process, and everything it started, when the host that owns it goes away.

    **The loop below is ended from outside, and a host that died abruptly can no longer end it.**
    This process is started in a session of its own so that stopping the episode stops everything
    the episode spawned, which also means nothing reaps it when its parent dies: it is handed to
    init and goes on serving a world nobody is coming back for, holding a port and a scratch
    directory, while a resumed harness starts a second one. Neither the token nor the port nor the
    group number survives the parent, so there is nothing left that names it.

    So the parent holds one end of a pipe and this process holds the other. A read on it never
    returns anything, because nothing is ever written; what it returns is end-of-file, at the
    instant the last copy of the writing end is closed, which is the instant the parent exits
    however it exits. That is a fact from the kernel rather than a message from a process, which
    is the same standard everything else in this protocol is held to.

    The whole group is signalled and not just this process, for the reason the host signals a
    group: agent code runs here and is free to spawn, and a worker that exited politely on its own
    while leaving descendants behind would be the same orphan under a different name. Only when
    this process really leads that group, because a platform without ``setsid`` would otherwise
    have this signalling the host's own group.

    ``PR_SET_PDEATHSIG`` is asked for as well, and is not what this rests on: it is Linux only, and
    it fires on the death of the parent *thread* rather than the parent process. The pipe is what
    works on both platforms this port runs on, and the two together cover the window between the
    fork and this call.

    A daemon thread, because the read blocks forever in the ordinary case and the process must be
    free to exit around it."""
    _pdeathsig()
    if descriptor is None:
        return

    def _wait() -> None:
        try:
            while os.read(descriptor, 1):
                pass
        except OSError:
            pass
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpgrp(), signal.SIGKILL)
        os._exit(1)

    threading.Thread(target=_wait, daemon=True).start()


def _pdeathsig() -> None:
    """Ask Linux to kill this process when its parent thread exits, and shrug where it cannot.

    Belt to the pipe's braces, and never the mechanism: it does not exist on macOS, and on Linux
    it is armed against the parent thread rather than the parent process. Failure is silence,
    because everything this covers the pipe covers too."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        # PR_SET_PDEATHSIG
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


def serve(root: str, token: str, keepalive: Optional[int] = None) -> int:
    """Bind a loopback port, say which one, and serve until the host stops this process.

    One request at a time, and every one of them on the main thread. AppWorld runs an agent's code
    under a signal alarm, and the standard library only lets the main thread install one, so a
    handler on a worker thread fails on the first line of code an agent runs. Sequential is right
    anyway: there is one world here, and two commands into it at once is two commands into the
    same mutable state.

    **This loop is ended from outside and never from inside.** There is no close command to ask
    for; the host signals this process's group and confirms it is gone. A shutdown the protocol
    could request would be a shutdown agent-authored code could request, and the fact the host
    needs is that this process stopped rather than that it said so.

    ``keepalive`` is the one thing outside that is not a signal: a descriptor whose end-of-file
    means the host is gone (see :func:`watch_parent`). It is armed before the world is built,
    because building one is seconds in which the parent can die."""
    watch_parent(keepalive)
    os.environ["APPWORLD_ROOT"] = root
    episode = Episode()
    holder: List[Any] = []
    server = HTTPServer(("127.0.0.1", 0), build_handler(episode, token, holder))
    holder.append(server)
    # The parent learns the port from here and from nowhere else: binding zero and reporting back
    # is what keeps two episodes started at the same moment off each other's port.
    print(json.dumps({"port": server.server_port}), flush=True)
    server.serve_forever()
    return 0


def install() -> int:
    """Unpack the app sources the wheel ships packed."""
    from appworld.install import install_package

    install_package()
    return 0


def unpack(bundle: str, into: str) -> int:
    """Unpack a verified data bundle. The caller checks the digest; this only opens the archive."""
    from appworld.common.constants import PASSWORD, SALT
    from appworld.common.utils import unpack_bundle

    unpack_bundle(bundle_file_path=bundle, base_directory=into, password=PASSWORD, salt=SALT)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="an AppWorld world, behind a loopback port")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="one world; reads {root, token} from stdin")
    commands.add_parser("grade", help="one task's checks; reads {root, task_id, ...} from stdin")
    commands.add_parser("install")
    unpacking = commands.add_parser("unpack")
    unpacking.add_argument("--bundle", required=True)
    unpacking.add_argument("--into", required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        opening = _handshake()
        return serve(opening["root"], opening["token"], opening.get("keepalive"))
    if args.command == "grade":
        opening = _handshake()
        # Armed before the evaluator is loaded, for the reason `serve` arms it before it builds a
        # world: this process is short-lived but it is not instant, and a host that dies inside
        # its ten minutes would otherwise leave it running under init with nothing naming it.
        watch_parent(opening.get("keepalive"))
        print(json.dumps({"output": grade(opening)}), flush=True)
        return 0
    if args.command == "install":
        return install()
    return unpack(args.bundle, args.into)


if __name__ == "__main__":
    sys.exit(main())
