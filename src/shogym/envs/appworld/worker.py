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
the end state this one flushes to disk. The convention the filing is graded against is never sent
to either: there is no field in the protocol for one and no comparison in this file.

**What it still cannot do is contain an agent.** The code an agent writes runs as this process,
with this process's filesystem and network. The environment is scrubbed to an allow-list and the
working directory is a scratch directory, so an inherited API key or a relative path to the run's
own records is not simply lying there, and the corpus this process is given carries no
``ground_truth`` directory at all. It is not a sandbox: an agent that goes looking can still read
whatever the user running it can read. The port's README says so in those words, and a run whose
scores have to survive an adversary needs a container around this process, not a longer allow-list
inside it.

Usage::

    python worker.py serve            # {"root": ..., "token": ...} on stdin
    python worker.py grade            # {"root": ..., "task_id": ..., "experiment": ...} on stdin
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


class Episode:
    """The one world this process serves, and the randomness it was handed."""

    def __init__(self) -> None:
        self.world: Any = None
        self.caller_rng: Any = None

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
        requester, and the caller chooses it from the task alone. The process's own generator
        state is put aside first and handed back at close, so an episode leaves the randomness
        where it found it: AppWorld saves databases and not generator state, so a world replayed
        from its databases alone agrees on its contents and disagrees on its next draw."""
        from appworld import AppWorld

        self.caller_rng = random.getstate()
        self.world = AppWorld(
            task_id=body["task_id"],
            experiment_name=body["experiment"],
            random_seed=int(body["seed"]),
            # The answers are not loaded into the process that runs the agent's code: nothing here
            # holds a check, an expected value or an evaluator to call.
            load_ground_truth=False,
        )
        task = self.world.task
        return {
            "instruction": task.instruction,
            "supervisor": dict(task.supervisor),
            "datetime": task.datetime.isoformat(),
        }

    def execute(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Run one snippet of the agent's code against the world and return what it printed."""
        return {"output": self.world.execute(str(body["code"]))}

    def quiesce(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Stop everything this worker started, before the state below it is read, and say what
        is still running if anything is.

        Sealing closes the tool surface; it does not stop work an earlier `execute` left running.
        A subprocess still writing while :meth:`read` takes its filing and saves the snapshot
        leaves the two from different moments, and the evaluator then scores a state no single
        instant of the episode ever had.

        Descendants only. This process is the one being asked, and killing it here would take the
        snapshot with it, so it stays until the host closes it.

        **What this cannot do, it reports rather than implies.** A thread inside this interpreter
        is not signallable, so an episode that left one running cannot be quiesced at all, only
        observed; and a process table this cannot read is not an empty process table. Both used to
        come back as ``stopped: 0``, which is the same answer a clean episode gives. The answer
        now carries ``quiesced``, which is true only when the group was enumerated, the group is
        empty, and no thread but this one is alive. :meth:`read` proves separately that the
        pair it took is one instant, because a thread this cannot stop is exactly the thing that
        would spoil it."""
        import os
        import signal
        import subprocess
        import threading
        import time

        mine = os.getpid()
        threads = [
            thread.name
            for thread in threading.enumerate()
            if thread is not threading.main_thread() and thread.is_alive()
        ]
        try:
            group = os.getpgid(mine)
        except (OSError, AttributeError) as failure:
            return {
                "stopped": 0,
                "descendants": [],
                "threads": threads,
                "quiesced": False,
                "note": "this process is not in a readable process group: %s" % failure,
            }

        def descendants() -> Optional[List[int]]:
            """Every live process in this worker's group but this one, or ``None`` if the table
            could not be read.

            ``ps`` runs in a session of its own. It used to inherit this worker's group and so
            appear in the very listing it was producing, which meant the group was never seen
            empty: both settle windows ran to their five seconds on every submission, whether or
            not the episode had left anything behind, and a terminal that would have been prompt
            paid ten seconds for the privilege.

            Exited-but-unreaped entries are not counted. A process this call killed is in the
            table until somebody waits on it, and a zombie holds no memory, no descriptors and no
            ability to write: counting one as unquiesced would report the success of this function
            as its failure."""
            try:
                listing = subprocess.run(
                    ["ps", "-o", "pid=,pgid=,stat=", "-A"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    # Out of the group being enumerated. See above.
                    start_new_session=True,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            if listing.returncode != 0:
                # A `ps` that ran and refused is a table this could not read. Reported as such
                # rather than as the empty listing it hands back, which is the answer a quiet
                # group gives.
                return None
            found: List[int] = []
            for line in listing.stdout.splitlines():
                fields = line.split()
                if len(fields) >= 3 and fields[1].isdigit() and fields[0].isdigit():
                    pid, pgid, state = int(fields[0]), int(fields[1]), fields[2]
                    if pgid == group and pid != mine and not state.startswith("Z"):
                        found.append(pid)
            return found

        stopped = 0
        live = descendants()
        for how in (signal.SIGTERM, signal.SIGKILL):
            if not live:
                break
            for pid in live:
                try:
                    os.kill(pid, how)
                    stopped += 1
                except OSError:
                    pass
            deadline = time.monotonic() + 5
            live = descendants()
            while live and time.monotonic() < deadline:
                time.sleep(0.02)
                live = descendants()
        note = ""
        if live is None:
            note = "the process table could not be read, so the group is unknown rather than empty"
        elif live:
            note = "%d process(es) outlived SIGKILL: %s" % (len(live), sorted(live))
        elif threads:
            note = "thread(s) still running in this interpreter, which cannot be signalled: %s" % (
                sorted(threads),
            )
        return {
            "stopped": stopped,
            "descendants": sorted(live or []),
            "threads": sorted(threads),
            "quiesced": live is not None and not live and not threads,
            "note": note,
        }

    def read(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """What is in the named project, plus digests of what the world became.

        Read off the models rather than through the APIs: an API read is an agent action, and this
        is not an agent. It must see what is there, not what an authenticated session is allowed
        to see.

        The digests are what make "the same task twice is the same world" checkable, one over the
        world's own end-state databases and one over the global generator's state.

        Call :meth:`quiesce` first. What is read here has to be one instant of the episode, and it
        is not one instant if something the episode started is still writing underneath it.

        **The pair is proved rather than assumed.** The filing is read off the live models and the
        snapshot is written from those same models, so anything writing between the two makes the
        evaluator score a state no instant of the episode had. Quiescence stops processes and
        cannot stop threads, so the two are bracketed instead: the state is saved and digested,
        the filing is read, and the state is saved and digested again. Equal digests mean nothing
        touched any model across the read, which makes the filing and the snapshot the same
        instant. Unequal digests mean the pair is not one instant, and ``stable`` says so rather
        than a number quietly being wrong. Saving twice is what the world already does at the end
        of every ``execute``, over a tree that is tens of kilobytes."""
        self.world.models.reset_db_home_path()
        # Flush the end state so the grader, which is a different process, has something to read.
        # AppWorld writes the *initial* state at startup and nothing after it.
        self.world._save_state(self.world.output_db_home_path_on_disk)
        before = _directory_digest(self.world.output_db_home_path_on_disk)
        filing = _read_filing(self.world.models, body)
        self.world._save_state(self.world.output_db_home_path_on_disk)
        after = _directory_digest(self.world.output_db_home_path_on_disk)
        return {
            "filing": filing,
            "world_digest": after,
            "rng_digest": _digest(repr(random.getstate())),
            "stable": before == after,
        }

    def close(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Shut the world down and give the process's randomness back to the caller."""
        if self.world is not None:
            self.world.close()
            self.world = None
        if self.caller_rng is not None:
            random.setstate(self.caller_rng)
            self.caller_rng = None
        return {}


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
        # A label with no row to carry it is a colour nobody chose for anything.
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
    """The request handler: one method per protocol command, and a token on every one of them."""
    commands = {
        "/seed": episode.seed,
        "/open": episode.open,
        "/execute": episode.execute,
        "/quiesce": episode.quiesce,
        "/read": episode.read,
        "/close": episode.close,
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
            if self.path == "/close":
                # From a thread of its own: `shutdown` blocks until the serving loop stops, and
                # the serving loop is this thread.
                threading.Thread(target=server[0].shutdown, daemon=True).start()

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
    identifier has to be."""
    os.environ["APPWORLD_ROOT"] = body["root"]
    from appworld.evaluator import evaluate_task
    from appworld.task import Task

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
    return {"checks": checks}


def _handshake() -> Dict[str, Any]:
    """The one line of stdin a worker is started with, and the last thing it reads from it.

    On stdin rather than on ``sys.argv`` because agent-authored code runs in this process, and
    ``sys.argv`` survives for the life of it while a line of stdin does not."""
    line = sys.stdin.readline()
    sys.stdin.close()
    return json.loads(line)


def serve(root: str, token: str) -> int:
    """Bind a loopback port, say which one, and serve until told to close.

    One request at a time, and every one of them on the main thread. AppWorld runs an agent's code
    under a signal alarm, and the standard library only lets the main thread install one, so a
    handler on a worker thread fails on the first line of code an agent runs. Sequential is right
    anyway: there is one world here, and two commands into it at once is two commands into the
    same mutable state."""
    os.environ["APPWORLD_ROOT"] = root
    episode = Episode()
    holder: List[Any] = []
    server = HTTPServer(("127.0.0.1", 0), build_handler(episode, token, holder))
    holder.append(server)
    # The parent learns the port from here and from nowhere else: binding zero and reporting back
    # is what keeps two episodes started at the same moment off each other's port.
    print(json.dumps({"port": server.server_port}), flush=True)
    server.serve_forever()
    episode.close({})
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
        return serve(opening["root"], opening["token"])
    if args.command == "grade":
        print(json.dumps({"output": grade(_handshake())}), flush=True)
        return 0
    if args.command == "install":
        return install()
    return unpack(args.bundle, args.into)


if __name__ == "__main__":
    sys.exit(main())
