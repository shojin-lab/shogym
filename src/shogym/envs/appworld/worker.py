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

**The port is bound to loopback and gated by a token minted at spawn.** AppWorld's own environment
server publishes ``evaluate``, ``save_state`` and ``load_state`` with no authentication, on every
interface. A world where the ground truth is one unauthenticated request away is a world an agent
with a shell can grade itself against, and "the agent probably will not port-scan" is not a
property anybody can check. This one answers nothing without the token, and the token is held by
the serving process and never reaches an agent: not in the instructions, not in a tool schema, not
in a tool result.

**Nothing about the treatment lives here.** This process is a generic handle on an AppWorld world:
it writes the rows it is told to write, runs the code it is handed, and reports what the world
holds. The convention the filing is graded against is never sent to it, there is no field in the
protocol for one, and there is no comparison in this file. A world an agent had complete control
of still could not be made to say what the answer was.

Usage::

    python worker.py serve --root <appworld root> --token <token>
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
            load_ground_truth=True,
        )
        task = self.world.task
        return {
            "instruction": task.instruction,
            "supervisor": dict(task.supervisor),
            "datetime": task.datetime.isoformat(),
            "checks": [entry["requirement"] for entry in _test_data(task)],
        }

    def execute(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Run one snippet of the agent's code against the world and return what it printed."""
        return {"output": self.world.execute(str(body["code"]))}

    def read(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """What is in the named project, plus digests of what the world became.

        Read off the models rather than through the APIs: an API read is an agent action, and this
        is not an agent. It must see what is there, not what an authenticated session is allowed
        to see.

        The digests are what make "the same task twice is the same world" checkable, one over the
        world's own end-state databases and one over the global generator's state."""
        self.world.models.reset_db_home_path()
        return {
            "filing": _read_filing(self.world.models, body),
            "world_digest": _directory_digest(self.world.output_db_home_path_on_disk),
            "rng_digest": _digest(repr(random.getstate())),
        }

    def evaluate(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """The base task's own checks, in the order the task lists them.

        Reported by position rather than by requirement text: a requirement is a paragraph of
        English naming the models and values it asserts on, so carrying one out of here would
        carry part of the task's answer with it. The position is the task's own order and is all a
        row identifier has to be."""
        with _ignoring(list(body.get("ignore") or ())):
            tracker = self.world.evaluate(suppress_errors=True)
        reported = tracker.to_dict(stats_only=False)
        passed = {entry["requirement"] for entry in reported["passes"]}
        failed = {entry["requirement"] for entry in reported["failures"]}
        checks = []
        for position, entry in enumerate(_test_data(self.world.task), start=1):
            requirement = entry["requirement"]
            verdict = True if requirement in passed else (False if requirement in failed else None)
            checks.append([f"aw.{position:03d}", verdict])
        return {"checks": checks}

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
        "/read": episode.read,
        "/evaluate": episode.evaluate,
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
    serving = commands.add_parser("serve")
    serving.add_argument("--root", required=True, help="the directory whose data/ is the corpus")
    serving.add_argument("--token", required=True, help="the secret every request must carry")
    commands.add_parser("install")
    unpacking = commands.add_parser("unpack")
    unpacking.add_argument("--bundle", required=True)
    unpacking.add_argument("--into", required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve(args.root, args.token)
    if args.command == "install":
        return install()
    return unpack(args.bundle, args.into)


if __name__ == "__main__":
    sys.exit(main())
