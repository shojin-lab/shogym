# This file is run by an interpreter baked into the worker image, not by the one shogym runs
# under, and `appworld` cannot be installed beside shogym at all (it pins `pydantic<2`). Its
# imports are expected to be unresolved in the type-check environment.
# pyright: reportMissingImports=false
"""One episode's world, in a container of its own, speaking frames on stdin and stdout.

Two reasons this is a process and not an object, and either would be enough on its own.

**AppWorld cannot share an interpreter with shogym.** It pins ``pydantic<2``; shogym's MCP layer
needs ``pydantic>=2.7``. There is no environment that satisfies both, so the world runs under an
interpreter of its own and the two talk over a pipe. That is also why this file imports nothing
from shogym: it is run by path, under a Python that has never heard of shogym, and its only
imports are the standard library and ``appworld``.

**AppWorld cannot share an interpreter with itself.** It freezes the clock with a library that
patches ``datetime`` for the whole process, every app's model classes hold their database engine
on a class attribute, and building a second world calls ``close_all()`` on the first. Two episodes
in one process are one episode the other keeps unfreezing. So an episode gets a process.

**The transport is stdin and stdout, and that is what lets the container have no network.** This
was a loopback HTTP port with a bearer token on every request. A container-loopback listener
cannot be forwarded, a published port is not loopback-only, and ``--network none`` and ``-p`` are
mutually exclusive, so a port meant the container had to have a network stack. Frames on the pipe
pair the parent created mean it does not: there is no socket to find, no port to guess, and no
process on the machine other than the parent that can write to this one's stdin.

**The token went with the port and nothing replaced it.** It authenticated a caller on an
interface any process could connect to. A pipe has one writer, held by the parent that made it, so
there is no unauthenticated caller for a token to turn away. What a token never protected against
is code running *inside* this process, and that is unchanged.

**The protocol channel is taken off the ordinary file descriptors before any world exists.**
Descriptors 0 and 1 are duplicated, then 0 is pointed at ``/dev/null`` and 1 at standard error. A
library that prints on import, or agent code that writes to ``sys.stdout``, therefore cannot
corrupt a frame, and agent code reading ``sys.stdin`` reads end-of-file rather than the parent's
next command.

**What this process must not be able to tell an agent, it does not hold.** The world is built with
``load_ground_truth=False``, so the answers and the base task's checks are not objects in the
process that runs agent-authored code, and there is no evaluator here to call. Grading the base
task happens in a second, short-lived container that never runs agent code (:func:`grade`),
reading the end state this one flushes to disk. The convention the filing is graded against is
never sent to either: there is no field in the protocol for it and no comparison in this file.

**And what this file could not do before, the container does.** The code an agent writes still
runs as this process, with this process's filesystem. That filesystem is now one task's served
tree, this episode's own output directory, and a tmpfs. The run tree, the grader's tree, the
repository, the corpus and every other episode's world are not hidden from it: they are not
mounted. See :mod:`shogym.envs.appworld.container`.

Usage::

    python worker.py serve   # {"root": ...} as a frame on stdin, then one frame per command
    python worker.py seed    # one frame: what to write, and where
    python worker.py grade   # one frame: {"root": ..., "task_id": ..., "experiment": ...}
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
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

class Episode:
    """The one world this process serves, and the randomness it was handed."""

    def __init__(self) -> None:
        self.world: Any = None
        self.caller_rng: Any = None

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
            # The answers are not loaded into the process that runs the agent's code. Grading the
            # base task reads the end state from disk in a process of its own, so nothing here
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
        """Stop everything this worker started, before the state below it is flushed.

        Sealing closes the tool surface; it does not stop work an earlier `execute` left running.
        A subprocess still writing while :meth:`seal` saves the snapshot leaves a file that is half
        of one moment and half of another, and no later check would see it.

        Descendants only. This process is the one being asked, and killing it here would take the
        flush with it, so it stays until the parent removes the container. What cannot be stopped
        is a thread inside this interpreter: threads are not signallable. That one is bounded by
        the container going away, which ends the whole namespace, so the tree the grader reads is
        settled whatever an episode left running.

        Read off ``/proc``, which exists because this runs on Linux in a container, and where the
        process table is this container's alone: the pids visible here are the ones this worker is
        responsible for and nothing else on the machine."""
        import signal
        import time

        mine = os.getpid()

        def descendants() -> List[int]:
            found: List[int] = []
            try:
                entries = os.listdir("/proc")
            except OSError:
                return found
            for entry in entries:
                if not entry.isdigit() or int(entry) == mine or int(entry) == 1:
                    continue
                found.append(int(entry))
            return found

        stopped = 0
        for how in (signal.SIGTERM, signal.SIGKILL):
            live = descendants()
            if not live:
                break
            for pid in live:
                try:
                    os.kill(pid, how)
                    stopped += 1
                except OSError:
                    pass
            deadline = time.monotonic() + 5
            while descendants() and time.monotonic() < deadline:
                time.sleep(0.02)
        return {"stopped": stopped, "left": len(descendants())}


    def seal(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Flush the end state to the episode's output tree, and read nothing off the live world.

        Call :meth:`quiesce` first: what this writes has to be one instant of the episode, and it
        is not one instant if something the episode started is still writing underneath it.

        **This used to be where the filing and the world's digest were read, and that was the
        wrong place for them.** They were observed on a live world while whatever an earlier
        ``execute`` had started was still running, and the container was removed only after the
        answer came back, so the bytes a grader opened afterwards could differ from the bytes
        those values described. What is scored now is read from this flush, in the grading
        container, after this one is gone: one immutable tree, read once, by a process that never
        ran a line the agent wrote.

        What is still read here is the state of the process's own generator, which is a fact about
        this interpreter and cannot be recovered from a file. It is a diagnostic that two servings
        of one task agreed, never an input to a score, so reading it from a world that is still
        alive costs nothing."""
        self.world.models.reset_db_home_path()
        # AppWorld writes the *initial* state at startup and nothing after it.
        self.world._save_state(self.world.output_db_home_path_on_disk)
        return {"rng_digest": _digest(repr(random.getstate()))}

    def close(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Shut the world down and give the process's randomness back to the caller."""
        if self.world is not None:
            self.world.close()
            self.world = None
        if self.caller_rng is not None:
            random.setstate(self.caller_rng)
            self.caller_rng = None
        return {}


# ----- seeding, in a container of its own -----


def seed(body: Dict[str, Any]) -> Dict[str, Any]:
    """Write a task's database log with extra rows added, through the model layer.

    The rows are created rather than hand-written as SQL. A task's database file is a replayable
    statement log with a per-row hash column and full-text shadow tables behind it, and an insert
    that got either of those wrong would be a world that looks right and diffs wrong. Creating the
    rows through the models is the same code path AppWorld uses for its own state, so what lands
    in the log is correct by construction.

    The clock is frozen at the world's own datetime first, so every row's creation timestamp is
    the world's "today" and two builds of one task produce the same bytes.

    In a container of its own, and a short-lived one. It runs no agent code, so what it is given
    is not a boundary question; what it is given is small anyway, one task's input databases and
    the staging directory the seeded log is being written into."""
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
            todoist.Section.create(project_id=project.id, name=name, order_index=order).save()
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


# ----- the transport -----


def send_frame(stream: BinaryIO, payload: Dict[str, Any]) -> None:
    """One frame: a decimal byte count, a newline, then that many bytes of JSON.

    Length-prefixed rather than newline-delimited. JSON escapes newlines inside strings, so
    line-delimited would work today and would break silently the first time a frame carried raw
    bytes; a count is not a property of the payload's contents."""
    encoded = json.dumps(payload).encode()
    stream.write(b"%d\n" % len(encoded))
    stream.write(encoded)
    stream.flush()


def read_frame(stream: BinaryIO) -> Optional[Dict[str, Any]]:
    """The next frame, or ``None`` at end of input.

    End of input is how a worker learns its parent is gone: the pipe closes, this returns nothing,
    and the serving loop stops. That is the property that makes an orphaned container impossible
    to leave running by accident."""
    header = stream.readline()
    if not header:
        return None
    length = int(header.strip())
    body = b""
    while len(body) < length:
        chunk = stream.read(length - len(body))
        if not chunk:
            return None
        body += chunk
    return json.loads(body)


def _take_channel() -> Tuple[BinaryIO, BinaryIO]:
    """Move the protocol off descriptors 0 and 1 before anything else can write to them.

    Two hazards, one move. A library that prints on import would land its greeting in the middle
    of a frame; agent code that reads ``sys.stdin`` would eat the parent's next command. So the
    two descriptors are duplicated for this module's own use, and then 0 is pointed at
    ``/dev/null`` and 1 at standard error. Ordinary printing still works and goes somewhere
    harmless.

    **What this is not.** The duplicates stay open in the interpreter that runs agent-authored
    Python, and nothing here closes them, because the worker needs them between commands and a
    pipe end that is closed cannot be reopened. Code running in this process can therefore write a
    frame the parent will read. What that is worth is bounded, and by design rather than by luck:
    the protocol carries no key, no grade, no answer and no host path in either direction, and
    since the filing, the digest and the base task's checks are all read in the grading container
    from a tree this process cannot reach once it is stopped, nothing that is scored travels over
    it at all. The most an in-process actor gains is control of its own episode's ``execute``
    output, which is its own output, and the ability to make its own episode fail. It is not a
    boundary between agent code and the worker; the boundary is the container, and it is between
    the worker and everything else.

    The duplicates are marked not inheritable, so a process agent code starts does not receive
    them. That is a real reduction rather than a claim: a child cannot forge a frame it has no
    descriptor for."""
    reader = os.fdopen(os.dup(0), "rb", buffering=0)
    writer = os.fdopen(os.dup(1), "wb", buffering=0)
    for handle in (reader, writer):
        os.set_inheritable(handle.fileno(), False)
    devnull = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(devnull, 0)
    finally:
        os.close(devnull)
    os.dup2(2, 1)
    sys.stdin = open(os.devnull, "r")
    sys.stdout = sys.stderr
    return reader, writer


def serve(root: Optional[str] = None) -> int:
    """Answer one frame at a time until the parent says close, or goes away.

    The root comes from ``APPWORLD_ROOT``, which the container sets and upstream requires anyway.
    It used to arrive on stdin so that it was not on ``sys.argv`` where agent code could read it
    back; inside the container it is the constant ``/corpus`` and naming a mount point tells
    nobody anything, so it is where upstream already looks for it.

    One command at a time, and every one of them on the main thread. AppWorld runs an agent's code
    under a signal alarm, and the standard library only lets the main thread install one, so a
    handler on a worker thread fails on the first line of code an agent runs. Sequential is right
    anyway: there is one world here, and two commands into it at once is two commands into the
    same mutable state."""
    reader, writer = _take_channel()
    root = root or os.environ.get("APPWORLD_ROOT") or os.getcwd()
    os.environ["APPWORLD_ROOT"] = root
    episode = Episode()
    commands = {
        "open": episode.open,
        "execute": episode.execute,
        "quiesce": episode.quiesce,
        "seal": episode.seal,
        "close": episode.close,
    }
    # The parent waits for this before it sends anything: a cold container importing upstream and
    # its clock-patching library is not fast, and a parent that started sending commands into an
    # interpreter that had not finished starting would have no deadline to enforce.
    send_frame(writer, {"ready": True})
    while True:
        request = read_frame(reader)
        if request is None:
            break
        # Echoed on the answer, whatever the answer is. The parent matches on it, so a reply
        # that arrives after its caller stopped waiting is a reply the next caller discards
        # rather than one it reads as its own.
        identifier = request.get("id")
        command = commands.get(str(request.get("command")))
        if command is None:
            send_frame(writer, {"id": identifier, "error": "no such command: %s" % request.get("command")})
            continue
        try:
            send_frame(writer, {"id": identifier, "output": command(dict(request.get("body") or {}))})
        except Exception as exc:  # the world's failures are answers, not crashes
            send_frame(writer, {"id": identifier, "error": "%s: %s" % (type(exc).__name__, exc)})
        if request.get("command") == "close":
            break
    return 0


def _one_shot(handler: Any) -> int:
    """Read one frame, answer it, and stop. The shape ``seed`` and ``grade`` both have."""
    reader, writer = _take_channel()
    request = read_frame(reader)
    if request is None:
        return 1
    try:
        send_frame(
            writer,
            {"id": request.get("id"), "output": handler(dict(request.get("body") or request))},
        )
    except Exception as exc:
        send_frame(writer, {"id": request.get("id"), "error": "%s: %s" % (type(exc).__name__, exc)})
        return 1
    return 0


def grade(body: Dict[str, Any]) -> Dict[str, Any]:
    """The base task's own checks, in the order the task lists them, in a container of its own.

    This is the only place ground truth is loaded, and it happens after the world is sealed, in a
    process that has never run a line the agent wrote and in a container the agent's own never
    shared a mount with. It reads the end state the serving worker flushed to disk, so nothing
    about the world has to be carried across.

    Checks are reported by position rather than by requirement text: a requirement is a paragraph
    of English naming the models and values it asserts on, so carrying one out of here would carry
    part of the task's answer with it. The position is the task's own order and is all a row
    identifier has to be."""
    os.environ["APPWORLD_ROOT"] = body["root"]
    from appworld.evaluator import evaluate_task
    from appworld.task import Task

    dbs = os.path.join(body["experiment"], "tasks", body["task_id"], "dbs")
    # Before the evaluator, and in this order for a mechanical reason rather than a preference:
    # AppWorld holds each app's database engine on a class attribute, so a collection loaded after
    # the evaluator has built its own reads the evaluator's world instead of the one it asked for.
    # Both read the same tree either way, because the container that could have changed it is
    # gone; what the order buys is that this one reads what it named.
    filing = _read_filing(_models_on_disk(dbs, body["task_id"]), body)
    world_digest = _directory_digest(dbs)
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
        # Read here rather than off the live world, and from the same tree the evaluator above
        # just read. The container that could have changed these bytes was removed before this
        # process started, so what is scored and what is graded are one state by construction
        # rather than by two observations happening to agree.
        "filing": filing,
        "world_digest": world_digest,
    }


def _models_on_disk(dbs: str, tag: str) -> Any:
    """Load one task's flushed databases, read-only, into an interpreter of their own.

    Through upstream's own model layer rather than by parsing the statement log: the log is a
    replayable format with a per-row hash column and full-text shadow tables behind it, and a
    reader that got either wrong would be reading a world that is not the one that was saved."""
    from appworld.collections.models import ModelCollection

    return ModelCollection.load(
        to_db_home_path=f":memory:shogym-read-{tag}",
        from_db_home_path=dbs,
        load_apps=["todoist"],
    )


def install() -> int:
    """Unpack the app sources the wheel ships packed. Run once, when the image is built."""
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
    parser = argparse.ArgumentParser(description="an AppWorld world, behind a pipe")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="one world, on APPWORLD_ROOT; one frame per command")
    commands.add_parser("seed", help="write one task's seeded database log; one frame in")
    commands.add_parser("grade", help="one task's checks; one frame in")
    commands.add_parser("install")
    unpacking = commands.add_parser("unpack")
    unpacking.add_argument("--bundle", required=True)
    unpacking.add_argument("--into", required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve()
    if args.command == "seed":
        return _one_shot(seed)
    if args.command == "grade":
        return _one_shot(grade)
    if args.command == "install":
        return install()
    return unpack(args.bundle, args.into)


if __name__ == "__main__":
    sys.exit(main())
