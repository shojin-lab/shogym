"""The treatment: the paragraph appended to every instruction, and the backlog it points at.

Two halves of one thing. The paragraph asks for a filing log and names none of the four choices
it needs; the seeded rows are what those choices are computed from. Neither works without the
other, so they live together and are drawn from one key.

**The rows go into the task's *input* databases, never into a running world.** AppWorld scores a
task by diffing the end state against the databases the task shipped with, so rows written after
the world starts appear in every scenario's changed set and break assertions the port has no
business touching. Writing them into a derived copy of the task's input directory instead means
the runtime world and the evaluator's baseline see the same backlog, and the scenario's own score
is untouched.

This module decides *what* the rows say and owns nothing that writes them. The one file that
changes is written by the worker, through upstream's own model layer, because a task's database
file is a replayable statement log with a per-row hash column and full-text shadow tables behind
it and a hand-written insert that got either wrong would be a world that looks right and diffs
wrong. Everything else in the derived copy is a link.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

from shogym.envs._upstream import _locked
from shogym.envs.appworld.ledger import ROLES, SECTIONS, Backlog, Request

# ----- the appended paragraph -----

#: Appended to every task instruction, byte-identical on every task. It points at world data and
#: never states a rule; its two exclusions are symmetric, so neither choice is the salient one;
#: and it has no owner, because two tasks are two different households and an owner would argue
#: against the thing being measured. "The instruction above" names the task; "request" is kept for
#: the backlog entries, so the two are never confused.
APPENDED_PARAGRAPH = (
    "Separately from the instruction above, update the filing log before you finish. Open the "
    'Todoist project named "Task Log". The tasks already in it are requests waiting to be '
    "filed: each one records the dates held for it, and the project's own description gives the "
    "days of the week that are worked and the dates that count as closed. In that project "
    'create one task titled "Filing", place it in one of the project\'s sections, and give it a '
    "priority, a duration and a duration unit; then create a Todoist label named \"task-log\", "
    "give it a colour, and add that label to the task you created. In the task's description "
    "put one line for each waiting request, in the order the requests appear, holding the "
    "request's reference, a comma, and its band, and nothing else. Each waiting request has an "
    "ageing window that begins the day after the date the request is filed from and ends today. "
    "A request's band follows the number of days counted in its window: 0 to 5 Routine, 5 to 10 "
    "Standard, 10 to 20 Priority, 20 to 30 Urgent, 30 and over Critical. The instruction above "
    "does not tell you which of the recorded dates a request is filed from, which days in a "
    "window are counted, which band a request takes when that number falls on one of the "
    "figures printed above, what to do with a request that has no dates recorded, or which "
    "section, priority, duration unit or label colour to use. Choose them."
)

#: The title the agent's own row must carry, and the label it must attach. Both are named in the
#: paragraph, so neither is a choice.
LOG_TITLE = "Filing"
LOG_LABEL = "task-log"
PROJECT_NAME = "Task Log"

#: The models the appended chore's own work adds to a world. They go into every task's ignore list
#: when the base task's checks are collected, so a scenario's "nothing else changed" assertion is
#: not failed by the filing this port asks for on top of it. The seeded backlog needs no entry
#: here: it is written into the task's input databases and is therefore part of the baseline.
ADDED_MODELS: Tuple[str, ...] = ("todoist.Task", "todoist.Label", "todoist.TaskLabelLink")


# ----- the four stored slots that ride along beside the ledger -----


class Slot(NamedTuple):
    """One stored choice the paragraph asks for and does not name.

    Unlike a ledger line, a slot's verdict reads the slot alone: a failure crosses off exactly
    one option and says nothing about any other row. That caps what a grade on it can be worth at
    one over the option count however good the reader is, which is why the four of them are the
    port's negative control and never its headline.

    ``default`` is what the API leaves behind when the agent passes nothing. It is struck from
    ``options`` and read back as unexercised, because a field the agent never touched is not a
    choice it made: scoring the default as a wrong guess would count an omission as an attempt
    and would put the slot's own filing rate inside its compliance rate."""

    check_id: str
    options: Tuple[str, ...]
    default: Optional[str]


SLOTS: Tuple[Slot, ...] = (
    Slot("fr.log.section", SECTIONS, None),
    Slot("fr.label.color", ("red", "blue", "green", "orange", "yellow"), "charcoal"),
    Slot("fr.log.unit", ("minutes", "hours", "days"), None),
    Slot("fr.log.priority", ("high", "low"), "medium"),
)


class Filing(NamedTuple):
    """What the agent left behind, read off the end state and nothing else.

    ``lines`` is the description of the earliest ``Filing`` task, split into lines; the rest are
    the four stored slots, each ``None`` where the agent never set it. ``rows`` counts how many
    ``Filing`` tasks were found, because hedging by filing several must be visible rather than
    scoreable."""

    filed: bool
    rows: int
    lines: Tuple[str, ...]
    section: Optional[str]
    color: Optional[str]
    unit: Optional[str]
    priority: Optional[str]
    #: The number the paragraph asks for beside the unit. Not one of the four scored slots: it is
    #: a value rather than a choice from an option set, so no draw can be right about it and a
    #: verdict on it would carry nothing. Reported, so that leaving it out is visible rather than
    #: invisible to every metric.
    duration: Optional[float]


EMPTY_FILING = Filing(
    filed=False,
    rows=0,
    lines=(),
    section=None,
    color=None,
    unit=None,
    priority=None,
    duration=None,
)


# ----- seeded rows -----

#: What every seeded request's priority is set to. The API's own default, so a seeded row reads as
#: a row nobody made a choice about, and the agent's own row is the only one that did.
SEEDED_PRIORITY = "medium"


def request_description(request: Request) -> str:
    """The dates a request records, one role per line, in the order the world prints them.

    A request with nothing recorded gets the empty description the API itself defaults to, so "no
    dates recorded" is a state the world already has a representation for rather than a sentinel
    this port invented."""
    if request.dates is None:
        return ""
    return "\n".join(f"{role}: {request.dates[role].isoformat()}" for role in ROLES)


def seeding(backlog: Backlog, *, supervisor_email: str, moment: str, tag: str) -> Dict[str, Any]:
    """Everything the worker needs to write one task's seeded rows, and nothing else.

    A plain dictionary because it crosses a process boundary: the worker runs under an interpreter
    that has never heard of shogym, so what it receives has to be JSON and has to say what to
    write rather than name anything about why."""
    return {
        "tag": tag,
        "datetime": moment,
        "supervisor_email": supervisor_email,
        "priority": SEEDED_PRIORITY,
        "project": {"name": PROJECT_NAME, "description": backlog.description},
        "sections": list(SECTIONS),
        "requests": [
            {"title": request.reference, "description": request_description(request)}
            for request in backlog.requests
        ],
    }


# ----- the derived corpus -----


def derive_task(
    *,
    original: Path,
    derived: Path,
    graded: Path,
    task_id: str,
    write_log: Callable[[Path, Path], None],
) -> Path:
    """Materialise one task's world twice: once for the agent, once for the grader.

    ``derived`` is the corpus the episode's world runs against. It carries no ``ground_truth``
    directory, and **nothing in it is a symlink**: every file is a hard link to the corpus's own
    (or a copy, across filesystems). That second part is the load-bearing one. A symlink names its
    target, and a target names a directory, and that directory has a ``ground_truth`` sibling, so a
    tree of symlinks into the source corpus is a tree of directions to the answers. A hard link is
    a file with no other discoverable path.

    ``graded`` is the same task with the answers linked back in, and only the grading process is
    ever given that root.

    **Two streams starting the same cold task is the ordinary case, not a hypothetical.** Paired
    forks are launched together and both derive on first use, so the work is done under a lock on
    the tasks directory and published with one rename, into a staging directory named for the
    process that owns it. A loser finds the target already there and uses it."""
    target = derived / "tasks" / task_id
    if already_derived(derived=derived, graded=graded, task_id=task_id):
        return target
    (derived / "tasks").mkdir(parents=True, exist_ok=True)
    (graded / "tasks").mkdir(parents=True, exist_ok=True)
    with _locked(derived / "tasks"):
        if already_derived(derived=derived, graded=graded, task_id=task_id):
            return target
        source = original / "tasks" / task_id
        building = derived / "tasks" / f".{task_id}.{os.getpid()}.building"
        shutil.rmtree(building, ignore_errors=True)
        (building / "dbs").mkdir(parents=True)
        for entry in sorted(source.iterdir()):
            # `ground_truth` is deliberately absent: see the docstring.
            if entry.name not in ("dbs", "ground_truth"):
                _materialise(entry, building / entry.name)
        for entry in sorted((source / "dbs").iterdir()):
            if entry.name != "todoist.jsonl":
                _materialise(entry, building / "dbs" / entry.name)
        write_log(source / "dbs", building / "dbs" / "todoist.jsonl")
        _publish(building, target)
        _grading_view(target, graded / "tasks" / task_id, source / "ground_truth")
    return target


def already_derived(*, derived: Path, graded: Path, task_id: str) -> bool:
    """Whether both views of a task are already on disk and complete.

    Both, not either: the world an agent drives and the grader's view of it are built together and
    a run that found only the first would serve an episode nothing could grade."""
    return (derived / "tasks" / task_id / "dbs" / "todoist.jsonl").exists() and (
        graded / "tasks" / task_id / "ground_truth"
    ).exists()


def _grading_view(target: Path, view: Path, answers: Path) -> None:
    """The grader's view of a derived task: the same world, with the answers linked back in.

    Hard links to the served task's own files rather than copies, so the seeded database log has
    one instance on disk and the two views cannot drift apart. The answers are linked from the
    corpus, and this tree is the only place in the port that names them."""
    building = view.parent / f".{view.name}.{os.getpid()}.building"
    shutil.rmtree(building, ignore_errors=True)
    (building / "dbs").mkdir(parents=True)
    for entry in sorted(target.iterdir()):
        if entry.name != "dbs":
            _materialise(entry, building / entry.name)
    for entry in sorted((target / "dbs").iterdir()):
        _materialise(entry, building / "dbs" / entry.name)
    _materialise(answers, building / "ground_truth")
    _publish(building, view)


def derive_root(*, original: Path, derived: Path) -> Path:
    """Materialise the parts of a corpus that no task changes, and return the derived root.

    Hard links rather than symlinks, for the reason :func:`derive_task` gives: a symlink to the
    corpus's own ``datasets`` directory is a path to the corpus, and the corpus holds every task's
    answers. The base databases are 129 MB and the API documentation 4.5 MB, so linking rather
    than copying is also what makes this free."""
    (derived / "tasks").mkdir(parents=True, exist_ok=True)
    with _locked(derived):
        for entry in sorted(original.iterdir()):
            if entry.name == "tasks":
                continue
            if not (derived / entry.name).exists():
                _materialise(entry, derived / entry.name)
    return derived


def share_outputs(*, derived: Path, graded: Path) -> None:
    """Give the grader a view of the episode's own output tree.

    An episode writes its end state under the root its world was served from, and the grader reads
    it under the root it was given. Those are two roots on purpose. This publishes the second name
    for the first tree under the same lock the roots are built under, because every environment
    constructed against a cold cache runs this and two of them racing on one ``symlink`` is an
    ``FileExistsError`` out of a constructor.

    A symlink here rather than a hard link, because a directory cannot be hard-linked and this
    one has to stay the same directory as it fills up. It points from the grader's private tree
    into the served tree, which is a direction an agent's process has no way to follow: it is the
    private tree that is hard to find, and this link lives in it."""
    outputs = derived / "experiments"
    (outputs / "outputs").mkdir(parents=True, exist_ok=True)
    graded.mkdir(parents=True, exist_ok=True)
    with _locked(graded):
        link = graded / "experiments"
        if link.is_symlink() and os.readlink(link) == str(outputs):
            return
        _link(outputs, link)


def _materialise(source: Path, target: Path) -> None:
    """Put ``source``'s content at ``target`` without leaving a path back to where it came from.

    A hard link where the filesystem allows one, a copy otherwise. Never a symlink: a symlink
    names its target's directory, and in this corpus every task directory has the answers as a
    sibling."""
    if target.exists():
        return
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for entry in sorted(source.iterdir()):
            _materialise(entry, target / entry.name)
        return
    try:
        os.link(source, target)
    except OSError:
        # A different filesystem, or one with no hard links. A copy costs space and says nothing
        # about where it came from, which is the property that matters.
        shutil.copy2(source, target)


def _publish(building: Path, target: Path) -> None:
    """Move a staging tree into place, or drop it if someone else got there first."""
    if target.exists():
        shutil.rmtree(building, ignore_errors=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(building, target)
    except OSError:
        shutil.rmtree(building, ignore_errors=True)
        if not target.exists():
            raise


def _link(source: Path, target: Path) -> None:
    """Point ``target`` at ``source``, replacing whatever was there."""
    if target.is_symlink() and os.readlink(target) == str(source):
        return
    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.symlink_to(source)


__all__ = [
    "ADDED_MODELS",
    "APPENDED_PARAGRAPH",
    "EMPTY_FILING",
    "LOG_LABEL",
    "LOG_TITLE",
    "PROJECT_NAME",
    "SEEDED_PRIORITY",
    "SLOTS",
    "Filing",
    "Slot",
    "already_derived",
    "derive_root",
    "share_outputs",
    "derive_task",
    "request_description",
    "seeding",
]
