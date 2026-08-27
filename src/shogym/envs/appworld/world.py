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
    port's negative control and never its headline."""

    check_id: str
    options: Tuple[str, ...]


#: Every option set has the API's own default struck out of it, so an unset field reads as
#: unexercised rather than as a choice the agent made.
SLOTS: Tuple[Slot, ...] = (
    Slot("fr.log.section", SECTIONS),
    Slot("fr.label.color", ("red", "blue", "green", "orange", "yellow")),
    Slot("fr.log.unit", ("minutes", "hours", "days")),
    Slot("fr.log.priority", ("high", "low")),
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


EMPTY_FILING = Filing(
    filed=False, rows=0, lines=(), section=None, color=None, unit=None, priority=None
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
    task_id: str,
    write_log: Callable[[Path, Path], None],
) -> Path:
    """Materialise one task's directory under ``derived``, with its filing log written into it.

    Everything the task ships is linked rather than copied, so a derived corpus of the whole split
    costs one small file per task. Only the todoist log is a real file, because it is the only one
    this port changes: ``write_log`` is handed the task's own database directory and the path the
    rewritten log belongs at.

    Built beside the target and moved into place in one rename, so a task is either fully derived
    or not derived at all: a half-written log picked up by a later run would be a world nobody
    could reproduce. Idempotent, which is what makes the same task served twice the same world."""
    target = derived / "tasks" / task_id
    if (target / "dbs" / "todoist.jsonl").exists():
        return target
    source = original / "tasks" / task_id
    building = derived / "tasks" / f".{task_id}.building"
    if building.exists():
        shutil.rmtree(building)
    (building / "dbs").mkdir(parents=True)
    for entry in sorted(source.iterdir()):
        if entry.name != "dbs":
            _link(entry, building / entry.name)
    for entry in sorted((source / "dbs").iterdir()):
        if entry.name != "todoist.jsonl":
            _link(entry, building / "dbs" / entry.name)
    write_log(source / "dbs", building / "dbs" / "todoist.jsonl")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(building, target)
    return target


def derive_root(*, original: Path, derived: Path) -> Path:
    """Materialise the parts of a corpus that no task changes, and return the derived root.

    The base databases, the API documentation and the split files are linked in whole: they are
    read-only inputs and linking them means one derived corpus rather than a copy per run."""
    (derived / "tasks").mkdir(parents=True, exist_ok=True)
    for entry in sorted(original.iterdir()):
        if entry.name == "tasks":
            continue
        _link(entry, derived / entry.name)
    return derived


def _link(source: Path, target: Path) -> None:
    """Point ``target`` at ``source``, replacing whatever was there."""
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
    "derive_root",
    "derive_task",
    "request_description",
    "seeding",
]
