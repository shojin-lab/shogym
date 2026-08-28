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
import secrets
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

from shogym.envs._upstream import _locked
from shogym.envs.appworld.ledger import ROLES, SECTIONS, Backlog, Request


#: The parts of a corpus that no task changes and every world reads: an allowlist, not "everything
#: except ``tasks``".
#:
#: **Named because the mount set is built from this.** A derived root used to hold whatever the
#: source corpus had at its top level, and the served container mounted all of it, so an entry
#: nobody here had heard of was inside the boundary by default. The pinned bundle already ships
#: two (`LICENSE`, `README_BEFORE_SHARING.md`), and ``APPWORLD_ROOT`` takes any directory with a
#: ``data/tasks`` in it, so what a custom corpus happened to carry was agent-readable because
#: nothing had said it should not be.
#:
#: Anything else is left where it is rather than refused. A corpus is somebody else's directory
#: and a file this port does not use is not a defect in it; what would be a defect is deriving it,
#: mounting it, and calling the result an exhaustive list.
SHARED_ENTRIES: Tuple[str, ...] = ("api_docs", "base_dbs", "datasets", "version.txt")

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
    verify: Optional[Callable[[str], None]] = None,
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
    process that owns it. A loser finds the target already there and uses it.

    ``verify`` is handed the name of the unit about to be read and raises if the corpus no longer
    holds what the caller was built against. It is called under the lock, after the decision to
    build and before a byte is copied, so a warm task pays nothing for it. A task is derived on its
    first use, which may be hours and two hundred episodes after the env stated what corpus it
    serves; without this, an in-place edit in that window would put changed databases and a changed
    ground truth into a world and its grading baseline under a run identity that had never read
    them (see :meth:`~adapter.CorpusSnapshot.verify` for what it does and does not prove)."""
    target = derived / "tasks" / task_id
    if already_derived(derived=derived, graded=graded, task_id=task_id):
        return target
    (derived / "tasks").mkdir(parents=True, exist_ok=True)
    (graded / "tasks").mkdir(parents=True, exist_ok=True)
    # Advisory rather than load-bearing: the staging directory and the rename that publishes it
    # are what make a concurrent build safe, and a filesystem with no locks still gets a task that
    # is whole or absent. What the lock buys is that two cold builders usually do the work once.
    with _locked(derived / "tasks"):
        if already_derived(derived=derived, graded=graded, task_id=task_id):
            return target
        if verify is not None:
            verify(f"tasks/{task_id}")
        source = original / "tasks" / task_id
        building = _staging(derived / "tasks", task_id)
        (building / "dbs").mkdir(parents=True)
        for entry in sorted(source.iterdir()):
            # `ground_truth` is deliberately absent: see the docstring.
            if entry.name not in ("dbs", "ground_truth"):
                _materialise(entry, building / entry.name)
        for entry in sorted((source / "dbs").iterdir()):
            if entry.name != "todoist.jsonl":
                _materialise(entry, building / "dbs" / entry.name)
        write_log(source / "dbs", building / "dbs" / "todoist.jsonl")
        # Last into the staging tree, so it reaches the task's real name in the same rename that
        # publishes it: there is no ordering in which a reader sees the marker over a half-built
        # task.
        _mark_complete(building)
        # `replacing`, because what got past the check above is either nothing at all or a task
        # some process was interrupted in the middle of. A publish that dropped its build on
        # finding the name taken would leave the broken tree exactly where it was and rebuild it
        # again on the next episode, forever.
        _publish(building, target, replacing=True)
        _grading_view(target, graded / "tasks" / task_id, source / "ground_truth")
    return target


def already_derived(*, derived: Path, graded: Path, task_id: str) -> bool:
    """Whether both views of a task are on disk and were finished rather than merely started.

    Both, not either: the world an agent drives and the grader's view of it are built together and
    a run that found only the first would serve an episode nothing could grade.

    What is asked is the completion marker each tree carries, which reaches its name in the same
    rename that publishes the tree (see :func:`_mark_complete`). It says the derivation finished;
    it does not say the bytes are still the ones that were written, which on a development host is
    the operator's own filesystem to answer for."""
    return _complete(derived / "tasks" / task_id) and _complete(graded / "tasks" / task_id)


def _grading_view(target: Path, view: Path, answers: Path) -> None:
    """The grader's view of a derived task: the same world, with the answers linked back in.

    Independent copies of the served task's own files. They were hard links, one instance on disk
    under two names, which made this view move whenever a served episode wrote through its own: the
    baseline the grader diffs against was editable by the thing being graded. The answers are
    copied from the corpus, and this tree is the only place in the port that names them."""
    building = _staging(view.parent, view.name)
    (building / "dbs").mkdir(parents=True)
    for entry in sorted(target.iterdir()):
        # The served task's own completion marker is not this tree's: this one holds the answers
        # as well, so it is a different set of files and gets a marker of its own below.
        if entry.name not in ("dbs", _COMPLETE):
            _materialise(entry, building / entry.name)
    for entry in sorted((target / "dbs").iterdir()):
        _materialise(entry, building / "dbs" / entry.name)
    _materialise(answers, building / "ground_truth")
    _mark_complete(building)
    _publish(building, view, replacing=True)


def derive_view(*, derived: Path, view: Path, task_id: str) -> Path:
    """One episode's own served corpus, and return the root to hand the worker.

    **Why an episode cannot share the served tree.** The derived corpus is one deterministic
    global root and every worker was handed it as ``APPWORLD_ROOT``. The files in it are writable
    by the process that runs agent-authored code, and nothing put them back, so a write through
    episode A's served view was still there in episode B's starting inputs. Two arms of one pair
    are the same task served at the same time, so the arm that was supposed to differ only in what
    it was told could also differ in the world it was given, and the difference would be one the
    treatment did not make.

    **The task is copied; everything else is a symlink.** A task's own world is 28 KB, so a copy
    per episode is nothing, and it is the only part an episode has state in. The rest is 129 MB of
    databases and documentation that would make a per-episode copy absurd, so those are named
    rather than copied.

    Symlinks here and not in :func:`derive_task`, and the difference is what they point at. A
    symlink into the *corpus* names a directory whose task folders have ``ground_truth`` siblings,
    which is a direction to the answers. These name the derived tree, which has no answers in it
    by construction. That is the property this function is for and the one the port's isolation
    claims rest on: **no path an episode can resolve through its served root reaches the answers,
    another task's databases, or the grader.**

    **What it does not give, on this host worker.** The shared base and the shared task cache are
    named rather than copied, and they are writable by the user the worker runs as, so an episode
    that goes looking can change what a later episode, or the other arm of its own pair, starts
    from. Nothing here prevents that: upstream never writes there, so it does not happen by
    accident, and preventing it deliberately is what the container does, which mounts the shared
    base read-only. The host worker is for development, and this is the line where that
    matters."""
    data = view / "data"
    data.mkdir(parents=True, exist_ok=True)
    for entry in sorted(derived.iterdir()):
        if entry.name == "tasks":
            continue
        named = data / entry.name
        if not named.exists():
            named.symlink_to(entry)
    tasks = data / "tasks"
    tasks.mkdir(exist_ok=True)
    served = tasks / task_id
    # Rebuilt rather than reused. This is per episode, and an episode that found the previous
    # episode's leftovers here would be exactly the thing this function exists to prevent.
    shutil.rmtree(served, ignore_errors=True)
    shutil.copytree(derived / "tasks" / task_id, served)
    return view


def derive_root(
    *, original: Path, derived: Path, verify: Optional[Callable[[str], None]] = None
) -> Path:
    """Materialise the parts of a corpus that no task changes, and return the derived root.

    Copies rather than symlinks, for the reason :func:`derive_task` gives: a symlink to the
    corpus's own ``datasets`` directory is a path to the corpus, and the corpus holds every task's
    answers. The base databases are 129 MB and the API documentation 4.5 MB, so this is built once
    per corpus and every episode's view names it rather than copying it.

    **Staged and published, never built in place.** Each entry is built under a staging name of
    this process's own, marked complete there, and given its final name by one rename; a loser
    finds the target already published and drops what it built. That is the protocol
    :func:`derive_task` already uses, and it is correct without a lock rather than because of one.

    ``verify`` is handed each entry's name before that entry is copied, and raises if the corpus no
    longer holds what the caller was built against. Only outstanding entries are checked, so the
    ordinary warm construction pays nothing: it copies nothing, so there is nothing to be wrong
    about. On the cold path this is 134 MB of shared base databases about to become every episode's
    starting state, and reading them without a check would build that state out of whatever the
    corpus held at the moment of the copy rather than out of what the run says it is serving."""
    derived.mkdir(parents=True, exist_ok=True)
    # Required, for the reason :func:`derive_task`'s is: the body below opens this directory for
    # writing and seals it again, and what a second process without exclusion would find is not a
    # stale tree but an open one. A window that another process can close is not a window.
    with _locked(derived, required=True):
        # What is missing is decided before the directory is opened, so a construction that has
        # nothing to build never opens it at all. That matters because the ordinary case is warm:
        # an env is constructed while another episode of the pair is already running, and opening
        # the parent on every construction would put a writable window beside every live worker
        # rather than only beside a cold build.
        #
        # A target that exists but is not both complete and sealed was left by a crash or by a
        # chmod that failed part way through. It is not repaired in place: a fresh tree is staged
        # beside it and published over it, so nothing ever reads a directory while it is being
        # made correct.
        # Named rather than enumerated, and only the ones this corpus has: a corpus missing one
        # of them fails where the world tries to open it, with upstream's own words, which is
        # where it failed before this list existed too. What the list changes is the other
        # direction, which is that an entry nobody named is not derived and so is never mounted.
        outstanding = [
            original / name
            for name in SHARED_ENTRIES
            if (original / name).exists()
            and not (_complete(derived / name) and _sealed(derived / name))
        ]
        (derived / "tasks").mkdir(exist_ok=True)
        for entry in outstanding:
            if verify is not None:
                verify(entry.name)
            target = derived / entry.name
            building = _staging(derived, entry.name)
            _materialise(entry, building)
            _mark_complete(building)
            _publish(building, target, replacing=True)
    return derived


def _staging(parent: Path, name: str) -> Path:
    """A directory to build ``name`` in, under a name no other process can be using.

    The pid alone was the old answer and it is not unique enough: pids are recycled, and a
    crashed builder's leftovers under the same number would be deleted out from under a live one.
    Eight random bytes beside it make the name this call's."""
    parent.mkdir(parents=True, exist_ok=True)
    building = parent / f".{name}.{os.getpid()}.{secrets.token_hex(8)}.building"
    shutil.rmtree(building, ignore_errors=True)
    return building


#: Written into a materialised directory once every file under it is there. A directory without it
#: is a directory some process was interrupted in the middle of.
_COMPLETE = ".shogym-complete"


def _complete(target: Path) -> bool:
    """Whether ``target`` was finished, rather than merely started."""
    if not target.exists():
        return False
    if target.is_dir():
        return (target / _COMPLETE).exists()
    return True


def _mark_complete(target: Path) -> None:
    if target.is_dir():
        (target / _COMPLETE).write_text("")


def _materialise(source: Path, target: Path) -> None:
    """Put ``source``'s content at ``target`` without leaving a path back to where it came from.

    An independent copy. Never a symlink, because a symlink names its target's directory and in
    this corpus every task directory has the answers as a sibling; and never a hard link either,
    because a hard link is the same file under a second name. The worker runs as the same user
    that built this and the file is writable, so a write through the served pathname would change
    the corpus it was copied from and the baseline the grader diffs against, which is a served
    episode editing the thing it is scored on.

    The corpus is 134 MB and this is paid once per derived root, not per episode."""
    if target.exists():
        return
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for entry in sorted(source.iterdir()):
            _materialise(entry, target / entry.name)
        return
    shutil.copy2(source, target)


def _publish(building: Path, target: Path, *, replacing: bool = False) -> None:
    """Give a staging tree its final name, or drop it if someone else got there first.

    ``replacing`` is for the callers that have to displace an existing target: a served task, a
    grading view, or a shared base entry left incomplete by a crash. It is still not a build in
    place. The finished tree is renamed *aside* and the staged one renamed in, and
    the displaced one is removed afterwards.

    **Two renames are two operations, so the name is briefly absent even when this succeeds.**
    Between them the target does not exist, and the builders' lock excludes other builders rather
    than the live workers resolving paths through this tree. The window is a syscall wide and it
    is real. What this does promise is that the name never holds a half-made tree, because what is
    renamed in was complete before it had this name.

    **A publish that fails puts the incumbent back.** For a shared base entry a name left absent
    is worse than a failed build, because an episode already running resolves absolute names
    through it. So the incumbent is restored, and if the restore itself fails the displaced copy
    is *retained* under its own name rather than removed, because it is then the only copy there
    is."""
    if target.exists() and not replacing:
        shutil.rmtree(building, ignore_errors=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    displaced = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.displaced")
    published = False
    # Whether the incumbent is sitting under `displaced` and is the only copy of it there is.
    aside = False
    try:
        if target.exists():
            os.replace(target, displaced)
            aside = True
        os.replace(building, target)
        published = True
    except OSError:
        shutil.rmtree(building, ignore_errors=True)
        if aside:
            try:
                os.replace(displaced, target)
                aside = False
            except OSError:
                # Left where it is. The `finally` below removes a displaced tree only after a
                # publish that worked, so this one survives this call and can be found by name.
                pass
            # The publish did not happen, whatever the restore did. Raised rather than swallowed,
            # because what holds the name now is the tree this call was asked to replace: for a
            # task or a view that is the entry `already_derived` had already refused, and a caller
            # told nothing would go on to serve an episode out of it.
            raise
        if not target.exists():
            raise
        # Somebody else published while this build was staging. Their tree is under the name and
        # this one has been dropped.
    finally:
        if published and aside:
            shutil.rmtree(displaced, ignore_errors=True)


__all__ = [
    "SHARED_ENTRIES",
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
    "derive_task",
    "request_description",
    "seeding",
]
