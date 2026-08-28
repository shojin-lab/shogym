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

import hashlib
import json
import os
import secrets
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

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
    serves; without this, an in-place edit in that window put changed databases and a changed
    ground truth into a world and its grading baseline under a run identity that had never read
    them (see :meth:`~adapter.CorpusSnapshot.verify` for what it does and does not prove)."""
    target = derived / "tasks" / task_id
    if already_derived(derived=derived, graded=graded, task_id=task_id):
        return target
    (derived / "tasks").mkdir(parents=True, exist_ok=True)
    (graded / "tasks").mkdir(parents=True, exist_ok=True)
    # The lock is **required** here, not advisory. What follows it opens two published directories
    # for writing and seals them again, and a permission window is only correct while nothing else
    # is inside it: a second builder that arrives without exclusion does not find a stale tree, it
    # finds an open one, and it seals it shut again under the first builder's feet. Staging and
    # renaming makes the *contents* safe without a lock and can do nothing for the modes.
    with (
        _locked(derived / "tasks", required=True),
        _opened(derived / "tasks"),
        _opened(graded / "tasks"),
    ):
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
        # Sealed before it is published, so what appears under the name is read-only from the
        # instant it exists. This is a *shared* task: every episode of it copies its own view out
        # of here, so a served worker able to write to it could change what a later episode, or
        # the other arm of its own pair, starts from. See `derive_view` for the invariant.
        _seal(building)
        # After the seal, because the modes are part of what a warm episode checks; and the last
        # thing written into the staging tree, so it reaches the task's real name in the same
        # rename that publishes the tree. See `_write_manifest`.
        _write_manifest(building)
        # `replacing`, because what got past the check above is either nothing at all or a task
        # that is on disk and is not the task it claims to be. A publish that dropped its build
        # on finding the name taken would leave the broken tree exactly where it was and rebuild
        # it again on the next episode, forever.
        _publish(building, target, replacing=True)
        _grading_view(target, graded / "tasks" / task_id, source / "ground_truth")
    return target


def already_derived(*, derived: Path, graded: Path, task_id: str) -> bool:
    """Whether both views of a task are on disk, whole, and still the bytes that were derived.

    Both, not either: the world an agent drives and the grader's view of it are built together and
    a run that found only the first would serve an episode nothing could grade.

    **Two path existences were not a task.** This asked whether the served ``dbs/todoist.jsonl``
    and the graded ``ground_truth`` were there, and answered yes for a tree with everything else
    missing, for one whose databases had been changed after derivation, and for one whose seal had
    come off. Every one of those is reused rather than rebuilt, and what is reused is the world an
    episode starts in and the baseline it is graded against. So the question is now asked of a
    manifest written when the task was derived: every path, its mode, its size and its digest (see
    :func:`_write_manifest`).

    It costs about two milliseconds for the pair, which is a task's 28 KB and a grading view's
    52 KB across some thirty files, once per episode against a construction of about four
    seconds."""
    return _manifest_holds(derived / "tasks" / task_id) and _manifest_holds(
        graded / "tasks" / task_id
    )


def _grading_view(target: Path, view: Path, answers: Path) -> None:
    """The grader's view of a derived task: the same world, with the answers linked back in.

    Independent copies of the served task's own files. They were hard links, one instance on disk
    under two names, which made this view move whenever a served episode wrote through its own: the
    baseline the grader diffs against was editable by the thing being graded. The answers are
    copied from the corpus, and this tree is the only place in the port that names them."""
    building = _staging(view.parent, view.name)
    (building / "dbs").mkdir(parents=True)
    for entry in sorted(target.iterdir()):
        # The served task's manifest is not this tree's: this one holds the answers as well, so
        # it is a different set of files and gets a record of its own below.
        if entry.name not in ("dbs", _MANIFEST):
            _materialise(entry, building / entry.name)
    for entry in sorted((target / "dbs").iterdir()):
        _materialise(entry, building / "dbs" / entry.name)
    _materialise(answers, building / "ground_truth")
    # Sealed for the same reason the served copy is: this is the baseline the evaluator diffs
    # against, and a baseline that anything can edit is not one.
    _seal(building)
    _write_manifest(building)
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
    by construction.

    **The invariant, stated once.** *Nothing an episode can reach through its served root is both
    shared with another episode and writable by this one.* The copy of the task under this view is
    the episode's own and is writable; everything else it can reach is shared and read-only,
    sealed by :func:`_seal` when it is built.

    That covers the shared *tasks* as well as the shared base, and it did not use to. The links
    below name the derived tree, and the derived tree's own task cache is the pristine source that
    every later view is copied out of: a served worker that could write to it would be writing
    into what the next episode, or the other arm of its own pair, starts from. So a published task
    is sealed in its staging directory before it is given its name, and the directory they are
    published into is sealed too, opened only under the derivation lock. What makes two arms of a
    pair comparable is that the placebo arm cannot observe anything its twin did, and the reason
    it cannot is that its twin could not change anything they both read.

    **Which covers the names as well as the bytes.** These links are absolute, so what an episode
    resolves is the entry's content *and* the name that reaches it, and a name lives in its
    parent. Sealing every entry and leaving the parent open left the second half of the path
    writable: ``base_dbs`` could be renamed aside and something else put there under the same
    name, and every view that resolved it afterwards would follow. The shared parent is therefore
    sealed too, and opened only under the derivation lock (see :func:`derive_root`).

    **What the invariant rests on, and where it stops.** It rests on file permissions, and the
    worker runs as the user that owns those files, so code that goes looking can chmod them back.
    An ordinary write fails, upstream itself never writes there, and cross-arm contamination by
    accident is closed; a deliberate one is not, in the same way and for the same reason that the
    corpus stays host-readable (see the README). shojin-lab/shogym#140 mounts the shared base into
    the worker's container read-only, which is the boundary rather than the convention."""
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
    _unseal(served)
    shutil.rmtree(served, ignore_errors=True)
    shutil.copytree(derived / "tasks" / task_id, served)
    # The shared task is sealed and `copytree` carries its modes across, so the copy arrives
    # read-only. This one is the episode's own and has to be writable: it is the only thing under
    # the served root that is, which is the invariant above stated as a permission.
    _unseal(served)
    return view


def derive_root(
    *, original: Path, derived: Path, verify: Optional[Callable[[str], None]] = None
) -> Path:
    """Materialise the parts of a corpus that no task changes, and return the derived root.

    Copies rather than symlinks, for the reason :func:`derive_task` gives: a symlink to the
    corpus's own ``datasets`` directory is a path to the corpus, and the corpus holds every task's
    answers. The base databases are 129 MB and the API documentation 4.5 MB, so this is built once
    per corpus and every episode's view names it rather than copying it.

    **Staged and published, never built in place.** This used to unseal, delete, copy and mark
    the final directory while holding :func:`_locked`, which at the time yielded with no exclusion
    at all on a filesystem whose ``flock`` it could not take. Under it, two cold processes both
    deleted and rebuilt the same live target, and either could publish the completeness marker over
    a tree the other was halfway through writing. So each entry is built under a staging name of
    this process's own, sealed and marked there, and given its final name by one rename; a loser
    finds the target already published and drops what it built. That is the protocol
    :func:`derive_task` already uses, and it is correct without a lock rather than because of one.
    The lock is now *required* rather than advisory anyway, for the modes rather than the
    contents: staging cannot make a permission window safe, and this one opens a published
    directory. Both defences are kept, because they answer different failures.

    **Sealed read-only, which is what lets a view name it.** Every episode's served root reaches
    these directories, so an episode that could write to one could leave something in the next
    episode's inputs, including the other arm of its own pair's. Upstream never writes here (it
    reads ``base_dbs``, ``datasets`` and ``api_docs``, and writes only under the episode's own
    output root), so read-only costs nothing and closes the ordinary write. It does not close a
    deliberate one, because the worker runs as the user that owns these files: see
    :func:`derive_view` for where that residual is stated and what closes it.

    **And the directory holding them is sealed too, which sealing each entry does not do.** A
    view names these by absolute path, so what an episode resolves is not only the entry's bytes
    but the *name* that reaches them. This used to seal every entry and leave their parent
    owner-writable: the entries could not be written through, and ``base_dbs`` could still be
    renamed aside and a directory of the episode's own choosing put there under the same name,
    which every current and later view would then resolve to. So the parent is opened only for
    the length of a build (see :func:`_opened`), under the lock, and is read-only the rest of the
    time — the same treatment the published tasks directory beside it already had.

    That stops at this directory and not above it. The seeded root above holds this port's own
    cache stamp and is written by cold construction, and the cache root above that is where the
    port provisions; both stay writable, so a process willing to work one level up can still move
    ``data`` itself. Nor is any of it a boundary against a process that means to defeat it, which
    is the same residual :func:`derive_view` states: the worker runs as the user that owns these
    files and can chmod them back. shojin-lab/shogym#140 mounts the shared base into the worker's
    container read-only, which is a boundary rather than a convention, and it is what closes both
    the modes and the names.

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
        outstanding = [
            entry
            for entry in sorted(original.iterdir())
            if entry.name != "tasks"
            and not (_complete(derived / entry.name) and _sealed(derived / entry.name))
        ]
        if outstanding or not (derived / "tasks").is_dir():
            with _opened(derived):
                (derived / "tasks").mkdir(exist_ok=True)
                for entry in outstanding:
                    if verify is not None:
                        verify(entry.name)
                    target = derived / entry.name
                    building = _staging(derived, entry.name)
                    _materialise(entry, building)
                    _mark_complete(building)
                    _seal(building)
                    _publish(building, target, replacing=True)
        elif _writable(derived):
            # Complete, and built by a head that sealed the entries and left their parent open.
            # Sealed here, without touching anything under it: the entries are already correct
            # and what was missing is the name boundary above them.
            _chmod(derived, 0o555)
    return derived


def _sealed(target: Path) -> bool:
    """Whether every node under ``target`` has had its write bits taken off.

    Walked rather than inferred. The top-level mode used to stand for the whole subtree, on the
    reasoning that :func:`_seal` sets it last, and that reasoning depended on every nested
    ``chmod`` having succeeded. One that failed left a writable child permanently hidden behind a
    read-only top, which is the shape of failure the seal exists to prevent, reported as success.
    The walk is a few hundred ``lstat`` calls on this corpus, once per construction."""
    if not target.exists():
        return False
    if _writable(target):
        return False
    if target.is_dir() and not target.is_symlink():
        return all(_sealed(child) for child in sorted(target.iterdir()))
    return True


def _writable(target: Path) -> bool:
    try:
        return bool(target.lstat().st_mode & 0o222)
    except OSError:
        return True


@contextmanager
def _opened(directory: Path) -> "Iterator[None]":
    """Make ``directory`` itself writable for the length of the body, then seal it again.

    The directory node only, never what is under it. The published tasks below it are sealed and
    stay sealed; what this opens is the right to add a name to the directory, which is what
    publishing one needs and what a served worker must not have. A worker that could add or remove
    a name here could take a task out of the cache the next episode is built from.

    Sealed again in a ``finally``, so a build that raises does not leave the directory open."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o755)
    except OSError:
        pass
    try:
        yield
    finally:
        _chmod(directory, 0o555)


def _staging(parent: Path, name: str) -> Path:
    """A directory to build ``name`` in, under a name no other process can be using.

    The pid alone was the old answer and it is not unique enough: pids are recycled, and a
    crashed builder's leftovers under the same number would be deleted out from under a live one.
    Eight random bytes beside it make the name this call's."""
    parent.mkdir(parents=True, exist_ok=True)
    building = parent / f".{name}.{os.getpid()}.{secrets.token_hex(8)}.building"
    _unseal(building)
    shutil.rmtree(building, ignore_errors=True)
    return building


def _seal(target: Path) -> None:
    """Take the write bits off ``target`` and everything under it, children first.

    Children first so that the top-level mode is the last thing to change: a caller that finds the
    top read-only can then take it as proof the whole entry was sealed, rather than as proof that
    a seal had begun.

    A symlink is left alone rather than followed: ``chmod`` through one changes the mode of
    whatever it names, which here would be something outside the tree being sealed. Nothing this
    is called on is a symlink; that is the point of :func:`_materialise`, and this is what keeps
    it true if it ever stops being."""
    if target.is_symlink():
        return
    if target.is_dir():
        for child in sorted(target.iterdir()):
            _seal(child)
    _chmod(target, 0o555 if target.is_dir() else 0o444)


def _unseal(target: Path) -> None:
    """Put the owner's write bit back on ``target`` and everything under it, top down.

    Top down, because removing a file needs write permission on the directory holding it, so the
    parent has to be writable before the child can be reached for. Symlinks are skipped, for the
    reason :func:`_seal` gives.

    Unlike :func:`_seal`, a failure here is tolerated: this is called to make something removable,
    and the removal that follows reports its own failure. There is no invariant resting on a mode
    this function set."""
    if target.is_symlink() or not target.exists():
        return
    try:
        if target.is_dir():
            os.chmod(target, 0o755)
            for child in sorted(target.iterdir()):
                _unseal(child)
        else:
            os.chmod(target, 0o644)
    except OSError:
        pass


def _chmod(target: Path, mode: int) -> None:
    """``chmod``, and a failure is a failure.

    This used to swallow every error, on the reasoning that a tree it could not seal was a weaker
    guarantee rather than a broken derivation. That was wrong in the direction that matters: the
    seal is what keeps one episode out of the next one's inputs, and the caller that checks it
    reads a mode. A chmod that failed silently left a writable child under a read-only parent,
    which reads as sealed and is not. A filesystem that cannot do this cannot host the served
    corpus, and saying so at derivation is better than a run that discovers it in its numbers."""
    os.chmod(target, mode)


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


#: Written into a derived task once every file of it is there, and read before that task is
#: reused. A task without one is a task some process was interrupted in the middle of, so this is
#: the completion marker for a derived task as well as its contents.
_MANIFEST = ".shogym-manifest"


def _write_manifest(target: Path) -> None:
    """Record every node of a derived task: its path, its kind, its mode, its size and its bytes.

    **Written last, inside the staging directory, which is what makes it atomic.** A manifest is
    only a completion marker if it cannot appear before the thing it completes, and this one
    appears under the task's real name in the same ``rename`` that publishes the tree. There is no
    ordering in which a reader sees a manifest over a half-built task.

    Modes are in it because the seal is one of the properties being reused: a node whose write bit
    came back is a shared task an episode could edit, and every episode of that task and the other
    arm of its pair start from what it holds. Sizes are in it beside the digests because a size
    mismatch is a rebuild decided without reading the file, and because a digest that is only ever
    compared to itself is one nobody would notice going missing.

    The manifest does not describe itself. Its own bytes are what is being trusted, so signing
    them with a digest they contain would prove nothing; what it rests on is the publish, which
    gives the whole tree one name at one instant. Its mode is still checked, because a manifest
    anything can rewrite is a claim about a tree rather than a record of one.

    Called after :func:`_seal` and on a staging tree, so the directory is opened for one write and
    closed again. Nothing else knows that name yet."""
    entries: List[List[Any]] = []
    for path in sorted(_nodes(target)):
        relative = str(path.relative_to(target))
        mode = path.lstat().st_mode & 0o7777
        if path.is_symlink():
            entries.append([relative, "l", mode, 0, os.readlink(path)])
        elif path.is_dir():
            entries.append([relative, "d", mode, 0, ""])
        else:
            data = path.read_bytes()
            entries.append([relative, "f", mode, len(data), hashlib.sha256(data).hexdigest()])
    manifest = target / _MANIFEST
    _chmod(target, 0o755)
    # Whatever was there is not this tree's record. A staging directory built by copying another
    # sealed tree can arrive holding that tree's manifest, read-only.
    manifest.unlink(missing_ok=True)
    manifest.write_text(json.dumps(entries, sort_keys=True))
    _chmod(manifest, 0o444)
    _chmod(target, 0o555)


def _manifest_holds(target: Path) -> bool:
    """Whether ``target`` is exactly the tree its manifest says was derived there.

    Every entry, and nothing besides: a file that is gone, a file whose bytes changed, a node whose
    write bit came back and a path nobody derived are all answered the same way, by saying this is
    not a derived task and letting the caller build one. The manifest itself is the one path
    excluded from the comparison, for the reason :func:`_write_manifest` gives.

    Anything unreadable is a no rather than an error, and so is a manifest whose shape this code
    does not recognise. This is asked on the ordinary warm path, before any lock, and the honest
    answer to "I could not tell" is the one that rebuilds; a manifest written by a version of this
    that recorded something else is a tree to build again rather than a crash at construction."""
    try:
        entries = json.loads((target / _MANIFEST).read_text())
        if _writable(target / _MANIFEST):
            return False
        if {str(path.relative_to(target)) for path in _nodes(target)} != {
            str(entry[0]) for entry in entries
        }:
            return False
        for relative, kind, mode, size, content in entries:
            path = target / relative
            stamp = path.lstat()
            if stamp.st_mode & 0o7777 != mode:
                return False
            if kind == "l":
                if not path.is_symlink() or os.readlink(path) != content:
                    return False
            elif kind == "d":
                if path.is_symlink() or not path.is_dir():
                    return False
            else:
                if path.is_symlink() or not path.is_file() or stamp.st_size != size:
                    return False
                if hashlib.sha256(path.read_bytes()).hexdigest() != content:
                    return False
    except (LookupError, OSError, TypeError, ValueError):
        return False
    return True


def _nodes(target: Path) -> "Iterator[Path]":
    """Every path under ``target`` but the manifest, directories included, links never followed."""
    for directory, children, names in os.walk(target, followlinks=False):
        here = Path(directory)
        linked = [child for child in children if (here / child).is_symlink()]
        children[:] = sorted(child for child in children if child not in linked)
        for child in children:
            yield here / child
        for name in sorted([*names, *linked]):
            if here == target and name == _MANIFEST:
                continue
            yield here / name


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

    ``replacing`` is for the one caller that has to displace an existing target: a shared base
    entry left incomplete or partly sealed by a crash. It is still not a build in place. The
    finished tree is renamed *aside* first and the staged one renamed in, so the name is never
    absent and never holds a half-made tree; the displaced one is then removed at leisure."""
    if target.exists() and not replacing:
        _unseal(building)
        shutil.rmtree(building, ignore_errors=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    displaced = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.displaced")
    try:
        if target.exists():
            os.replace(target, displaced)
        os.replace(building, target)
    except OSError:
        _unseal(building)
        shutil.rmtree(building, ignore_errors=True)
        if not target.exists():
            raise
    finally:
        _unseal(displaced)
        shutil.rmtree(displaced, ignore_errors=True)


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
    "derive_task",
    "request_description",
    "seeding",
]
