"""The ORCA-bench task model and index: pure functions over a cached dataset directory.

Each task is a self-contained Harbor package on disk::

    <task name>/
      instruction.md              the prompt (the ONLY thing describe() may surface)
      task.toml                   [metadata], which carries the FULL ground truth
      environment/Dockerfile      pins the snapshot the stack replays (ENV SNAPSHOT_NAME=...)
      environment/docker-compose.yaml
      tests/                      the judge (check_prediction.py) + expected.json + rubrics
      solution/                   the oracle

**The redaction property this module exists to hold:** ``task.toml``'s ``[metadata]`` table
contains the answer: the feature flag that caused the incident, every root-cause event and its
time, and the quiet-window bounds for a control task. Upstream is safe because Harbor never
mounts ``task.toml`` into the agent's container; a shogym env has no such accident to rely on, so
:meth:`OrcaTaskRef.instructions` reads ``instruction.md`` and nothing else, and
:func:`answer_strings` names exactly what must never appear in anything published to an agent.

A :class:`OrcaTaskRef` carries the labels a caller slices on: difficulty, section, control, and
the snapshot the task replays. Grouping by snapshot is what lets a runner stage one snapshot and
run every task that shares it; 755 tasks span 125 snapshots.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

# The difficulty ladder the benchmark publishes per-tier numbers for. Read from `difficulty`,
# never from `granularity`: the two ladders overlap in the word "hard" while meaning different
# tiers, so reading the wrong field silently redefines the published columns.
DIFFICULTIES = ("easy", "medium", "hard")

# The `section` values: how precisely the report time is stated, plus `control` (no incident).
SECTIONS = (
    "exact",
    "exact_range",
    "broad_range_time_of_day",
    "broad_range_day",
    "control",
)

_SNAPSHOT_RE = re.compile(r"^ENV\s+SNAPSHOT_NAME=(\S+)\s*$", re.MULTILINE)

# The `[metadata]` keys that are, or contain, the answer. `describe()` never reads task.toml at
# all, so this is not a filter. It is the checkable statement of what redaction means, used by
# the tests and available to anyone auditing a published trace.
_ANSWER_KEYS = (
    "flag",  # the feature flag that caused the incident
    "description",  # what that flag does
    "qid",  # the question id, which embeds the flag and the variant
    "incident_time",
    "current",  # the snapshot's "now", which brackets the incident
    "quiet_window_start",
    "quiet_window_end",
)
_ANSWER_EVENT_KEYS = ("event_id", "root_cause", "event_time")


class TaskIndexError(RuntimeError):
    """A cached task directory is missing a file the index needs, or carries unexpected labels."""


@dataclass(frozen=True)
class OrcaTaskRef:
    """One indexed task: its identity plus the labels a caller slices and stages on.

    Deliberately carries **no** ground truth. ``section`` is a label, not an answer, but it does
    say whether a task is a control, so it never reaches an agent either: it exists for slicing
    a run and for reproducing the benchmark's per-tier numbers.

    ``name`` is the identity; ``dataset_index`` is **provenance, never a selector**. An env may
    be built over a *slice* of the dataset, and then the id an episode is served and recorded
    under is that task's position in the slice, which is not its position in the 755. Keeping the
    canonical number under its own name is what stops the two being interchanged: an id resolved
    in the wrong space either raises or, worse, silently selects a different task.
    """

    name: str
    dataset_index: int  # position in the canonical (name-sorted) 755-task order. Provenance only.
    difficulty: str  # easy | medium | hard
    section: str  # see SECTIONS
    is_control: bool  # a task with no incident to find
    snapshot: str  # the telemetry snapshot this task replays (phase-2 staging key)
    task_dir: Path

    @property
    def instruction_path(self) -> Path:
        return self.task_dir / "instruction.md"

    def instructions(self) -> str:
        """The task's ``instruction.md``, verbatim: the only task text an agent may see."""
        return self.instruction_path.read_text(encoding="utf-8")


def load_metadata(task_dir: Path) -> Dict[str, Any]:
    """Parse a task's ``task.toml`` ``[metadata]`` table (ground truth included)."""
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        raise TaskIndexError(f"no task.toml in {task_dir}")
    return dict(tomllib.loads(toml_path.read_text(encoding="utf-8")).get("metadata", {}))


def read_snapshot(task_dir: Path) -> str:
    """The snapshot id the task's environment image replays (``ENV SNAPSHOT_NAME=`` )."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        raise TaskIndexError(f"no environment/Dockerfile in {task_dir}")
    match = _SNAPSHOT_RE.search(dockerfile.read_text(encoding="utf-8"))
    if match is None:
        raise TaskIndexError(f"no SNAPSHOT_NAME in {dockerfile}")
    return match.group(1)


def load_ref(task_dir: Path, dataset_index: int) -> OrcaTaskRef:
    """Build one :class:`OrcaTaskRef` from a cached task directory."""
    metadata = load_metadata(task_dir)
    difficulty = str(metadata.get("difficulty", ""))
    if difficulty not in DIFFICULTIES:
        raise TaskIndexError(
            f"{task_dir.name} reports difficulty {difficulty!r}, expected one of "
            f"{list(DIFFICULTIES)}; the dataset's ladder moved and the per-tier slices would "
            "silently change meaning"
        )
    return OrcaTaskRef(
        name=task_dir.name,
        dataset_index=dataset_index,
        difficulty=difficulty,
        section=str(metadata.get("section", "")),
        # Upstream's own definition (the leaderboard's `task_groups._label`): a control task is
        # one with no incident to find. `section == "control"` agrees on every task of the pinned
        # revision, but the events list is the thing the judge actually branches on.
        is_control=not bool(metadata.get("events")),
        snapshot=read_snapshot(task_dir),
        task_dir=task_dir,
    )


def load_index(dataset: Path, names: Optional[Iterable[str]] = None) -> List[OrcaTaskRef]:
    """Index a cached dataset directory, ordered by name.

    Name order is the canonical order: it is stable across hosts and across re-downloads (the
    hub's own row order is not), so ``dataset_index`` means the same task everywhere. A caller
    that slices this list gets refs whose ``dataset_index`` still refers to the full order, which
    is why it is provenance rather than a selector (see :class:`OrcaTaskRef`).

    ``names`` indexes exactly those identities and nothing else, which is how the provisioned
    cache is read: the authenticated set is the only thing that should decide how many tasks
    there are and what each id refers to, so it is passed in rather than inferred from whatever
    directories happen to be on disk. Without it every task-shaped directory is indexed, which is
    right for a directory a caller vouches for by hand and wrong for a cache. Either way each
    ref's ``dataset_index`` is its position in the dataset's own name-sorted order, so a
    constrained load reports the same numbers the unconstrained one does."""
    if not dataset.is_dir():
        raise TaskIndexError(f"no dataset directory at {dataset}")
    # The canonical order is the dataset's, not the request's, and it is established before any
    # filtering: `dataset_index` has to mean the same thing whether or not a caller asked for a
    # subset, or a subset would claim a provenance it does not have.
    canonical = sorted(p.name for p in dataset.iterdir() if (p / "task.toml").is_file())
    position = {name: index for index, name in enumerate(canonical)}
    if names is None:
        wanted = canonical
    else:
        wanted = sorted(names)
        seen = set()
        for name in wanted:
            if name in seen:
                raise TaskIndexError(f"task {name!r} was requested more than once")
            seen.add(name)
        absent = [name for name in wanted if name not in position]
        if absent:
            raise TaskIndexError(
                f"{len(absent)} of the requested tasks are absent from {dataset} "
                f"(e.g. {absent[:3]})"
            )
    return [load_ref(dataset / name, position[name]) for name in wanted]


# ----- slicing -----


def select(
    refs: Iterable[OrcaTaskRef],
    *,
    difficulty: Optional[str] = None,
    section: Optional[str] = None,
    is_control: Optional[bool] = None,
    snapshot: Optional[str] = None,
) -> List[OrcaTaskRef]:
    """The subset of ``refs`` matching every given label, in the order it was given."""
    out = []
    for ref in refs:
        if difficulty is not None and ref.difficulty != difficulty:
            continue
        if section is not None and ref.section != section:
            continue
        if is_control is not None and ref.is_control is not is_control:
            continue
        if snapshot is not None and ref.snapshot != snapshot:
            continue
        out.append(ref)
    return out


def group_by_snapshot(refs: Iterable[OrcaTaskRef]) -> Dict[str, List[OrcaTaskRef]]:
    """Group tasks by the snapshot they replay, snapshots in first-seen order.

    Staging a snapshot is the expensive part of running a task, so a runner that walks these
    groups pays for each snapshot once instead of once per task."""
    groups: Dict[str, List[OrcaTaskRef]] = {}
    for ref in refs:
        groups.setdefault(ref.snapshot, []).append(ref)
    return groups


# ----- redaction -----


def answer_strings(task_dir: Path) -> Set[str]:
    """The ground-truth strings a published task contract must never contain.

    Drawn from ``task.toml``'s ``[metadata]``: the causing flag and what it does, the question
    id (which embeds the flag), the incident and event times, and a control task's quiet-window
    bounds. Deliberately **not** the fields the benchmark hands the agent on purpose: the
    reported time, its phrasing, and the user-facing complaint are all in ``instruction.md``
    upstream, so treating them as secrets would assert something false.
    """
    metadata = load_metadata(task_dir)
    out: Set[str] = set()
    for key in _ANSWER_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            out.add(value)
    for event in metadata.get("events") or []:
        if not isinstance(event, dict):
            continue
        for key in _ANSWER_EVENT_KEYS:
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                out.add(value)
    return out


__all__ = [
    "DIFFICULTIES",
    "SECTIONS",
    "OrcaTaskRef",
    "TaskIndexError",
    "answer_strings",
    "group_by_snapshot",
    "load_index",
    "load_metadata",
    "load_ref",
    "read_snapshot",
    "select",
]
