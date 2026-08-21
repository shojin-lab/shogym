"""A synthetic ORCA-bench dataset directory, in the real on-disk shape.

The real dataset is 192 MB, downloaded on demand and never vendored, and every real
``task.toml`` carries that task's answer, which is exactly what must not be committed to a public
repo. So the offline tests build their own tiny dataset here instead: the same file layout
(``instruction.md`` + ``task.toml`` + ``environment/Dockerfile``), the same ``[metadata]`` keys,
and answers that are visibly invented. A network-marked test re-checks the same properties
against one real task pulled into the cache.

The invented answers are chosen to be unmistakable in a failure message (``syntheticPaymentDrain``
appears nowhere in an instruction), so a redaction test that catches one is pointing at a real
leak rather than at a coincidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

SNAPSHOT_A = "20260419T000000Z-aaaaaaaaaaaaaaaa"
SNAPSHOT_B = "20260420T000000Z-bbbbbbbbbbbbbbbb"

# The base image every real task's Dockerfile builds from.
BASE_IMAGE = "orcabench/sre-otel-snapshot:data-0418-harbor-template"

_INSTRUCTION = """\
# Root Cause Analysis

You are an expert site reliability engineer. The current time is Apr 20, 2026 at 00:00 ET.

## Task Description

The following issue was reported at {reported} ET:

{complaint}

NOTE:
* The time that an issue was reported is not necessarily the same as the time the incident
  actually began. Your task is to pinpoint the root cause events and the times of their
  occurrence.
* There may be multiple root causes or no root cause at all.

## Instructions

Write a structured incident RCA report to `/app/report.md`. Cite the exact PromQL query behind
every metric claim.
"""


@dataclass(frozen=True)
class SyntheticEvent:
    """One ground-truth root-cause event, in the shape ``[[metadata.events]]`` uses."""

    event_id: str
    root_cause: str
    event_time: str


@dataclass(frozen=True)
class SyntheticTask:
    """One synthetic task: the labels the index reads plus the answers it must never publish."""

    name: str
    difficulty: str
    section: str
    snapshot: str
    complaint: str
    reported: str
    flag: Optional[str] = None
    description: Optional[str] = None
    qid: str = ""
    incident_time: Optional[str] = None
    quiet_window_start: Optional[str] = None
    quiet_window_end: Optional[str] = None
    events: List[SyntheticEvent] = field(default_factory=list)

    @property
    def is_control(self) -> bool:
        return not self.events


# Four tasks: three incidents across the difficulty ladder and both snapshots, plus one control.
# Names are sorted-order-significant: the index's canonical order is by name.
TASKS: List[SyntheticTask] = [
    SyntheticTask(
        name="aaaa000000000001",
        difficulty="easy",
        section="exact",
        snapshot=SNAPSHOT_A,
        complaint="users cannot complete their purchase",
        reported="5:02 PM yesterday",
        flag="syntheticPaymentDrain",
        description="Synthetic fault: drain the payment service",
        qid="syn-d1-syntheticPaymentDrain-01-hard",
        incident_time="2026-04-19T20:52:46Z",
        events=[
            SyntheticEvent(
                event_id="syn-d1-syntheticPaymentDrain",
                root_cause="syntheticPaymentDrain",
                event_time="2026-04-19T20:52:46.740910+00:00",
            )
        ],
    ),
    SyntheticTask(
        name="aaaa000000000002",
        difficulty="medium",
        section="exact_range",
        snapshot=SNAPSHOT_A,
        complaint="the product page is slow",
        reported="between 6:00 PM and 6:30 PM yesterday",
        flag="syntheticCatalogStall",
        description="Synthetic fault: stall the catalog service",
        qid="syn-d2-syntheticCatalogStall-02-easy",
        incident_time="2026-04-19T22:10:00Z",
        events=[
            SyntheticEvent(
                event_id="syn-d2-syntheticCatalogStall",
                root_cause="syntheticCatalogStall",
                event_time="2026-04-19T22:10:00.000000+00:00",
            ),
            SyntheticEvent(
                event_id="syn-d2-syntheticCartWedge",
                root_cause="syntheticCartWedge",
                event_time="2026-04-19T22:40:00.000000+00:00",
            ),
        ],
    ),
    SyntheticTask(
        name="aaaa000000000003",
        difficulty="hard",
        section="broad_range_day",
        snapshot=SNAPSHOT_B,
        complaint="users are reporting site issues",
        reported="sometime yesterday",
        flag="syntheticAdMeltdown",
        description="Synthetic fault: melt down the ad service",
        qid="syn-d3-syntheticAdMeltdown-03-universal",
        incident_time="2026-04-19T09:15:00Z",
        events=[
            SyntheticEvent(
                event_id="syn-d3-syntheticAdMeltdown",
                root_cause="syntheticAdMeltdown",
                event_time="2026-04-19T09:15:00.000000+00:00",
            )
        ],
    ),
    SyntheticTask(
        name="aaaa000000000004",
        difficulty="easy",
        section="control",
        snapshot=SNAPSHOT_B,
        complaint="users are reporting site issues",
        reported="at 9:00 AM today",
        qid="syn-d4-control-04-universal",
        quiet_window_start="2026-04-19T22:40:09.898386+00:00",
        quiet_window_end="2026-04-20T14:14:18.261797+00:00",
    ),
]


def _task_toml(task: SyntheticTask) -> str:
    lines = [
        'schema_version = "1.3"',
        "artifacts = []",
        "",
        "[task]",
        f'name = "orca-bench/{task.name}"',
        "",
        "[metadata]",
        'category = "sre"',
        'tags = [ "root cause analysis", "telemetry",]',
        f'qid = "{task.qid}"',
        f'user_facing_issue = "{task.complaint}"',
        f'section = "{task.section}"',
        f'difficulty = "{task.difficulty}"',
        'granularity = "hard"',
        f'reported_styled = "{task.reported} ET"',
        'current = "2026-04-20T04:00:00.000000+00:00"',
    ]
    for key in ("flag", "description", "incident_time", "quiet_window_start", "quiet_window_end"):
        value = getattr(task, key)
        if value:
            lines.append(f'{key} = "{value}"')
    for event in task.events:
        lines += [
            "",
            "[[metadata.events]]",
            f'event_id = "{event.event_id}"',
            f'root_cause = "{event.root_cause}"',
            f'event_time = "{event.event_time}"',
        ]
    lines += ["", "[verifier]", "timeout_sec = 1200.0", "", "[agent]", "timeout_sec = 3600.0", ""]
    return "\n".join(lines)


def write_task(task_dir: Path, task: SyntheticTask) -> Path:
    """Write one synthetic task, complete in the sense the loader validates.

    Every file the port requires of a cached task is written, with the verifier files stubbed:
    their *contents* are never read offline (phase 1 parses only the three the index and
    ``describe`` use), but their presence is what the completeness check is about."""
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    task_dir.joinpath("instruction.md").write_text(
        _INSTRUCTION.format(reported=task.reported, complaint=task.complaint), encoding="utf-8"
    )
    task_dir.joinpath("task.toml").write_text(_task_toml(task), encoding="utf-8")
    task_dir.joinpath("environment", "Dockerfile").write_text(
        f"FROM {BASE_IMAGE}\nENV SNAPSHOT_NAME={task.snapshot}\n", encoding="utf-8"
    )
    task_dir.joinpath("environment", "docker-compose.yaml").write_text(
        "services:\n  main:\n    image: ${MAIN_IMAGE_NAME}\n", encoding="utf-8"
    )
    task_dir.joinpath("tests", "test.sh").write_text(
        "#!/bin/bash\npython /tests/check_prediction.py\n", encoding="utf-8"
    )
    task_dir.joinpath("tests", "check_prediction.py").write_text(
        "# the task's own LLM judge (stub)\n", encoding="utf-8"
    )
    task_dir.joinpath("tests", "expected.json").write_text(
        '{"events": [], "current": "2026-04-20T04:00:00.000000+00:00"}', encoding="utf-8"
    )
    return task_dir


def write_dataset(root: Path, tasks: Optional[List[SyntheticTask]] = None) -> Path:
    """Write a synthetic dataset directory under ``root`` and return it."""
    root.mkdir(parents=True, exist_ok=True)
    for task in tasks if tasks is not None else TASKS:
        write_task(root / task.name, task)
    return root


def answers(task: SyntheticTask) -> List[str]:
    """Every invented answer string for one task: what must never reach an agent.

    Deduplicated and sorted: a flag is both ``metadata.flag`` and its event's ``root_cause``, and
    a leak check reads better when each distinct string is named once."""
    out = {task.qid}
    for key in ("flag", "description", "incident_time", "quiet_window_start", "quiet_window_end"):
        value = getattr(task, key)
        if value:
            out.add(value)
    for event in task.events:
        out.update({event.event_id, event.root_cause, event.event_time})
    return sorted(out)


def by_name() -> Dict[str, SyntheticTask]:
    return {task.name: task for task in TASKS}


__all__ = [
    "BASE_IMAGE",
    "SNAPSHOT_A",
    "SNAPSHOT_B",
    "SyntheticEvent",
    "SyntheticTask",
    "TASKS",
    "answers",
    "by_name",
    "write_dataset",
    "write_task",
]
