"""Reading a recorded workflow history the way a reader holding only the run would."""

from __future__ import annotations

from typing import Tuple

from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowHistory


def the_activity_that_failed(history: WorkflowHistory) -> Tuple[str, str]:
    """Return the type and the id of the one Activity ``history`` records as having failed.

    The failure event does not carry the Activity's identity. It carries the id of the event that
    scheduled the Activity, and the scheduling is where the type and the id were written down. So
    that is the join a reader makes, and it is the join a row's ``failure_activity_id`` has to
    land on for the row to be a handle to anything.
    """
    [failed] = [
        event
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED
    ]
    scheduled_event_id = failed.activity_task_failed_event_attributes.scheduled_event_id
    [scheduled] = [event for event in history.events if event.event_id == scheduled_event_id]
    written = scheduled.activity_task_scheduled_event_attributes
    return written.activity_type.name, written.activity_id
