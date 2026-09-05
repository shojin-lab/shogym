"""Guard tests for the AutomationBench cell runner: what it composes, what it reads, what it runs.

Three halves, and none needs a model or a durable service. The composition is a pure function of
the roster and the schedule, so what is checked there is that a schedule name decides what the
agent is told and that nothing else does. The table is a pure function of three files, so what is
checked there is the join: a score belongs to the task the roster says it does, a position nobody
reached says so, and a tool call is counted against the attempt it named. The launch is a pair of
commands, so what is checked there is what each container was given: the boundary this cell rests
on is a mount list, and a mount list is something a test can read.

Nothing here spawns the ``claude`` CLI or spends a token, and nothing here provisions the
AutomationBench upstream source: the grade identity this cell composes over is a declaration and
importing it reaches no upstream package.

Two tests at the end are the exception, and they are because of what they check: neither a network
nor an empty cache is something a fixture can answer for. So the two domains are stood up on this
host, once with the probe run from a container joined to the server's own network stack, and once
from a cache that starts empty. Both are skipped unless the host already holds both images,
because a check of the boundary has no business building an image to make its point.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tarfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

from examples.automationbench_cell import cell as launcher  # noqa: E402
from examples.automationbench_cell import pinned  # noqa: E402
from examples.automationbench_cell import sandbox  # noqa: E402
from examples.automationbench_cell import serve as cell  # noqa: E402
from examples.automationbench_cell import table as read_back  # noqa: E402
from shogym.envs.automationbench.protocol_v2 import AUTOMATIONBENCH_GRADE  # noqa: E402
from shogym.serve.protocol_v2 import GRADED_HORIZON  # noqa: E402
from shogym.serve.protocol_v2.gateway import terminal_manifest  # noqa: E402
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    AttemptRecord,
    PresentedMessage,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    DELIVER,
    HONEST_V1,
    NO_RELEASE,
    ORDINARY,
    WITHHOLD,
)
from shogym.task import TaskSpec, ToolManifest  # noqa: E402

CLAIM = "c" * 64

SPEC = TaskSpec(
    env_name="automationbench",
    task_id="0",
    instructions="the first task's own instructions",
    tools=[
        ToolManifest(
            name="api_fetch",
            description="route a REST call into this session's world",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}},
        ),
        ToolManifest(
            name="done",
            description="end the task",
            input_schema={"type": "object", "properties": {}},
            terminal_kind="score",
        ),
    ],
    horizon=52,
)


def composed(bodies: List[str], schedule: str):
    return cell.compose(
        SPEC,
        terminal_manifest(SPEC),
        bodies=bodies,
        release=cell.release_for(schedule),
        grade=AUTOMATIONBENCH_GRADE,
        claim_hash=CLAIM,
    )


# ----- the roster -----


def test_a_roster_is_read_in_the_order_it_was_written() -> None:
    # Order is part of what a rerun matches, so a roster is never sorted on the way in.
    assert cell.roster("4,0,2") == [4, 0, 2]
    assert cell.roster("3-6") == [3, 4, 5, 6]
    assert cell.roster("0, 2-4 ,9") == [0, 2, 3, 4, 9]


def test_a_roster_that_names_a_task_twice_is_refused() -> None:
    # Two positions over one task would be two rows in the table and two worlds' worth of work
    # scored against one benchmark task, which is a different measurement than the one asked for.
    with pytest.raises(ValueError, match="twice"):
        cell.roster("1,2,1")


def test_a_roster_that_names_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="names no task"):
        cell.roster(" , ")


def test_the_roster_this_cell_reruns_is_the_one_that_run_was_served() -> None:
    """The stream the earlier cell dispensed, recomputed from the seeds it drew it with.

    These are its first twenty and its last ten, read off that run's own records. The order is
    part of the measurement, so it is checked rather than trusted: a rerun over the same tasks in
    another order differs from the run it is being compared with in more than the serving.
    """
    stream = cell.cell_one_stream()
    assert len(stream) == 480
    assert stream[:10] == [247, 481, 266, 121, 326, 156, 348, 454, 414, 27]
    assert stream[-10:] == [341, 405, 155, 384, 255, 19, 81, 561, 198, 543]
    # A shorter rerun works a prefix of it: the same tasks, in the same order, and fewer of them.
    assert cell.roster("cell-one:5") == stream[:5]
    assert cell.roster("cell-one") == stream


@pytest.mark.parametrize("named", ["cell-one-typo", "cell-one:", "cell-one:x", "cell-one :2"])
def test_a_name_that_is_nearly_this_cells_roster_names_no_roster(named: str) -> None:
    # A near miss used to be the whole experiment: anything starting `cell-one` was accepted and
    # what followed the colon was passed to a slice. A roster is the measurement, so a spelling
    # this cell does not serve is refused at the boundary rather than turned into 480 tasks.
    with pytest.raises(ValueError, match="not this cell's roster"):
        cell.roster(named)


@pytest.mark.parametrize("size", ["0", "-1", "999"])
def test_a_prefix_this_stream_does_not_have_is_refused(size: str) -> None:
    # Zero got past parsing and failed later at the first position; a negative prefix quietly
    # meant all but one task and an oversized one quietly meant all of them.
    with pytest.raises(ValueError, match="prefix|not this cell's roster"):
        cell.roster(f"cell-one:{size}")


def test_the_held_out_fifth_is_not_in_the_stream_this_cell_serves() -> None:
    # The split is the earlier cell's, and its held-out tasks were never trained on. A rerun that
    # drew them would be measuring the serving contract over tasks that cell never saw.
    assert set(cell.cell_one_stream()).isdisjoint({2, 7, 8, 15, 16, 17, 18, 24, 28, 31})


def test_a_schedule_this_cell_does_not_serve_is_refused_rather_than_defaulted() -> None:
    # The regime is the pin, so a misspelled one has to fail loudly rather than quietly serve
    # whichever plan happens to be the default.
    with pytest.raises(ValueError, match="not a schedule"):
        cell.release_for("Never")


# ----- what the composition is -----


def test_the_queue_is_the_roster_in_order_one_attempt_to_a_task() -> None:
    start = composed(["first", "second", "third"], "immediate")
    assert [item.body for item in start.tasks] == ["first", "second", "third"]
    assert [item.task_position for item in start.tasks] == [0, 1, 2]
    assert len({item.attempt_id for item in start.tasks}) == 3
    assert start.profile == ORDINARY


def test_immediate_stamps_every_position_with_the_honest_policy() -> None:
    start = composed(["first", "second"], "immediate")
    assert all(row.creates_payload_obligation for row in start.assignments)
    assert [row.kind for row in start.dispositions] == [DELIVER, DELIVER]
    assert {row.cell for row in start.dispositions} == {HONEST_V1.cells[0]}


def test_never_creates_no_obligation_anywhere_and_says_why() -> None:
    # The column an analysis reads is what the generation did, so a Never roster has to read
    # afterwards the way it behaved: no row created a payload, because none did.
    start = composed(["first", "second"], "never")
    assert not any(row.creates_payload_obligation for row in start.assignments)
    assert [row.kind for row in start.dispositions] == [WITHHOLD, WITHHOLD]
    assert {row.reason for row in start.dispositions} == {NO_RELEASE}


def test_the_two_schedules_differ_in_the_release_plan_and_in_nothing_else() -> None:
    # This is the pin the rerun rests on: the same roster under two schedules is the same
    # queue, the same bodies and the same profile, and one released plan apart.
    bodies = ["first", "second", "third"]
    immediate, never = composed(bodies, "immediate"), composed(bodies, "never")
    assert [item.body for item in immediate.tasks] == [item.body for item in never.tasks]
    assert immediate.profile == never.profile == ORDINARY
    assert immediate.configuration_hash == never.configuration_hash
    assert immediate.release.release_plan_id != never.release.release_plan_id


def test_the_roster_file_is_the_join_and_carries_no_score(tmp_path: Path) -> None:
    start = composed(["first", "second"], "never")
    attempts = [item.attempt_id for item in start.tasks]
    path = cell.record_roster(
        tmp_path, domain="simple", schedule="never", positions=[7, 11], attempts=attempts
    )
    written = json.loads(path.read_text())
    assert written["schedule"] == "never" and written["domain"] == "simple"
    assert written["release_plan_id"] == cell.release_for("never").release_plan_id
    assert [entry["task"] for entry in written["tasks"]] == ["7", "11"]
    assert [entry["attempt_id"] for entry in written["tasks"]] == attempts
    assert "score" not in path.read_text()


# ----- reading it back -----


def message_id(position: int, kind: str) -> str:
    """One of an attempt's three message identifiers, in the shape the protocol mints them."""
    return f"{position}{kind}" * 16


def record(attempt: str, position: int, **overrides) -> AttemptRecord:
    """A sealed, delivered attempt, which is the row every other shape here is a departure from.

    The fields are the ones a generation writes: an attempt the run never reached is planned with
    nothing delivered against it, an attempt the stream floored carries its reason and a resolved
    obligation, and both come through here as the overrides that make them.
    """
    fields = {
        "attempt_id": attempt,
        "task_position": position,
        "payload_position": position,
        "state": "ack_presented",
        "terminal_tool": "done",
        "terminal_source": "agent",
        "canonicalization_version": "shogym.automationbench.1",
        "submission_digest": "d" * 64,
        "score": 0.5,
        "decode_state": "decoded",
        "seal_ordinal": position,
        "final_failure": None,
        "deadline_expired": False,
        "task_message_id": message_id(position, "a"),
        "task_delivered": True,
        "ack_message_id": message_id(position, "b"),
        "ack_delivered": True,
        "payload_message_id": message_id(position, "c"),
        "payload_delivered": True,
        "creates_payload_obligation": True,
        "payload_state": "presented",
        "payload_policy": HONEST_V1.policy_name,
        "payload_disposition": DELIVER,
        "profile": ORDINARY,
    }
    fields.update(overrides)
    return AttemptRecord(**fields)  # type: ignore[arg-type]


def never_reached(attempt: str, position: int) -> AttemptRecord:
    """A position the manifest holds and the run stopped in front of.

    The generation answers with a row for every position it declared, so a run that stopped early
    is read back as planned positions and not as missing ones. Nothing was delivered against it
    and its obligation is where the manifest left it, which is assigned and unbuilt.
    """
    return record(
        attempt,
        position,
        state="planned",
        terminal_tool=None,
        terminal_source=None,
        submission_digest=None,
        score=None,
        decode_state=None,
        seal_ordinal=None,
        task_delivered=False,
        ack_delivered=False,
        payload_delivered=False,
        payload_state="assigned",
    )


def floored(attempt: str, position: int, reason: str) -> AttemptRecord:
    """An attempt the stream ended without a filing: the floor, the reason, and no payload.

    The ending resolves the obligation where it stands, so a floored row owes nothing afterwards
    however the schedule stamped it.
    """
    return record(
        attempt,
        position,
        state="final_failed",
        terminal_tool=None,
        terminal_source=None,
        submission_digest=None,
        score=0.0,
        decode_state=None,
        seal_ordinal=None,
        final_failure=reason,
        ack_delivered=False,
        payload_delivered=False,
        payload_state="final_failed",
    )


ROSTER = {
    "env": "automationbench",
    "domain": "public",
    "schedule": "immediate",
    "tasks": [
        {"task_position": 0, "task": "12", "attempt_id": "a" * 32},
        {"task_position": 1, "task": "31", "attempt_id": "b" * 32},
    ],
}


#: Which message each identifier here is one of, which is what lets the fixtures below build the
#: bytes the history delivered and the bytes the transcript holds out of the one identifier.
KINDS = {"a": "task", "b": "seal_ack", "c": "payload", "d": "done"}

#: The three messages one sealed and paid attempt is handed.
DELIVERED = [message_id(0, kind) for kind in ("a", "b", "c")]


def counted(refusals: int) -> read_back.Counted:
    """A refusal count the server wrote, which is the ordinary state of a run that served."""
    return read_back.Counted(count=refusals, absent=None)


def body(identifier: str) -> str:
    """The bytes one presented message goes as, which are the bytes both records are about."""
    return json.dumps(_message(identifier))


def presentations(identifiers: Optional[List[str]] = None) -> List[PresentedMessage]:
    """What the history answers that this generation handed over, in the order it handed it.

    The digest is of the same bytes the transcript fixture writes, because that is the whole of
    the comparison: two records of one delivery agree when the bytes do.
    """
    return [
        PresentedMessage(
            order=order,
            kind=KINDS[identifier[1]],
            message_id=identifier,
            attempt_id=None if KINDS[identifier[1]] == "done" else "a" * 32,
            visible_bytes_sha256=sha256(body(identifier).encode("utf-8")).hexdigest(),
        )
        for order, identifier in enumerate(DELIVERED if identifiers is None else identifiers)
    ]


def transcript(
    path: Path,
    *,
    received: Optional[List[str]] = None,
    texts: Optional[List[str]] = None,
    refusals: Optional[List[str]] = None,
) -> Path:
    """The agent's stream: what it called, and what the harness handed back to it.

    ``received`` is what the tool results carry, which is how a delivery the model never saw is
    written down: a message the history delivered and this list omits is one the transcript
    cannot confirm. ``texts`` writes other bytes under those identifiers, which is the other
    disconnect: the identifier arrives and what was around it does not. ``refusals`` adds an
    error result per code, which is the whole record a refusal has.
    """
    if received is None:
        received = list(DELIVERED)
    if texts is None:
        texts = [body(identifier) for identifier in received]
    lines = [
        {"type": "system", "subtype": "init", "session_id": "s"},
        _assistant([_call("mcp__curriculum__pull", {})]),
        *[_result(text) for text in texts],
        *[_refused(code) for code in refusals or []],
        _assistant(
            [_call("mcp__curriculum__api_fetch", {"attempt_id": "a" * 32, "arguments": {}})]
        ),
        _assistant(
            [
                _call("mcp__curriculum__api_search", {"attempt_id": "a" * 32, "arguments": {}}),
                _call("mcp__curriculum__done", {"attempt_id": "a" * 32, "arguments": {}}),
            ]
        ),
        _assistant([_call("Bash", {"command": "ls"})]),
        {"type": "result", "subtype": "success"},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _assistant(blocks):
    return {"type": "assistant", "message": {"content": blocks}}


def _call(name, arguments, call: str = "u"):
    return {"type": "tool_use", "id": call, "name": name, "input": arguments}


def _message(identifier: str) -> Dict[str, Any]:
    """One of the three records a harness hands back, as the wire carries it.

    They are built whole rather than abbreviated: what makes a result a task the run served is
    that it decodes as a Task, so a fixture carrying a few of a Task's fields would be asserting
    against something this cell does not count.
    """
    kind = KINDS[identifier[1]]
    common = {"message_id": identifier, "attempt_id": "a" * 32, "protocol_version": 2}
    if kind == "done":
        return {"message_id": identifier, "kind": kind, "protocol_version": 2}
    if kind == "seal_ack":
        return {
            **common,
            "kind": kind,
            "submission_digest": "d" * 64,
            "canonicalization_version": "shogym.automationbench.1",
        }
    return {**common, "kind": kind, "body": f"the {kind} body"}


def _result(text: str, *, call: str = "u", is_error: bool = False):
    block: Dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": call,
        "content": [{"type": "text", "text": text}],
    }
    if is_error:
        block["is_error"] = True
    return {"type": "user", "message": {"content": [block]}}


def _refused(code: str):
    """One refusal, as the model saw it: the canonical protocol error on the error channel."""
    return _result(
        json.dumps({"code": code, "kind": "protocol_error", "protocol_version": 2}),
        is_error=True,
    )


def test_calls_are_counted_against_the_attempt_they_named(tmp_path: Path) -> None:
    read = read_back.read_transcript(transcript(tmp_path / "transcript.jsonl"))
    # `pull` names no attempt, and asking for work is not doing it, so it is counted apart.
    assert read.per_attempt == {"a" * 32: 3}
    assert read.pulls == 1
    # The agent's own affordances are counted too: whether it worked through the tools the cell
    # gave it is a fact a rerun wants beside the score.
    assert read.unserved == 1


def test_a_transcript_that_stops_mid_line_still_counts_what_it_holds(tmp_path: Path) -> None:
    path = transcript(tmp_path / "transcript.jsonl")
    path.write_text(path.read_text() + '{"type": "assistant", "message"', encoding="utf-8")
    assert read_back.read_transcript(path).per_attempt == {"a" * 32: 3}


def test_the_table_joins_the_score_to_the_task_the_roster_names(tmp_path: Path) -> None:
    read = read_back.read_transcript(transcript(tmp_path / "transcript.jsonl"))
    rows = read_back.rows(ROSTER, [record("a" * 32, 0, score=0.75)], read)
    assert rows[0][:4] == ["12", "0", "a" * 32, "0.75"]
    assert rows[0][-1] == "3"
    assert "task" in read_back.format_table(rows).splitlines()[0]


def test_a_roster_position_the_history_answers_nothing_for_reads_as_unreached() -> None:
    # The join is written by the harness and the rows are answered by the stream, so a roster
    # naming an attempt this history does not hold is a row with nothing behind it.
    rows = read_back.rows(ROSTER, [record("a" * 32, 0)])
    assert rows[1][3] == "-" and rows[1][4] == "not reached" and rows[1][5] == "-"


def test_an_unsealed_attempt_reads_as_a_dash_and_never_as_a_zero() -> None:
    rows = read_back.rows(ROSTER, [record("a" * 32, 0, state="active", score=None)])
    assert rows[0][3] == "-"


def test_a_floored_attempt_says_which_ending_it_was() -> None:
    rows = read_back.rows(ROSTER, [floored("a" * 32, 0, "deadline")])
    assert rows[0][3] == "0" and rows[0][4] == "deadline"


def test_the_env_this_cell_serves_grades_a_spent_step_budget() -> None:
    # The cell refuses to serve the roster under any other rule, because the harness it reruns
    # graded a spent budget on the partial state and a cell that floored those would be compared
    # against numbers nobody else measured. This is the declaration that refusal reads.
    import shogym

    env = shogym.make(
        cell.ENV, config={"tasks": [{"prompt": [], "info": {}}], "max_steps": 50}
    )
    assert env.protocol_v2_horizon_ending() == GRADED_HORIZON
    assert env.describe().horizon == 52


def test_an_attempt_the_horizon_filed_is_not_read_as_one_the_agent_finished() -> None:
    # This env's horizon is a graded ending, so a run that spent all fifty-two calls is sealed
    # and scored like one that called `done`. How many of a cell's tasks the agent finished and
    # how many ran out of calls are two different numbers, and one column printing the state
    # alone would report every one of them as a workflow the agent chose to end.
    rows = read_back.rows(ROSTER, [record("a" * 32, 0, terminal_source="horizon")])
    assert rows[0][3] == "0.5" and rows[0][4] == "ack_presented (horizon)"
    assert read_back.rows(ROSTER, [record("a" * 32, 0)])[0][4] == "ack_presented"


def test_a_position_the_run_stopped_in_front_of_is_not_a_payload_it_failed_to_deliver(
    tmp_path: Path,
) -> None:
    # A generation answers with a row per declared position, so the tail of a run that stopped
    # early comes back planned with its obligation still assigned. That row was never reached,
    # and printing it as a payload this cell owed would count untouched tasks as missed ones.
    read = read_back.read_transcript(transcript(tmp_path / "transcript.jsonl"))
    rows = read_back.rows(ROSTER, [record("a" * 32, 0), never_reached("b" * 32, 1)], read)
    assert rows[1][3] == "-" and rows[1][4] == "not reached"
    assert rows[1][5] == "-" and rows[1][6] == "-"


def test_an_attempt_the_stream_floored_owes_no_payload_afterwards() -> None:
    # The ending resolves the obligation where it stands, so nothing is outstanding on this row.
    # Reading it as owed would report a permanently resolved ending as a delivery still to come.
    # The reason is abandonment rather than a spent budget: this env's horizon files rather than
    # floors, so a step cap is not an ending a cell over it can see.
    rows = read_back.rows(ROSTER, [floored("a" * 32, 0, "abandoned")])
    assert rows[0][4] == "abandoned" and rows[0][5] == "none (abandoned)"


def test_a_reached_attempt_the_agent_spent_no_call_on_counts_zero(tmp_path: Path) -> None:
    # Nought calls and no transcript to count are two different facts, and the row said neither
    # while a missing key printed as a dash.
    path = tmp_path / "transcript.jsonl"
    path.write_text(json.dumps({"type": "system", "subtype": "init"}) + "\n", encoding="utf-8")
    read = read_back.read_transcript(path)
    rows = read_back.rows(ROSTER, [record("a" * 32, 0)], read)
    assert rows[0][7] == "0"
    assert read_back.rows(ROSTER, [record("a" * 32, 0)], None)[0][7] == "-"


def test_a_row_the_transcript_confirms_is_read_apart_from_one_it_only_delivered(
    tmp_path: Path,
) -> None:
    """The disconnect: the history committed the delivery and the model's transcript has no such
    result, which is what an interrupted call leaves behind.

    Delivery is bytes handed to the transport and consumption is what the harness wrote down, so
    the table reports the two separately and names the message that only one of them holds.
    """
    seen = read_back.read_transcript(transcript(tmp_path / "seen.jsonl"))
    checked = read_back.reconcile(presentations(), seen)
    assert read_back.rows(ROSTER, [record("a" * 32, 0)], seen, checked)[0][6] == "ok"
    assert read_back.disagreements(checked, seen, counted(0), certified=True) == []

    dropped = read_back.read_transcript(
        transcript(tmp_path / "dropped.jsonl", received=DELIVERED[:2])
    )
    checked = read_back.reconcile(presentations(), dropped)
    rows = read_back.rows(ROSTER, [record("a" * 32, 0)], dropped, checked)
    # The delivery stands, because the history is what attests it, and the row says the model's
    # own record does not show the payload arriving.
    assert rows[0][5] == f"delivered ({HONEST_V1.policy_name})"
    assert rows[0][6] == "missing payload"
    assert read_back.unconfirmed(rows) == 1
    assert read_back.disagreements(checked, dropped, counted(0), certified=True) == [
        f"missing payload {message_id(0, 'c')} on attempt {'a' * 32}"
    ]


def test_a_result_carrying_the_right_identifier_and_other_bytes_is_not_the_message(
    tmp_path: Path,
) -> None:
    """The failure the identifier alone cannot see: the right names around the wrong bytes.

    A payload clipped to its first line, a result the harness rewrote and a message truncated
    mid-flight all keep the identifier they lead with. What the model was shown is the bytes, so
    the comparison is of the bytes, and a row whose identifiers all arrived under other content
    is a row the analysis may not read as feedback the agent received.
    """
    read = read_back.read_transcript(
        transcript(
            tmp_path / "rewritten.jsonl",
            texts=[body(identifier)[:-5] for identifier in DELIVERED],
        )
    )
    checked = read_back.reconcile(presentations(), read)
    rows = read_back.rows(ROSTER, [record("a" * 32, 0)], read, checked)
    assert [check.status for check in checked] == [read_back.MISMATCHED] * 3
    assert rows[0][6] == "mismatched task"
    assert read_back.unconfirmed(rows) == 1
    assert len(read_back.disagreements(checked, read, counted(0), certified=True)) == 3


def test_a_message_that_belongs_to_no_row_is_reconciled_too(tmp_path: Path) -> None:
    """Done ends the generation and sits on no attempt, so no row can report it missing.

    The same is true of a Wait and of a SealReject. They are bytes the model was shown, and a
    read that only compared the three messages a row has columns for would call a run whole while
    the record the model kept had lost the turn that ended it.
    """
    read = read_back.read_transcript(transcript(tmp_path / "no-done.jsonl"))
    ended = presentations([*DELIVERED, message_id(0, "d")])
    checked = read_back.reconcile(ended, read)
    rows = read_back.rows(ROSTER, [record("a" * 32, 0)], read, checked)
    # Every message the attempt owns arrived, so its row is clean and the run is not.
    assert rows[0][6] == "ok" and read_back.unconfirmed(rows) == 0
    assert read_back.disagreements(checked, read, counted(0), certified=True) == [
        f"missing done {message_id(0, 'd')}"
    ]


def test_a_refusal_is_read_out_of_the_transcript_and_checked_against_the_count(
    tmp_path: Path,
) -> None:
    """A refusal advances no protocol state, so the transcript is the whole record it has.

    The server keeps a count of the refusals it issued for exactly one purpose: a refusal sent
    and never delivered is then a difference between two records rather than a turn nobody holds.
    """
    read = read_back.read_transcript(
        transcript(tmp_path / "refused.jsonl", refusals=["invalid_attempt", "no_budget"])
    )
    assert read.refusals == ("invalid_attempt", "no_budget")
    checked = read_back.reconcile(presentations(), read)
    assert read_back.disagreements(checked, read, counted(2), certified=True) == []
    assert read_back.disagreements(checked, read, counted(3), certified=True) == [
        "the server refused 3 calls and this transcript holds 2"
    ]


def test_a_finished_run_with_no_refusal_count_is_a_run_that_disagrees(tmp_path: Path) -> None:
    """No count is not agreement, which is what reading the absence as nothing came to.

    The count is the one record that catches a refusal the server sent and the model never saw.
    A run with none of it fails that check by having nothing to check against, so a launch
    offering the run as finished is offering an episode whose refusals nobody can account for. A
    launch that says it stopped early may report the check as missing instead, because it is not
    presenting the run as a measurement in the first place.
    """
    read = read_back.read_transcript(transcript(tmp_path / "clean.jsonl"))
    checked = read_back.reconcile(presentations(), read)
    absent = read_back.read_refusals(tmp_path)
    assert absent.count is None and absent.absent is not None
    assert read_back.disagreements(checked, read, absent, certified=True) == [
        f"this run says it finished and {absent.absent}"
    ]
    assert read_back.disagreements(checked, read, absent, certified=False) == []


@pytest.mark.parametrize(
    "written", ["not json at all", '{"refusals": "two"}', '{"refusals": -1}', "[]"]
)
def test_a_refusal_count_that_is_not_a_count_is_no_count(tmp_path: Path, written: str) -> None:
    # A file the server half wrote, or one somebody edited, says nothing about how many refusals
    # were issued. Reading it as a number would compare the transcript against a guess.
    (tmp_path / cell.REFUSAL_FILE).write_text(written, encoding="utf-8")
    assert read_back.read_refusals(tmp_path).count is None


def test_the_count_is_on_disk_before_the_refusal_it_counts_is_answered(tmp_path: Path) -> None:
    """The fix for a server that is removed rather than asked to stop.

    Writing the count in the server's own teardown writes it at the one moment a container being
    force-removed never reaches, and sampling it on a timer only shortens that window: a refusal
    issued inside the last interval leaves a stale number, and a stale number is a valid one that
    a read cannot tell from a good one. So the transport hands the count over in the call that
    makes it, and there is no interval left to be killed inside.

    Which is what this checks, without a sleep or a teardown anywhere in it: the number is on
    disk by the time the call that made it returns.
    """
    counted = cell.refusal_sink(tmp_path)
    counted(1)
    assert read_back.read_refusals(tmp_path).count == 1
    # And killed here, with nothing running on the way out, the second one stands too.
    counted(2)
    assert read_back.read_refusals(tmp_path).count == 2


def test_the_count_is_written_whole_or_not_at_all(tmp_path: Path) -> None:
    """Half a number would read as no number, and no number fails a finished run's own check.

    The write is into a file beside the count and a move onto its name, so a read finds the old
    number or the new one and never the middle of either. Nothing is left behind to be found
    instead of it.
    """
    cell.record_refusals(tmp_path, 1)
    cell.record_refusals(tmp_path, 2)
    assert read_back.read_refusals(tmp_path).count == 2
    assert [path.name for path in sorted(tmp_path.iterdir())] == [cell.REFUSAL_FILE]


def test_a_result_the_cell_never_answered_is_not_a_message_the_cell_handed_over(
    tmp_path: Path,
) -> None:
    """The identifiers have to come back from the cell's own tools to count as delivery.

    An agent can write a message into a call of its own and read one back out of a file it wrote,
    and neither is the protocol handing it anything. So a result is read only where it answers a
    call to a served tool, by the identifier that call was made under.
    """
    lines: List[Dict[str, Any]] = []
    for identifier in DELIVERED:
        lines.append(_assistant([_call("Bash", {"command": "cat message.json"}, call="own")]))
        lines.append(_result(body(identifier), call="own"))
    path = tmp_path / "echoed.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    read = read_back.read_transcript(path)
    assert read.handed == ()
    checked = read_back.reconcile(presentations(), read)
    assert [check.status for check in checked] == [read_back.MISSING] * 3


@pytest.mark.parametrize(
    "received,refusals,status,exit_code",
    [
        # Every message arrived and the server left its count: the one shape that passes.
        (None, 0, None, 0),
        # The payload the history delivered is not in the transcript.
        (DELIVERED[:2], 0, None, 1),
        # Everything arrived, and the run this launch calls finished left no count of the
        # refusals it issued, so the check that catches an undelivered one cannot be made.
        (None, None, None, 1),
        # The same run, by a launch that says it stopped early: the check is reported missing
        # rather than counted against an episode nobody is offering as a measurement.
        (None, None, launcher.INCOMPLETE, 0),
    ],
)
def test_a_read_whose_two_records_disagree_is_a_read_that_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    received: Optional[List[str]],
    refusals: Optional[int],
    status: Optional[str],
    exit_code: int,
) -> None:
    """The exit code is the point. A run whose own transcript does not hold what the generation
    delivered is one whose analysis would be about feedback that may never have arrived, so the
    read fails rather than printing a table with a note under it. A run that kept no count of its
    refusals is the same failure with a different half missing, unless the launch has already
    said it did not finish.
    """
    from shogym.serve.protocol_v2 import reader as reader_module

    run_dir = tmp_path / "run"
    grades = run_dir / launcher.GRADES
    grades.mkdir(parents=True)
    (grades / cell.ROSTER_FILE).write_text(json.dumps(ROSTER), encoding="utf-8")
    launcher.write_run_file(run_dir, {} if status is None else {"status": status})
    if refusals is not None:
        cell.record_refusals(grades, refusals)
    transcript(run_dir / launcher.TRANSCRIPT, received=received)

    async def read(root: Path) -> reader_module.RunRecords:
        return reader_module.RunRecords(
            root=Path(root),
            workflow_id="a-generation",
            records=[record("a" * 32, 0)],
            presentations=presentations(),
        )

    monkeypatch.setattr(reader_module, "read_records", read)
    assert asyncio.run(launcher.table(run_dir)) == exit_code


def test_the_refusal_count_is_written_where_a_read_looks_for_it(tmp_path: Path) -> None:
    # The server writes it beside the join, and a directory that holds none says why rather than
    # answering with a number nobody wrote.
    assert read_back.read_refusals(tmp_path).count is None
    cell.record_refusals(tmp_path, 4)
    assert read_back.read_refusals(tmp_path) == counted(4)


def test_the_launch_is_the_command_the_earlier_cell_ran() -> None:
    """Every flag that cell passed, and nothing this one added.

    What is being compared is the serving contract, so the launch has to be the same launch: the
    same permission mode, the same output format, the same strictness about which MCP servers are
    in the run, and no deny list, because that cell's rollout arm left the agent's own tools in
    place.
    """
    argv = launcher.claude_argv(
        Path("/cfg/.mcp.json"),
        model="claude-opus-5",
        effort="xhigh",
        system_prompt="Get Better.",
        session_id="a-session",
    )
    assert argv[:3] == ["claude", "-p", launcher.KICKOFF]
    for flag, value in (
        ("--model", "claude-opus-5"),
        ("--effort", "xhigh"),
        ("--permission-mode", "bypassPermissions"),
        ("--output-format", "stream-json"),
        ("--append-system-prompt", "Get Better."),
        ("--session-id", "a-session"),
    ):
        assert argv[argv.index(flag) + 1] == value
    assert {"--strict-mcp-config", "--verbose", "--include-partial-messages"} <= set(argv)
    assert "--forward-subagent-text" in argv
    assert "--disallowedTools" not in argv and "--allowedTools" not in argv


def test_the_default_schedule_is_the_regime_the_earlier_cell_ran() -> None:
    # Read off that cell's broker rather than its documentation: the score reached the agent in
    # the result of the call that ended each task, on every task. Under this protocol that is the
    # honest payload released at the seal.
    assert launcher.CELL_ONE_SCHEDULE == "immediate"
    assert cell.release_for(launcher.CELL_ONE_SCHEDULE) is cell.SCHEDULES["immediate"]
    assert launcher.MODEL == "claude-opus-5" and launcher.EFFORT == "xhigh"


def test_the_served_tools_reach_the_model_under_the_name_that_cell_saw(tmp_path: Path) -> None:
    # A tool name is part of the prompt prefix, so the server key is pinned to the earlier cell's.
    # The prompt does not name it: the model meets the name in every tool it is offered.
    url = sandbox.gateway_url("a-server")
    config = json.loads(launcher.mcp_config(tmp_path, url=url).read_text())
    assert list(config["mcpServers"]) == [cell.SERVER] == ["curriculum"]
    served = config["mcpServers"][cell.SERVER]
    # The agent is given somewhere to connect and nothing to spawn, which is what puts the server
    # on the far side of a container rather than in the agent's own process tree.
    assert served == {"type": "http", "url": url}
    assert "command" not in served and "args" not in served and "env" not in served
    assert read_back.SERVED_PREFIX == "mcp__curriculum__"
    prompt = (Path(launcher.HERE) / "PROMPT.txt").read_text()
    assert prompt.startswith("Get Better.")
    # The prompt is the earlier cell's, the instruction and the loop only. The intended
    # difference is the loop's names: this protocol's pull and its done record. Everything
    # else the model must know, the record kinds and the attempt_id wrapper, is stated on the
    # tools themselves, so the prompt does not repeat it.
    assert "`pull`" in prompt and '{"kind": "done"}' in prompt
    assert "get_task" not in prompt and "attempt_id" not in prompt and "seal_ack" not in prompt


def test_a_row_that_owed_no_payload_is_not_read_as_one_that_missed_it() -> None:
    rows = read_back.rows(
        ROSTER,
        [
            record(
                "a" * 32,
                0,
                payload_delivered=False,
                creates_payload_obligation=False,
                payload_state=None,
                payload_policy=None,
                payload_disposition=WITHHOLD,
            )
        ],
    )
    assert rows[0][5].startswith("none")


# ----- what the launch is pinned to beside the command -----


def init_line(**overrides: Any) -> Dict[str, Any]:
    """The line Claude Code writes first, saying which build served the run and what it offered."""
    line: Dict[str, Any] = {
        "type": "system",
        "subtype": "init",
        "claude_code_version": pinned.CLI_VERSION,
        "tools": list(pinned.CLI_TOOLS),
        "agents": list(pinned.CLI_AGENTS),
        "skills": list(pinned.CLI_SKILLS),
    }
    line.update(overrides)
    return line


def test_the_agent_starts_from_the_file_that_cell_put_in_front_of_it(tmp_path: Path) -> None:
    # An empty working directory and one holding this file are two different system prompts, so
    # the file is seeded rather than assumed and its bytes are the recorded ones.
    work = tmp_path / "self"
    pinned.seed_workdir(work)
    assert (work / "CLAUDE.md").read_bytes() == b"# self\n"


def test_the_environment_each_container_gets_is_built_and_never_the_operators() -> None:
    """What either container is handed is a list, so a shell cannot reach into a run.

    The two the benchmark itself reads are the reason this matters: an operator holding either
    would serve tasks from somewhere other than the pinned source while the run's own record
    still described the standard cell.
    """
    ambient = {
        "PATH": "/usr/bin",
        "HOME": "/home/somebody",
        "CLAUDE_CODE_OAUTH_TOKEN": "a-token-nobody-should-read",
        "AUTOMATIONBENCH_SRC": "/tmp/another-benchmark",
        "SHOGYM_CACHE": "/tmp/another-cache",
        "ANTHROPIC_BASE_URL": "https://somewhere-else",
        "EDITOR": "vim",
    }
    built = pinned.agent_environment(ambient)
    assert built["CLAUDE_CODE_OAUTH_TOKEN"] == "a-token-nobody-should-read"
    assert built["IS_SANDBOX"] == "1" and built["ENABLE_TOOL_SEARCH"] == "true"
    # The image supplies the operating system, so the ambient one contributes nothing at all.
    assert set(built) == {"IS_SANDBOX", "ENABLE_TOOL_SEARCH", "CLAUDE_CODE_OAUTH_TOKEN"}
    # The server is told the run and nothing else, and the cache it reads is the one bound into
    # it rather than whichever path the launching shell was pointing at.
    served = sandbox.server_environment(tasks="cell-one:2", domain="public", schedule="immediate")
    assert served["SHOGYM_CACHE"] == sandbox.CACHE_MOUNT
    assert served["SHOGYM_CELL_RUN_DIR"] == sandbox.GRADES_MOUNT
    assert "AUTOMATIONBENCH_SRC" not in served
    assert not any(served.get(name) == value for name, value in ambient.items())
    # The record says which name carried the credential and never what it was worth.
    assert pinned.redacted(built)["CLAUDE_CODE_OAUTH_TOKEN"] == pinned.REDACTED
    assert pinned.credential_name(ambient) == "CLAUDE_CODE_OAUTH_TOKEN"
    assert pinned.credential_name({"PATH": "/usr/bin"}) is None


def test_the_recorded_image_inputs_are_the_ones_this_tree_would_build(tmp_path: Path) -> None:
    """An image id says which image answered and not what it was made of.

    The old cell recorded neither, and its image is gone, so nothing here can claim equality with
    it. What it can do is pin what this cell builds: the base by digest rather than by a tag that
    moves, the package and the registry the CLI comes from, the version, and a digest of the
    recipe. An edit to the file that nobody recorded here is a failing test rather than a rebuild
    somebody finds out about from a comparison months later.
    """
    assert sandbox.agent_build_inputs() == pinned.AGENT_IMAGE_BUILD
    assert "@sha256:" in pinned.AGENT_BASE
    assert "@sha256:" in server_inputs(tmp_path)["base"]
    pinned.check_image_build(pinned.AGENT_IMAGE_BUILD, allow_drift=False)
    with pytest.raises(ValueError, match="a-mirror"):
        pinned.check_image_build(
            {**pinned.AGENT_IMAGE_BUILD, "cli_registry": "https://a-mirror"}, allow_drift=False
        )
    pinned.check_image_build({}, allow_drift=True)


def test_the_agents_os_packages_are_a_frozen_archive_and_an_exact_version_each() -> None:
    """The shell the model reaches through Bash, pinned the way the CLI build is.

    A base pinned by digest fixes the image the build starts from and fixes nothing about what is
    installed on top of it: `apt-get install git curl jq` resolves against whatever the live
    repository is serving that day, so an image rebuilt next month held a different `git`, a
    different `curl` and a different `python3` under an identity that said nothing had moved. The
    archive is read at one immutable moment and every package names its version, and both are part
    of what the launch compares.
    """
    recipe = (Path(sandbox.HERE) / "agent.Dockerfile").read_text(encoding="utf-8")
    assert f"snapshot.debian.org/archive/debian/${{{'APT_SNAPSHOT'}}}" in recipe
    # The list lives in the pins rather than in the recipe, so what is installed and what is
    # recorded cannot be two lists that drift apart.
    assert "${APT_PACKAGES}" in recipe
    assert pinned.APT_PACKAGES and all(
        len(package.split("=")) == 2 and all(package.split("=")) for package in pinned.APT_PACKAGES
    )
    inputs = sandbox.agent_build_inputs()
    assert inputs["apt_snapshot"] == pinned.APT_SNAPSHOT
    assert inputs["apt_packages"] == " ".join(pinned.APT_PACKAGES)
    # A resolution that moved is the same refusal a moved base or a moved CLI build gets.
    with pytest.raises(ValueError, match="apt_packages"):
        pinned.check_image_build(
            {**pinned.AGENT_IMAGE_BUILD, "apt_packages": "git jq curl"}, allow_drift=False
        )
    with pytest.raises(ValueError, match="apt_snapshot"):
        pinned.check_image_build(
            {**pinned.AGENT_IMAGE_BUILD, "apt_snapshot": "20200101T000000Z"}, allow_drift=False
        )


def test_the_server_installs_the_lock_rather_than_resolving_the_ranges_beside_it(
    tmp_path: Path,
) -> None:
    """What the measurement is made of, pinned to what was chosen rather than to what is allowed.

    `pip install ".[automationbench]"` resolves the ranges in pyproject.toml live, which gave the
    server a FastMCP two major versions above the locked one, a benchmark loader and a validator
    nobody chose, and a label that said the recorded inputs matched. The lock says which
    distributions were chosen, so the image installs from it, with a hash for each, and the lock is
    one of the inputs the label is a digest of.
    """
    recipe = (Path(sandbox.HERE) / "server.Dockerfile").read_text(encoding="utf-8")
    assert "uv.lock" in recipe and "uv export --frozen" in recipe
    assert "--require-hashes" in recipe and '"uv==${UV_VERSION}"' in recipe
    assert 'pip install --no-cache-dir ".[automationbench]"' not in recipe
    assert sandbox.SERVER_LOCK == "uv.lock" and sandbox.SERVER_LOCK in sandbox.SERVER_SOURCE
    inputs = server_inputs(tmp_path)
    assert inputs["lock"] == sandbox.file_digest(sandbox.REPO / sandbox.SERVER_LOCK)
    assert inputs["uv_version"] == sandbox.UV_VERSION


def test_the_server_builds_no_wheel_and_so_resolves_no_build_backend() -> None:
    """The last live resolution in the image, which the hashed export did not cover.

    Building the project here is a PEP 517 construction: pip fetches the backend the ranges name,
    at whatever version the index is serving that day, and that backend produces the wheel holding
    the gateway and the grader. It is an input to the measurement that is in neither the hashed
    requirements nor the identity this image carries. So the project is not built at all: the
    source the image already copied is on the path, which is the same bytes the digest was taken
    over and one fewer thing resolved from outside it.
    """
    recipe = (Path(sandbox.HERE) / "server.Dockerfile").read_text(encoding="utf-8")
    assert "ENV PYTHONPATH=/app/src" in recipe
    assert "pip install --no-cache-dir --no-deps ." not in recipe
    assert "--no-build-isolation" not in recipe
    # The export is taken without the project, so the path above is what makes the project
    # importable and nothing else has to be.
    assert "--no-emit-project" in recipe


def test_each_image_is_built_with_the_pins_its_identity_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pins reach the build as arguments, so the identity a label holds is the identity the
    # build was handed rather than a description of one kept beside it.
    ran: List[List[str]] = []
    monkeypatch.setattr(
        sandbox,
        "_docker",
        lambda args, **kwargs: (
            ran.append(list(args)), subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        )[1],
    )
    sandbox.build_images(agent="an-agent", server="a-server", rebuild=True)
    agent, server = (command for command in ran if command[0] == "build")
    assert f"APT_SNAPSHOT={pinned.APT_SNAPSHOT}" in agent
    assert f"APT_PACKAGES={' '.join(pinned.APT_PACKAGES)}" in agent
    assert f"CLAUDE_CODE_VERSION={pinned.CLI_VERSION}" in agent
    assert f"UV_VERSION={sandbox.UV_VERSION}" in server


def test_an_image_whose_inputs_moved_is_rebuilt_rather_than_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tag is not the identity, so what the tag names is asked what it was built from.

    Skipping the build whenever the tag existed is what left a server image from an earlier
    checkout serving the grades under the name of this one, and an operator who forgot --build
    with no way to tell. The inputs are digested into a label at build time and read back before
    reuse, so an image that cannot say what it holds is one that gets built again.
    """
    labels = {"an-agent": sandbox.build_identity(sandbox.agent_build_inputs()), "a-server": "older"}
    ran: List[List[str]] = []

    def fake_docker(args, **kwargs):
        ran.append(list(args))
        if args[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{labels[args[-1]]}\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(sandbox, "_docker", fake_docker)
    built = sandbox.build_images(agent="an-agent", server="a-server")
    builds = [command for command in ran if command[0] == "build"]
    # The agent's image says what this tree builds, so it stands. The server's says something
    # else, so it is made again and labelled with what it was made of this time.
    assert [command[command.index("-t") + 1] for command in builds] == ["a-server"]
    identity = sandbox.build_identity(server_inputs(tmp_path))
    assert f"{sandbox.BUILD_LABEL}={identity}" in builds[0]
    assert built["an-agent"] == pinned.AGENT_IMAGE_BUILD
    # The lock is one of them, because it is what decides which distributions the image installs:
    # the ranges beside it say what is admissible and it says what was chosen, and only the second
    # is what serves the run.
    assert set(built["a-server"]) == {"base", "dockerfile", "lock", "source", "uv_version"}
    assert built["a-server"]["lock"] == sandbox.file_digest(sandbox.REPO / sandbox.SERVER_LOCK)
    # An image built before any of this existed says nothing, which is the answer no image gives.
    labels["an-agent"] = "<no value>"
    ran.clear()
    sandbox.build_images(agent="an-agent", server="a-server")
    assert [command[command.index("-t") + 1] for command in ran if command[0] == "build"] == [
        "an-agent",
        "a-server",
    ]


def test_a_generated_file_changes_neither_the_source_digest_nor_the_build_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What this repository generates is left out of what the image is built from, in both places.

    The digest and the context were two walks over one tree and only the digest was a list: docker
    was handed the directory, so a probe's own run directory, and the bytecode beside any source
    that had been imported, crossed into the image under a label saying the source in this checkout
    was what built it. The next launch computed another identity, rebuilt, and embedded the last
    run's transcripts and grades. So there is one archive now, and this is the test that it is one:
    the identity is a digest of the bytes the build is handed, and neither moves when the generated
    files appear.
    """
    monkeypatch.setattr(sandbox, "REPO", tmp_path)
    monkeypatch.setattr(sandbox, "SERVER_SOURCE", ("src", "examples/cell"))
    (tmp_path / sandbox.SERVER_LOCK).write_text("a lock\n", encoding="utf-8")
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "kept.py").write_text("kept\n", encoding="utf-8")
    (tmp_path / "examples" / "cell").mkdir(parents=True)
    (tmp_path / "examples" / "cell" / "recipe").write_text("recipe\n", encoding="utf-8")
    names = ("src", "examples/cell")
    held = ["examples/cell/recipe", "src/pkg/kept.py"]
    assert sorted(name for _, name in sandbox.source_files(names)) == held
    digest = sandbox.file_digest(
        sandbox.write_context(tmp_path / "before.tar", sandbox.source_files(names))
    )

    runs = tmp_path / "examples" / "cell" / "runs" / "probe-1"
    runs.mkdir(parents=True)
    (runs / "run.json").write_text("{}\n", encoding="utf-8")
    (runs / "stream.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "src" / "pkg" / "__pycache__" / "kept.cpython-312.pyc").write_bytes(b"\x00")
    (tmp_path / "src" / "pkg.egg-info").mkdir()
    (tmp_path / "src" / "pkg.egg-info" / "PKG-INFO").write_text("name\n", encoding="utf-8")

    assert sorted(name for _, name in sandbox.source_files(names)) == held
    archive = sandbox.server_context(tmp_path / "context.tar")
    assert sandbox.file_digest(archive) == digest
    with tarfile.open(archive) as bundle:
        assert sorted(bundle.getnames()) == held
    # And the build is handed that archive rather than a list it walks again: the bytes the
    # identity is a digest of are the bytes docker gets, so nothing saved between the two can
    # label one context as another.
    streamed: List[bytes] = []

    def fake_docker(args, **kwargs):
        if args[0] == "build":
            streamed.append(kwargs["stdin"].read())
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(sandbox, "_docker", fake_docker)
    built = sandbox.build_images(agent="an-agent", server="a-server", rebuild=True)
    assert built["a-server"]["source"] == digest
    assert f"sha256:{sha256(streamed[-1]).hexdigest()}" == digest
    # And the pattern that leaves a run directory out is the pattern that names the directory the
    # README's own probe and launch write to by default, which is how those files come to be here.
    monkeypatch.undo()
    default_runs = (sandbox.HERE / "runs" / "probe-1" / "run.json").relative_to(sandbox.REPO)
    assert sandbox.generated(default_runs.as_posix())


def test_the_benchmark_cache_is_filled_before_it_is_mounted_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first launch on a clean host, which had nowhere to put the benchmark it needed.

    The server reads the task definitions, the answers and the scoring assertions, and writes to
    none of them, so the source is mounted read only. Nothing filled it: the loader inside the
    server was what fetched the pinned source, and it met a read-only filesystem and stopped
    before the endpoint opened. The fetch is its own container now, holding that one directory
    writable and nothing else of the run, and it happens before the server is started.
    """
    commands: List[List[str]] = []

    def fake_docker(args, **kwargs):
        commands.append(["docker", *args])
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def fake_run(argv, **kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(sandbox, "_docker", fake_docker)
    monkeypatch.setattr(sandbox, "wait_for_gateway", lambda server, **kwargs: None)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    cache = tmp_path / "cache"
    domains = launcher.open_domains(
        tmp_path / "cell", tasks="cell-one:1", domain="public", schedule="immediate", cache=cache
    )
    source, target, mode = sandbox.cache_mounts(cache)[0]
    provisioned = [at for at, command in enumerate(commands) if f"{source}:{target}:rw" in command]
    served = [at for at, command in enumerate(commands) if domains.server in command]
    assert provisioned and served and provisioned[0] < served[0]
    provision = commands[provisioned[0]]
    assert provision[:3] == ["docker", "run", "--rm"]
    assert f"SHOGYM_CACHE={sandbox.CACHE_MOUNT}" in provision
    assert provision[-3:] == ["python", "-c", sandbox.PROVISION]
    assert domains.server_image in provision
    # It runs under this run's own name, so a launch that is stopped while it fetches has
    # something to take down. Nothing else on this host answers to that name.
    assert provision[provision.index("--name") + 1] == sandbox.provisioner_name(domains.server)
    # The mount the run then holds for as long as it serves is the read-only one: what the fetch
    # needed writable is a container that has already exited.
    assert mode == "ro" and f"{source}:{target}:ro" in sandbox.mount_record(domains.mounts)
    assert f"{source}:{target}:rw" not in sandbox.mount_record(domains.mounts)


def test_a_cache_that_cannot_be_filled_is_a_launch_that_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The refusal names the cache, because the alternative is the server dying on a read-only
    # filesystem inside a container whose log a launcher only saves at teardown.
    monkeypatch.setattr(
        sandbox,
        "_docker",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0 if args[0] == "info" else 1, stdout="", stderr="no space left on device"
        ),
    )
    with pytest.raises(RuntimeError, match="no space left on device"):
        launcher.open_domains(
            tmp_path / "cell",
            tasks="cell-one:1",
            domain="public",
            schedule="immediate",
            cache=tmp_path / "cache",
        )


def test_a_launch_on_an_image_built_from_other_inputs_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same refusal the CLI build gets, for the same reason: the shell, the Node and the OS
    # packages the model reaches through Bash belong to that image, so an image nobody recorded is
    # a different agent and not a different serving contract.
    ran = no_docker(monkeypatch)
    monkeypatch.setattr(
        sandbox,
        "build_images",
        lambda **kwargs: {
            kwargs["agent"]: {**pinned.AGENT_IMAGE_BUILD, "base": "node:22-bookworm-slim"},
            kwargs["server"]: {},
        },
    )
    run_dir = tmp_path / "cell"
    with pytest.raises(ValueError, match="node:22-bookworm-slim"):
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
            cache=tmp_path / "cache",
        )
    assert ran == [] and not (run_dir / launcher.RUN_FILE).exists()


def test_a_launch_the_environment_does_not_authenticate_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The agent's Claude Code home is fresh, so the environment is the only thing that can
    # authenticate it. Without a credential the launch would build both images, serve the roster
    # and record an agent that exited at once; it refuses before any of that instead.
    ran = no_docker(monkeypatch)
    for name in pinned.CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    run_dir = tmp_path / "cell"
    with pytest.raises(ValueError, match="CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY"):
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
            cache=tmp_path / "cache",
        )
    assert ran == [] and not (run_dir / launcher.RUN_FILE).exists()


def test_a_cli_that_is_not_the_recorded_build_is_refused_until_it_is_allowed() -> None:
    # A different build is a different agent: its system prompt, its built-in tools and its
    # compaction belong to the CLI rather than to the model, so drift is named or it is refused.
    pinned.check_cli_version(pinned.CLI_VERSION, allow_drift=False)
    with pytest.raises(ValueError, match="different agent"):
        pinned.check_cli_version("2.1.258", allow_drift=False)
    pinned.check_cli_version("2.1.258", allow_drift=True)


def test_the_surface_the_run_started_with_is_compared_to_the_one_that_cell_saw(
    tmp_path: Path,
) -> None:
    """The tools, subagents and skills the build offered, which no flag can pin.

    They exist only once the agent has started, so they are read off the transcript's first line
    and reported: a rerun that cannot hold them fixed can at least say how they differed.
    """
    assert pinned.surface_drift(init_line()) == {}
    drifted = pinned.surface_drift(
        init_line(
            claude_code_version="2.1.258",
            tools=[name for name in pinned.CLI_TOOLS if name != "WebSearch"] + ["Glob"],
            skills=[],
        )
    )
    assert drifted["claude_code_version"] == {
        "recorded": pinned.CLI_VERSION,
        "resolved": "2.1.258",
    }
    assert drifted["tools"] == {"missing": ["WebSearch"], "added": ["Glob"]}
    assert drifted["skills"]["missing"] == sorted(pinned.CLI_SKILLS)
    assert len(pinned.drift_report(drifted)) == 3

    # A run with no first line is a run that cannot say what served it, which is its own drift.
    empty = tmp_path / "stream.jsonl"
    empty.write_text("", encoding="utf-8")
    assert pinned.init_event(empty) is None
    assert "init" in pinned.surface_drift(None)


def server_inputs(scratch: Path) -> Dict[str, str]:
    """What the server's image is built from, read off a snapshot taken the way a build takes one.

    The source input is a digest of the archive docker is handed, so asking what an image is built
    from means writing that archive. Two writes of one tree are the same bytes, which is what lets
    a test compute the identity a build computed.
    """
    return sandbox.server_build_inputs(sandbox.server_context(scratch / "server.tar"))


#: What a working run leaves behind for the two reads that say whether it happened: the agent's
#: transcript holds the call that asked for work, and the gateway's log holds the request that
#: reached the server. A fake launch writes both, because a launch that produced neither is a
#: launch this cell refuses, which is its own test below.
SERVED_LOG = 'INFO:     172.31.0.3:56632 - "POST /mcp HTTP/1.1" 200 OK\n'


@pytest.fixture(autouse=True)
def a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs as an operator whose shell authenticates the agent.

    A launch refuses without a credential, so the one test about that refusal takes it away
    again; everything else is about what a launch does once it has one.
    """
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "a-token-nobody-should-read")


def no_docker(
    monkeypatch: pytest.MonkeyPatch,
    version: str = "2.1.220",
    *,
    pulled: bool = True,
    answered: bool = True,
    served: bool = True,
) -> List[List[str]]:
    """Answer for the daemon, and return the list every command a launch runs lands in.

    Nothing in this file starts a container. What a launch decides is which commands it would run,
    and those are what the boundary is made of, so they are collected rather than executed.

    ``pulled`` and ``served`` are the two halves of whether the run happened: what the agent's
    transcript records asking for, and what the server's log records answering. ``answered`` is
    the difference between the two halves of the first: a pull the model wrote, and a pull that
    came back with a task.
    """
    ran: List[List[str]] = []
    monkeypatch.setattr(pinned, "resolve_cli_version", lambda command=(): version)
    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(
        sandbox,
        "build_images",
        lambda **kwargs: {
            kwargs["agent"]: dict(pinned.AGENT_IMAGE_BUILD),
            kwargs["server"]: {"base": "python", "dockerfile": "sha256:1", "source": "sha256:2"},
        },
    )
    monkeypatch.setattr(sandbox, "image_id", lambda image: f"sha256:{image}")
    monkeypatch.setattr(sandbox, "create_network", lambda network: None)
    monkeypatch.setattr(sandbox, "remove_network", lambda network: None)
    monkeypatch.setattr(sandbox, "remove_container", lambda name: None)
    monkeypatch.setattr(sandbox, "wait_for_gateway", lambda server, **kwargs: None)
    monkeypatch.setattr(sandbox, "provision_source", lambda image, *, cache, name: None)
    monkeypatch.setattr(sandbox, "listening_sockets", lambda server: ["0.0.0.0:9000"])
    monkeypatch.setattr(
        sandbox, "save_logs", lambda server, path: path.write_text(SERVED_LOG if served else "")
    )

    def fake_run(argv, **kwargs):
        ran.append(list(argv))
        if "stdout" in kwargs and hasattr(kwargs["stdout"], "write"):
            lines: List[Dict[str, Any]] = [init_line(claude_code_version=version)]
            if pulled:
                lines.append(_assistant([_call(read_back.PULL_TOOL, {})]))
            if pulled and answered:
                lines.append(_result(body(message_id(0, "a"))))
            for line in lines:
                kwargs["stdout"].write(json.dumps(line).encode() + b"\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    return ran


def test_a_launch_records_the_argv_environment_and_directories_it_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole effective launch, written where a comparison can read it back.

    Nothing here starts a container: what is checked is that the command the agent would be
    started with is the one the record names, that the record carries a digest of every directory
    the agent started from, and that the surface the run reported is read back out of the
    transcript afterwards.
    """
    ran = no_docker(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "a-token-nobody-should-read")
    monkeypatch.setenv("AUTOMATIONBENCH_SRC", "/tmp/another-benchmark")
    run_dir = tmp_path / "cell"
    assert (
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
            cache=tmp_path / "cache",
        )
        == 0
    )
    text = (run_dir / launcher.RUN_FILE).read_text()
    written = json.loads(text)
    server_argv, agent_argv = ran
    assert written["argv"] == agent_argv
    assert written["cwd"] == sandbox.WORK == "/work"
    assert agent_argv[agent_argv.index("-w") + 1] == sandbox.WORK
    assert written["environment"] == pinned.redacted(pinned.agent_environment(dict(os.environ)))
    assert written["credential"] == "CLAUDE_CODE_OAUTH_TOKEN"
    # The credential is handed to docker by name, so the run's record and the host's process table
    # both hold which name authenticated the run and neither holds what it was worth.
    assert "a-token-nobody-should-read" not in text
    assert "a-token-nobody-should-read" not in " ".join(agent_argv + server_argv)
    assert "-e" in agent_argv and "CLAUDE_CODE_OAUTH_TOKEN" in agent_argv
    assert "AUTOMATIONBENCH_SRC" not in written["environment"]
    assert written["cli_version"] == written["cli_version_recorded"] == pinned.CLI_VERSION
    assert written["digests"]["work"] == pinned.digest_tree(run_dir / launcher.SELF)
    assert written["digests"]["config"] == pinned.digest_tree(run_dir / launcher.CONFIG)
    assert written["init"]["claude_code_version"] == pinned.CLI_VERSION
    assert written["drift"] == {} and written["exit_code"] == 0
    # A run that served work and was taken down afterwards says so, and says nothing else.
    assert written["status"] == launcher.COMPLETE and written["reason"] == []


def test_a_launch_on_a_build_that_is_not_the_recorded_one_starts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The refusal comes before the directories, the network and either container, so a host that
    # cannot run the comparison leaves no run directory that looks as though it did.
    ran = no_docker(monkeypatch, version="2.1.258")
    run_dir = tmp_path / "cell"
    with pytest.raises(ValueError, match="2.1.258"):
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
            cache=tmp_path / "cache",
        )
    assert ran == [] and not (run_dir / launcher.RUN_FILE).exists()


def test_a_launch_that_allows_the_drift_records_it_rather_than_hiding_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The override is what a rerun on another host needs, and what it costs is written down: the
    # run's own record names the build that served it and how its surface differed.
    no_docker(monkeypatch, version="2.1.258")
    run_dir = tmp_path / "cell"
    assert (
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
            cache=tmp_path / "cache",
            allow_cli_drift=True,
        )
        == 0
    )
    written = json.loads((run_dir / launcher.RUN_FILE).read_text())
    assert written["cli_version"] == "2.1.258"
    assert written["drift"]["claude_code_version"]["resolved"] == "2.1.258"


def test_the_agent_is_given_three_directories_and_no_fourth(tmp_path: Path) -> None:
    """The boundary this cell rests on, read off the command that makes it.

    Two of the three are the agent's own and one names the endpoint. The run directory, this
    repository and the benchmark cache are in none of them, which is the whole of the answer to
    what an agent under bypassPermissions can read about the tasks it is playing.
    """
    run_dir = tmp_path / "cell-immediate-stamp-abc123"
    mounts = sandbox.agent_mounts(run_dir, self_dir="self", home_dir="home", config_dir="cfg")
    assert sandbox.mount_record(mounts) == [
        f"{run_dir / 'self'}:/work:rw",
        f"{run_dir / 'home'}:/root/.claude:rw",
        f"{run_dir / 'cfg'}:/cfg:ro",
    ]
    argv = sandbox.agent_argv(
        image="an-image",
        name="an-agent",
        network="a-network",
        mounts=mounts,
        environment=pinned.PINNED_ENVIRONMENT,
        credential="CLAUDE_CODE_OAUTH_TOKEN",
        command=["claude", "-p", "Begin."],
    )
    bound = [argv[index + 1] for index, flag in enumerate(argv) if flag == "-v"]
    assert bound == sandbox.mount_record(mounts)
    for absent in (str(run_dir / "grades"), str(sandbox.REPO), str(sandbox.default_cache())):
        assert not any(mount.startswith(f"{absent}:") for mount in bound)
    assert argv[-3:] == ["claude", "-p", "Begin."]
    assert "-w" in argv and argv[argv.index("-w") + 1] == "/work"
    assert "IS_SANDBOX=1" in argv and "ENABLE_TOOL_SEARCH=true" in argv


def test_the_probe_and_the_launch_build_one_container_between_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the probe measures is only the agent's if the container it asks from is the agent's.

    It used to write out an environment of its own, with the two pinned variables and no
    credential, so a variable the launch added and the probe did not know about was invisible to
    the check that claimed to have asked. Both are built here now, from one function of the
    launching environment.
    """
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "a-token-nobody-should-read")
    domains = launcher.Domains(
        network="a-network",
        server="a-server",
        agent="an-agent",
        agent_image="an-image",
        server_image="a-server-image",
        mounts=[],
        environment={},
        url="http://a-server:9000/mcp",
    )
    run_dir = tmp_path / "cell"
    environment = pinned.agent_environment(os.environ)
    started = launcher.agent_command(
        domains,
        run_dir,
        command=["bash", "-c", "echo"],
        environment=environment,
        credential=pinned.credential_name(os.environ),
    )
    assert started.argv[started.argv.index("--name") + 1] == "an-agent"
    assert started.mounts == sandbox.agent_mounts(
        run_dir, self_dir=launcher.SELF, home_dir=launcher.HOME, config_dir=launcher.CONFIG
    )
    assert "IS_SANDBOX=1" in started.argv and "ENABLE_TOOL_SEARCH=true" in started.argv
    # The credential reaches the probe's container the way it reaches the agent's, by name.
    assert "CLAUDE_CODE_OAUTH_TOKEN" in started.argv
    assert "a-token-nobody-should-read" not in " ".join(started.argv)


def test_the_measurement_is_bound_into_the_server_and_published_to_nobody(tmp_path: Path) -> None:
    """The other side of the same boundary: the run, the source and the grades are all in there.

    The server publishes no port. It is reachable by container name on the private network the
    run makes for itself, so the endpoint is the only way in and it is only a way in for the
    container this run started.
    """
    run_dir = tmp_path / "cell-immediate-stamp-abc123"
    cache = tmp_path / "cache"
    mounts = sandbox.server_mounts(run_dir, grades_dir="grades", cache=cache)
    assert sandbox.mount_record(mounts) == [
        f"{run_dir / 'grades'}:/grades:rw",
        f"{cache / 'automationbench'}:/cache/automationbench:ro",
        f"{cache / 'cell-server-temporal'}:/cache/temporal:rw",
    ]
    argv = sandbox.server_argv(
        image="an-image",
        name="a-server",
        network="a-network",
        mounts=mounts,
        environment=sandbox.server_environment(
            tasks="cell-one:2", domain="public", schedule="immediate"
        ),
    )
    assert "-p" not in argv and "--publish" not in argv
    assert argv[argv.index("--network") + 1] == "a-network"
    # The path the server publishes, and not the one it answers with a redirect.
    assert sandbox.gateway_url("a-server") == "http://a-server:9000/mcp"
    # The task definitions, the answers and the scoring assertions are read and never written.
    assert f"{cache / 'automationbench'}:/cache/automationbench:ro" in argv


def test_two_runs_on_one_host_reach_each_other_by_no_name_they_can_resolve() -> None:
    first = sandbox.names("aaaaaa")
    second = sandbox.names("bbbbbb")
    assert not set(first) & set(second)
    assert sandbox.gateway_url(first[1]) != sandbox.gateway_url(second[1])


def test_the_run_says_what_boundary_it_ran_behind(tmp_path: Path, monkeypatch) -> None:
    """The mount list and the network policy, kept where a later reader can check the claim.

    Built from the same lists the two commands were built from, so a record saying the agent was
    given three directories is saying what the launch gave it.
    """
    ran = no_docker(monkeypatch)
    run_dir = tmp_path / "cell-immediate-stamp-abc123"
    launcher.launch(
        run_dir,
        tasks="cell-one:2",
        domain="public",
        schedule="immediate",
        model="claude-opus-5",
        effort="xhigh",
        cache=tmp_path / "cache",
    )
    topology = json.loads((run_dir / launcher.RUN_FILE).read_text())["topology"]
    assert topology["kind"] == "two-domain"
    assert topology["gateway_url"] == sandbox.gateway_url(topology["server"]["container"])
    assert topology["agent"]["mounts"] == [
        f"{run_dir / 'self'}:/work:rw",
        f"{run_dir / 'home'}:/root/.claude:rw",
        f"{run_dir / 'cfg'}:/cfg:ro",
    ]
    assert topology["agent"]["workdir"] == "/work"
    # The record names the credential the launch was authenticated by, and never its value.
    assert topology["agent"]["credential"] == "CLAUDE_CODE_OAUTH_TOKEN"
    assert "a-token-nobody-should-read" not in (run_dir / launcher.RUN_FILE).read_text()
    assert f"{run_dir / 'grades'}:/grades:rw" in topology["server"]["mounts"]
    assert "no port is published to the host" in topology["network_policy"]
    # Each image is named three ways, because an id says which image on this host answered and
    # only the inputs can be compared with a rerun somewhere else.
    assert topology["agent"]["build"] == dict(pinned.AGENT_IMAGE_BUILD)
    assert topology["server"]["build"]["source"] == "sha256:2"
    # The agent's command carries the endpoint and never a command to spawn a server with.
    config = json.loads((run_dir / launcher.CONFIG / ".mcp.json").read_text())
    assert config["mcpServers"][cell.SERVER]["url"] == topology["gateway_url"]
    assert ran[1][ran[1].index("--mcp-config") + 1] == "/cfg/.mcp.json"


def test_a_launch_whose_agent_never_reached_the_endpoint_is_not_a_successful_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure the corrected endpoint had already caused once, told this time.

    A CLI that cannot negotiate with the server prints its opening line and exits nought, and that
    line reports the MCP server as pending whether or not the connection ever came, exactly as the
    recorded cell's own does. So neither the exit code nor the init line can answer whether the
    run happened, and a pilot reading either would file an empty cell beside a full one.
    """
    no_docker(monkeypatch, pulled=False, served=False)
    run_dir = tmp_path / "cell"
    assert (
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
            cache=tmp_path / "cache",
        )
        == 1
    )
    written = json.loads((run_dir / launcher.RUN_FILE).read_text())
    # Everything that used to say the run succeeded still says it, which is the point.
    assert written["exit_code"] == 0 and written["drift"] == {}
    assert written["init"]["claude_code_version"] == pinned.CLI_VERSION
    assert written["status"] == launcher.INCOMPLETE
    assert any("no pull came back with a task" in reason for reason in written["reason"])
    assert any("gateway answered no request" in reason for reason in written["reason"])


def test_a_pull_that_came_back_with_nothing_is_not_a_run_that_served_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The half of the first failure that survived its first fix.

    A `pull` in the transcript is a call the model wrote, not work it received. The CLI can
    negotiate far enough to list the tools, write the call, have the request refused or redirected
    or answered with a protocol error, and exit nought having been handed no task at all, while an
    initialization that arrived before any of that answers the server's side on its own. So the
    call counts for nothing here and the result is what is read.
    """
    no_docker(monkeypatch, answered=False)
    run_dir = tmp_path / "cell"
    assert (
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
            cache=tmp_path / "cache",
        )
        == 1
    )
    written = json.loads((run_dir / launcher.RUN_FILE).read_text())
    assert written["exit_code"] == 0 and written["status"] == launcher.INCOMPLETE
    assert written["reason"] == ["no pull came back with a task, so this run served nothing"]
    read = read_back.read_transcript(run_dir / launcher.TRANSCRIPT)
    assert read.pulls == 1 and read.tasks == 0


def test_a_pull_answered_with_an_error_or_a_record_that_is_not_a_task_served_nothing(
    tmp_path: Path,
) -> None:
    """What the transcript has to hold, asked of the four ways a pull comes back empty.

    The decoder is the protocol's own, so a task is what the wire calls a task rather than
    anything carrying the word. A refusal the harness marked as an error, a body that is not a
    record, a record of another kind and a task missing a field are each a pull that served no
    work, and a run made of them is a cell with nothing in it.
    """
    path = tmp_path / "stream.jsonl"

    def answered(result: Optional[Dict[str, Any]]) -> read_back.Transcript:
        lines: List[Dict[str, Any]] = [
            init_line(),
            _assistant([_call(read_back.PULL_TOOL, {}, call="p1")]),
        ]
        if result is not None:
            lines.append(result)
        path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        return read_back.read_transcript(path)

    def answer(text: str, *, error: bool = False) -> Dict[str, Any]:
        block: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": "p1",
            "content": [{"type": "text", "text": text}],
        }
        if error:
            block["is_error"] = True
        return {"type": "user", "message": {"content": [block]}}

    task = json.dumps(_message(message_id(0, "a")))
    assert answered(None).tasks == 0
    assert answered(answer(task, error=True)).tasks == 0
    assert answered(answer("MCP error -32603: the gateway refused")).tasks == 0
    assert answered(answer(json.dumps(_message(message_id(0, "c"))))).tasks == 0
    thin = json.loads(task)
    del thin["body"]
    assert answered(answer(json.dumps(thin))).tasks == 0
    # And the one shape that is a task: the canonical record, in the result of the call that
    # asked for it.
    served = answered(answer(task))
    assert served.pulls == 1 and served.tasks == 1


def test_the_gateway_log_answers_for_the_endpoint_and_for_requests_it_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The server's side of the run, read exactly rather than by prefix.

    Any line beginning `/mcp` used to count, whatever the path went on to say and whatever the
    server answered. So the redirect this cell's URL was corrected away from, a path that is not
    the endpoint, and a request the gateway refused each said the measurement had been reached.
    """
    assert sandbox.served_requests(SERVED_LOG) == 1
    for line in (
        'INFO:     172.31.0.3:1 - "POST /mcp/ HTTP/1.1" 307 Temporary Redirect\n',
        'INFO:     172.31.0.3:1 - "POST /mcpx HTTP/1.1" 404 Not Found\n',
        'INFO:     172.31.0.3:1 - "POST /mcp HTTP/1.1" 404 Not Found\n',
        'INFO:     172.31.0.3:1 - "POST /mcp HTTP/1.1" 403 Forbidden\n',
        'INFO:     172.31.0.3:1 - "POST /mcp HTTP/1.1" 500 Internal Server Error\n',
        "INFO:     Uvicorn running on http://0.0.0.0:9000\n",
    ):
        assert sandbox.served_requests(line) == 0
    # A run whose only request was refused is a run nothing reached, however the transcript reads.
    no_docker(monkeypatch)
    monkeypatch.setattr(
        sandbox,
        "save_logs",
        lambda server, path: path.write_text(
            'INFO:     172.31.0.3:1 - "POST /mcp HTTP/1.1" 404 Not Found\n'
        ),
    )
    run_dir = tmp_path / "cell"
    assert (
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
            cache=tmp_path / "cache",
        )
        == 1
    )
    written = json.loads((run_dir / launcher.RUN_FILE).read_text())
    assert written["reason"] == [
        "the gateway answered no request, so nothing reached the measurement"
    ]


def test_a_launch_the_server_never_heard_from_says_so_even_with_a_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The two sides are asked apart. A transcript is the agent's own record, so a run whose agent
    # recorded a call the gateway never answered is still a run that served nothing.
    no_docker(monkeypatch, served=False)
    run_dir = tmp_path / "cell"
    assert (
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
            cache=tmp_path / "cache",
        )
        == 1
    )
    written = json.loads((run_dir / launcher.RUN_FILE).read_text())
    assert written["reason"] == ["the gateway answered no request, so nothing reached the "
                                "measurement"]
    # The gateway's own access log is what answers this, and its readiness check leaves no line
    # there: a connection that says nothing is not a request the server answered.
    assert sandbox.served_requests("") == 0


def test_a_launch_stopped_by_a_signal_takes_both_domains_down_and_records_the_ending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary termination, which is how a scheduler ends a job that ran out of time.

    A signal used to end the launcher without unwinding it, so the agent went on calling the
    server and writing into the two directories the run is measuring while the launcher watching
    them was gone. What the run came to has to be the launcher's last act rather than something
    lost with it, so the signal is handled, both containers are stopped by name, the network comes
    down, and the record says which signal it was.
    """
    ran = no_docker(monkeypatch)
    taken: List[str] = []
    monkeypatch.setattr(sandbox, "remove_container", lambda name: taken.append(name))
    monkeypatch.setattr(sandbox, "remove_network", lambda network: taken.append(network))

    def signalled(argv, **kwargs):
        ran.append(list(argv))
        if "stdout" in kwargs and hasattr(kwargs["stdout"], "write"):
            # The agent is running, and the launcher is asked to stop.
            os.kill(os.getpid(), signal.SIGTERM)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(launcher.subprocess, "run", signalled)
    run_dir = tmp_path / "cell-immediate-stamp-abc123"
    before = signal.getsignal(signal.SIGTERM)
    code = launcher.launch(
        run_dir,
        tasks="cell-one:2",
        domain="public",
        schedule="immediate",
        model="claude-opus-5",
        effort="xhigh",
        cache=tmp_path / "cache",
    )
    assert code == 128 + int(signal.SIGTERM)
    network, server, agent = sandbox.names("abc123")
    # The fetch's name is taken first, because a launch owns that container before it owns a
    # network or a server, and every one of them is gone when the launcher is.
    assert taken == [sandbox.provisioner_name(server), agent, server, network]
    written = json.loads((run_dir / launcher.RUN_FILE).read_text())
    assert written["status"] == launcher.INCOMPLETE
    assert written["reason"] == ["the launcher was stopped by SIGTERM"]
    assert written["exit_code"] is None
    # And the handler is the launch's own for as long as the launch owns containers, not
    # something left behind for whatever the process does next.
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_launch_stopped_while_the_source_is_fetched_takes_the_fetch_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same termination, at the one moment the launch owns a container and nothing else.

    A signal ends the docker client and not the container it started, so a fetch interrupted here
    went on holding and writing the shared cache with the launch that started it gone. There was
    no run directory, no network and no server yet, so the teardown that takes a run down had
    nothing to take: the fetch was unnamed. It runs under this run's own name now and comes down
    however the fetch ends, which is what this asks for at the moment it is running.
    """
    no_docker(monkeypatch)
    taken: List[str] = []
    monkeypatch.setattr(sandbox, "remove_container", lambda name: taken.append(name))
    monkeypatch.setattr(sandbox, "remove_network", lambda network: taken.append(network))

    def fetching(image, *, cache, name):
        # The fetch is running, and the launcher is asked to stop.
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(sandbox, "provision_source", fetching)
    run_dir = tmp_path / "cell-immediate-stamp-abc123"
    code = launcher.launch(
        run_dir,
        tasks="cell-one:2",
        domain="public",
        schedule="immediate",
        model="claude-opus-5",
        effort="xhigh",
        cache=tmp_path / "cache",
    )
    assert code == 128 + int(signal.SIGTERM)
    _, server, _ = sandbox.names("abc123")
    # The fetch, and nothing else, because nothing else had been started yet.
    assert taken == [sandbox.provisioner_name(server)]
    written = json.loads((run_dir / launcher.RUN_FILE).read_text())
    assert written["status"] == launcher.INCOMPLETE
    assert written["reason"] == ["the launcher was stopped by SIGTERM"]


def test_a_teardown_attempts_every_target_before_it_reports_any_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Callers rely on this running to the end, so one container that will not die, or a log write
    # that fails, must not be what stops the network and the other container coming down.
    attempted: List[str] = []

    def refuse(server, path):
        attempted.append("log")
        raise OSError("read-only file system")

    def remove(name: str) -> Optional[str]:
        attempted.append(name)
        return f"{name} is still here" if "agent" in name else None

    monkeypatch.setattr(sandbox, "save_logs", refuse)
    monkeypatch.setattr(sandbox, "remove_container", remove)
    monkeypatch.setattr(sandbox, "remove_network", lambda network: attempted.append(network))
    domains = launcher.Domains(
        network="a-network",
        server="a-server",
        agent="an-agent",
        agent_image="an-image",
        server_image="a-server-image",
        mounts=[],
        environment={},
        url="http://a-server:9000/mcp",
    )
    failures = launcher.close_domains(domains, log=tmp_path / "server.log")
    assert attempted == ["log", "an-agent", "a-server", "a-network"]
    assert failures == ["the server's log could not be saved: read-only file system",
                        "an-agent is still here"]


def test_a_probe_removes_the_container_it_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe's container is the agent's, so it is one the teardown knows the name of.

    It used to run under a name of its own and the teardown removed the name it did not use, which
    left the probe's container running and its network up behind an interrupted diagnostic.
    """
    ran = no_docker(monkeypatch)
    taken: List[str] = []
    monkeypatch.setattr(sandbox, "remove_container", lambda name: taken.append(name))
    monkeypatch.setattr(sandbox, "remove_network", lambda network: taken.append(network))

    def answered(argv, **kwargs):
        ran.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok    something\nfailed=0\n", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", answered)
    run_dir = tmp_path / "probe-immediate-stamp-abc123"
    assert (
        launcher.probe(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            cache=tmp_path / "cache",
        )
        == 0
    )
    network, server, agent = sandbox.names("abc123")
    started = ran[-1][ran[-1].index("--name") + 1]
    assert started == agent
    assert taken == [sandbox.provisioner_name(server), agent, server, network]


def test_only_the_gateway_is_bound_where_another_container_could_reach_it() -> None:
    """The network half of the boundary, read the way the run reads it off the server.

    The rows are the kernel's own table, hexadecimal and least significant byte first. What the
    run asks of them is which listeners are on an address another container could route to, and
    the answer has to be the endpoint alone: the durable service holding the history is one of the
    others, and it is on that container's loopback.
    """
    table = "\n".join(
        [
            "  sl  local_address rem_address   st",
            "   0: 00000000:2328 00000000:0000 0A",  # 0.0.0.0:9000, the gateway
            "   1: 0100007F:1C39 00000000:0000 0A",  # 127.0.0.1:7225, the durable service
            "   2: 0B00007F:5F41 00000000:0000 01",  # not listening, so not a way in
        ]
    )
    listeners = sandbox.parse_listeners(table)
    assert listeners == ["0.0.0.0:9000", "127.0.0.1:7225"]
    assert sandbox.unexpected_listeners(listeners) == []


def test_a_port_that_merely_ends_in_the_gateways_digits_is_not_the_gateway() -> None:
    """The address and the port are compared apart, and each of them exactly.

    A suffix test waved through every port ending in the endpoint's digits, so a service on 19000
    read as the gateway. An address the parse cannot place is not loopback either: a listener
    nobody can locate is not one anybody has shown to be out of reach.
    """
    assert sandbox.unexpected_listeners(
        ["0.0.0.0:9000", "0.0.0.0:19000", "0.0.0.0:29000", "127.0.0.1:42233", "::1:7233"]
    ) == ["0.0.0.0:19000", "0.0.0.0:29000"]
    assert sandbox.unexpected_listeners(["somewhere:9001"]) == ["somewhere:9001"]


def test_the_agents_own_network_namespace_is_measured_and_not_assumed() -> None:
    """The claim the flag was standing in for, made into a question the probe can ask.

    A container started to share the server's network stack passes every other check here: the
    same three mounts, the same environment, the same resolver, the same 404s. What it also has is
    the server's loopback, which is where the durable service holding the history is bound. So the
    namespace is read on both sides and compared, and the addresses the server keeps to itself are
    the ones the agent's container is asked about, each at the address it was bound to.
    """
    assert "readlink /proc/1/ns/net" in sandbox.PROBE_SCRIPT
    assert sandbox.loopback_listeners(
        ["0.0.0.0:9000", "127.0.0.1:7233", "127.0.0.11:43711", "::1:8233", "somewhere:9001"]
    ) == ["127.0.0.1:7233", "127.0.0.11:43711", "::1:8233"]
    command = sandbox.probe_command(
        run_dir=Path("/runs/cell"),
        cache=Path("/cache"),
        server="a-server",
        environment=["IS_SANDBOX"],
        server_namespace="net:[4026532296]",
        server_loopback=["127.0.0.1:7233", "::1:8233"],
    )
    assert "net:[4026532296]" in command and "127.0.0.1:7233 ::1:8233" in command
    # Told neither, it fails the checks it cannot make rather than passing them: a namespace
    # nobody could ask about is not one anybody has shown to be separate.
    told_nothing = sandbox.probe_command(
        run_dir=Path("/runs/cell"), cache=Path("/cache"), server="a-server", environment=[]
    )
    assert told_nothing[-2:] == ["", ""]


def test_a_namespace_that_could_not_be_read_is_refused_like_a_listener_that_could_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Answering with a name that matches nothing would pass the comparison whenever the asking
    # broke, which is the failure the listener read already refuses.
    answers: List[subprocess.CompletedProcess] = []
    monkeypatch.setattr(sandbox, "_docker", lambda args, **kwargs: answers.pop(0))
    answers.append(subprocess.CompletedProcess([], 1, stdout="", stderr="No such container"))
    with pytest.raises(ValueError, match="unknown"):
        sandbox.network_namespace("a-server")
    answers.append(subprocess.CompletedProcess([], 0, stdout="\n", stderr=""))
    with pytest.raises(ValueError, match="unknown"):
        sandbox.network_namespace("a-server")
    answers.append(subprocess.CompletedProcess([], 0, stdout="net:[4026532296]\n", stderr=""))
    assert sandbox.network_namespace("a-server") == "net:[4026532296]"


def test_listeners_that_could_not_be_read_are_refused_and_never_read_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty list is what a container listening on nothing looks like, so answering one for a
    # container nobody could ask is a check that passes whenever the asking breaks.
    monkeypatch.setattr(
        sandbox,
        "_docker",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 1, stdout="", stderr="Error: No such container"
        ),
    )
    with pytest.raises(ValueError, match="unknown"):
        sandbox.listening_sockets("a-server")


def test_the_probe_reads_its_own_verdict() -> None:
    # The probe is a script in another container, so what it found comes back as text. A run that
    # printed no verdict is a run that proved nothing, which is not the same as one that passed.
    assert sandbox.read_probe("ok    something\nfailed=0\n") == 0
    assert sandbox.read_probe("FAIL  something\nfailed=1\n") == 1
    with pytest.raises(ValueError, match="no verdict"):
        sandbox.read_probe("ok    something\n")
    # Every host path the probe checks for is passed to it, so it is asking about this run.
    command = sandbox.probe_command(
        run_dir=Path("/runs/cell"),
        cache=Path("/cache"),
        server="a-server",
        environment=["IS_SANDBOX", "ENABLE_TOOL_SEARCH"],
    )
    assert "/runs/cell" in command and "/cache" in command
    assert str(sandbox.REPO) in command
    assert sandbox.gateway_url("a-server") in command


def test_what_the_probe_is_told_reaches_it_as_arguments_and_not_as_environment() -> None:
    """The probe asks what its own process was handed, so it cannot be told through that.

    Six variables of its own were how it used to carry the paths it was checking for, which left
    it measuring an environment no agent is ever launched with. They are arguments now, and the
    names the launch would set are one more of them, so what the check compares is the launch's
    own list rather than a copy of it kept here.
    """
    command = sandbox.probe_command(
        run_dir=Path("/runs/cell"),
        cache=Path("/cache"),
        server="a-server",
        environment=["CLAUDE_CODE_OAUTH_TOKEN", "ENABLE_TOOL_SEARCH", "IS_SANDBOX"],
    )
    assert command[:2] == ["bash", "-c"] and command[2] == sandbox.PROBE_SCRIPT
    assert not any("=" in part for part in command[3:])
    assert "CLAUDE_CODE_OAUTH_TOKEN ENABLE_TOOL_SEARCH IS_SANDBOX" in command
    assert " ".join(sandbox.IMAGE_ENVIRONMENT) in command


def test_the_probe_asks_about_every_local_surface_it_claims_to_have_measured() -> None:
    """The verdict names the boundary, so the script has to have asked about all of it.

    Reachability by URL was the whole of what it asked. A container also has a config directory it
    could rewrite, an environment somebody could have added to, a process table, a privilege and
    seccomp state, a container runtime socket, a resolver, and a metadata service holding the
    host's own credentials, and each of those is a way to the grades that answers no curl.
    """
    for surface in (
        "/cfg/.mcp.json",
        "/proc/1/environ",
        "/proc/[0-9]*",
        "CapBnd",
        "Seccomp",
        "docker.sock",
        "2375",
        "169.254.169.254",
        "metadata.google.internal",
        "getent hosts",
    ):
        assert surface in sandbox.PROBE_SCRIPT
    # And it says what it did not measure: general egress is retained rather than absent, so a
    # host the agent can reach is reported as the egress it is and never counted as isolation.
    assert "general egress is retained" in sandbox.PROBE_SCRIPT
    if shutil.which("bash"):
        syntax = subprocess.run(
            ["bash", "-n", "-c", sandbox.PROBE_SCRIPT], capture_output=True, text=True
        )
        assert syntax.returncode == 0, syntax.stderr


def _images_are_here() -> bool:
    """Whether this host already holds both images, whatever they were built from.

    Building one to make a point about a network or a cache is work these checks do not own, so
    without them they are skipped. What each image was built from does not bear on either, which
    is why only their presence is asked about.
    """
    if not sandbox.docker_available():
        return False
    images = (f"{sandbox.AGENT_IMAGE}:{pinned.CLI_VERSION}", f"{sandbox.SERVER_IMAGE}:latest")
    return all(sandbox.image_id(image) is not None for image in images)


def _domains_can_be_stood_up() -> bool:
    """Whether this host holds both images and the benchmark source the server reads."""
    return _images_are_here() and (sandbox.default_cache() / "automationbench").is_dir()


@pytest.mark.skipif(
    not _images_are_here(),
    reason="the live cold-cache check needs both images built here",
)
def test_a_launch_with_an_empty_cache_fills_it_and_reaches_the_gateway(tmp_path: Path) -> None:
    """The first launch on a clean host, run as the thing it used to be.

    A cache nobody has filled is the ordinary state of a pilot machine and of an operator who
    named a cache of their own, and it was the one state this cell could not launch from: the
    source is mounted read only, so the loader inside the server raised on a read-only filesystem
    and the endpoint never opened. What is checked here is the whole of that path, with a cache
    that starts empty: the pinned source is in it afterwards, and the gateway is listening.

    It is slower than the rest of this file by a lot, because an empty cache is also an empty
    durable-service cache and both are fetched here.
    """
    cache = tmp_path / "cache"
    run_dir = launcher.new_run_dir(tmp_path, schedule="immediate", prefix="probe")
    domains = launcher.open_domains(
        run_dir, tasks="cell-one:1", domain="public", schedule="immediate", cache=cache
    )
    try:
        # The source, under the commit the adapter pins, in the directory the server has read only.
        assert list((cache / "automationbench").glob("*/automationbench/__init__.py"))
        assert f"0.0.0.0:{sandbox.SERVER_PORT}" in sandbox.listening_sockets(domains.server)
    finally:
        assert launcher.close_domains(domains) == []


@pytest.mark.skipif(
    not _domains_can_be_stood_up(),
    reason="the live boundary check needs both images built here and the benchmark source cached",
)
def test_a_probe_sharing_the_servers_network_stack_is_a_probe_that_fails(tmp_path: Path) -> None:
    """The false negative the probe used to have, run as the thing it would have missed.

    Everything else the probe asks stays true of a container joined to the server's own network
    stack: the same three mounts, the same environment, the same resolver, the same 404s at the
    gateway, the same refusals on the alternate ports. What such a container also has is the
    server's loopback, where the durable service holding the history is bound. So the run below is
    the real launch with one thing changed, and the probe has to come back non-zero: the namespace
    it reports is the server's, and the addresses the server keeps to itself answer inside it.
    """
    cache = sandbox.default_cache()
    run_dir = launcher.new_run_dir(tmp_path, schedule="immediate", prefix="probe")
    pinned.seed_workdir(run_dir / launcher.SELF)
    (run_dir / launcher.HOME).mkdir(parents=True, exist_ok=True)
    domains = launcher.open_domains(
        run_dir, tasks="cell-one:2", domain="public", schedule="immediate", cache=cache
    )
    try:
        launcher.mcp_config(run_dir, url=domains.url)
        environment = pinned.agent_environment(os.environ)
        started = launcher.agent_command(
            # The one change: the agent's container joins the server's network stack instead of
            # the run's own network. Nothing else about the launch moves.
            domains._replace(network=f"container:{domains.server}"),
            run_dir,
            command=sandbox.probe_command(
                run_dir=run_dir,
                cache=cache,
                server=domains.server,
                environment=sorted(environment),
                server_namespace=sandbox.network_namespace(domains.server),
                server_loopback=sandbox.loopback_listeners(
                    sandbox.listening_sockets(domains.server)
                ),
            ),
            environment=environment,
            credential=pinned.credential_name(os.environ),
        )
        done = subprocess.run(started.argv, capture_output=True, text=True, check=False)
    finally:
        assert launcher.close_domains(domains) == []
    failures = [line for line in done.stdout.splitlines() if line.startswith("FAIL")]
    assert sandbox.read_probe(done.stdout) == len(failures) > 0, done.stdout
    assert any("network namespace" in line for line in failures), done.stdout
    assert any("is this container's loopback" in line for line in failures), done.stdout
