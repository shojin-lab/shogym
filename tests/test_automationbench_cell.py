"""Guard tests for the AutomationBench cell runner: the roster it composes and the table it reads.

Two halves, and neither needs a model or a durable service. The composition is a pure function of
the roster and the schedule, so what is checked there is that a schedule name decides what the
agent is told and that nothing else does. The table is a pure function of three files, so what is
checked there is the join: a score belongs to the task the roster says it does, a position nobody
reached says so, and a tool call is counted against the attempt it named.

Nothing here spawns the ``claude`` CLI or spends a token, and nothing here provisions the
AutomationBench upstream source: the grade identity this cell composes over is a declaration and
importing it reaches no upstream package.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

from examples.automationbench_cell import cell as launcher  # noqa: E402
from examples.automationbench_cell import pinned  # noqa: E402
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
    return json.dumps(
        {"kind": KINDS[identifier[1]], "message_id": identifier, "protocol_version": 2}
    )


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


def _call(name, arguments, *, call: str = "u"):
    return {"type": "tool_use", "id": call, "name": name, "input": arguments}


def _result(text: str, *, call: str = "u", is_error: bool = False):
    result: Dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": call,
        "content": [{"type": "text", "text": text}],
    }
    if is_error:
        result["is_error"] = True
    return {"type": "user", "message": {"content": [result]}}


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
    config = json.loads(
        launcher.mcp_config(
            tmp_path, tasks="cell-one:2", domain="public", schedule="immediate"
        ).read_text()
    )
    assert list(config["mcpServers"]) == [cell.SERVER] == ["curriculum"]
    served = config["mcpServers"][cell.SERVER]
    assert served["args"][-1].endswith("serve.py")
    assert served["env"]["SHOGYM_CELL_SCHEDULE"] == "immediate"
    assert served["env"]["SHOGYM_CELL_RUN_DIR"].endswith(launcher.GRADES)
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


def test_the_child_environment_is_built_and_never_the_operators(tmp_path: Path) -> None:
    """What the agent and its server are handed is a list, so a shell cannot reach into a run.

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
    built = pinned.child_environment(ambient, config_dir=tmp_path / "home")
    assert built["PATH"] == "/usr/bin" and built["HOME"] == "/home/somebody"
    assert built["CLAUDE_CODE_OAUTH_TOKEN"] == "a-token-nobody-should-read"
    assert built["IS_SANDBOX"] == "1" and built["ENABLE_TOOL_SEARCH"] == "true"
    assert built["CLAUDE_CONFIG_DIR"] == str(tmp_path / "home")
    for name in ("AUTOMATIONBENCH_SRC", "SHOGYM_CACHE", "ANTHROPIC_BASE_URL", "EDITOR"):
        assert name not in built
    # The record says which name carried the credential and never what it was worth.
    assert pinned.redacted(built)["CLAUDE_CODE_OAUTH_TOKEN"] == pinned.REDACTED
    assert pinned.credential_name(ambient) == "CLAUDE_CODE_OAUTH_TOKEN"
    assert pinned.credential_name({"PATH": "/usr/bin"}) is None


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


def test_a_launch_records_the_argv_environment_and_directories_it_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole effective launch, written where a comparison can read it back.

    Nothing here spawns the CLI: what is checked is that the process would be given the argv, the
    working directory and the environment the record names, that the record carries a digest of
    every directory the agent started from, and that the surface the run reported is read back
    out of the transcript afterwards.
    """
    monkeypatch.setattr(pinned, "resolve_cli_version", lambda executable="claude": "2.1.220")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "a-token-nobody-should-read")
    monkeypatch.setenv("AUTOMATIONBENCH_SRC", "/tmp/another-benchmark")
    spawned: Dict[str, Any] = {}

    def fake_run(argv, **kwargs):
        spawned.update(argv=argv, cwd=kwargs["cwd"], env=kwargs["env"])
        kwargs["stdout"].write(json.dumps(init_line()).encode("utf-8") + b"\n")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    run_dir = tmp_path / "cell"
    assert (
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
        )
        == 0
    )
    text = (run_dir / launcher.RUN_FILE).read_text()
    written = json.loads(text)
    assert written["argv"] == spawned["argv"] == list(spawned["argv"])
    assert written["cwd"] == str(spawned["cwd"]) == str(run_dir / launcher.SELF)
    assert written["environment"] == pinned.redacted(spawned["env"])
    assert written["credential"] == "CLAUDE_CODE_OAUTH_TOKEN"
    assert "a-token-nobody-should-read" not in text
    assert "AUTOMATIONBENCH_SRC" not in written["environment"]
    assert written["cli_version"] == written["cli_version_recorded"] == pinned.CLI_VERSION
    assert written["digests"]["work"] == pinned.digest_tree(run_dir / launcher.SELF)
    assert written["digests"]["config"] == pinned.digest_tree(run_dir / launcher.CONFIG)
    assert written["init"]["claude_code_version"] == pinned.CLI_VERSION
    assert written["drift"] == {} and written["exit_code"] == 0


def test_a_launch_on_a_build_that_is_not_the_recorded_one_spawns_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The refusal comes before the directories, the config and the process, so a host that cannot
    # run the comparison leaves no run directory that looks as though it did.
    monkeypatch.setattr(pinned, "resolve_cli_version", lambda executable="claude": "2.1.258")

    def never(*args: Any, **kwargs: Any):
        raise AssertionError("a launch that was refused must not spawn the agent")

    monkeypatch.setattr(launcher.subprocess, "run", never)
    run_dir = tmp_path / "cell"
    with pytest.raises(ValueError, match="2.1.258"):
        launcher.launch(
            run_dir,
            tasks="cell-one:2",
            domain="public",
            schedule="immediate",
            model="claude-opus-5",
            effort="xhigh",
        )
    assert not (run_dir / launcher.RUN_FILE).exists()


def test_a_launch_that_allows_the_drift_records_it_rather_than_hiding_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The override is what a rerun on another host needs, and what it costs is written down: the
    # run's own record names the build that served it and how its surface differed.
    monkeypatch.setattr(pinned, "resolve_cli_version", lambda executable="claude": "2.1.258")
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda argv, **kwargs: (
            kwargs["stdout"].write(
                json.dumps(init_line(claude_code_version="2.1.258")).encode("utf-8") + b"\n"
            ),
            subprocess.CompletedProcess(argv, 0),
        )[1],
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
            allow_cli_drift=True,
        )
        == 0
    )
    written = json.loads((run_dir / launcher.RUN_FILE).read_text())
    assert written["cli_version"] == "2.1.258"
    assert written["drift"]["claude_code_version"]["resolved"] == "2.1.258"
