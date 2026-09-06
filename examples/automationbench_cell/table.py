"""What the cell scored, joined back to the benchmark tasks it was over.

Three things have to meet for a cell to be compared with another one. The stream's history says
what each attempt filed and what it scored; the run directory's roster says which AutomationBench
task each attempt was; and the harness transcript says how much work the agent did on it. None of
them can answer for the others: the history is keyed by attempt because nothing on the wire names
a task, the roster carries no score, and the transcript is the only place a tool call exists at
all.

So this reads all three and prints one row per roster position. The score column is the one the
history answered with, never a number recomputed here, and an attempt nobody sealed reads as a
dash rather than as a zero.

Delivery and consumption are two columns rather than one. What the history commits is that the
exact bytes were handed to the transport carrying them, which is all a server can attest; whether
the model received them is a fact about the harness, and the transcript is where it is written.
The two agree in every ordinary run, and where they disagree the run has lost a message the
analysis would otherwise assume the agent read. So the payload column reports the delivery the
history committed, and the column beside it reports what the transcript shows arriving.

That second column is a comparison of bytes and not of names. The history says what every
message it presented was and what its bytes hashed to; the transcript says which served call
each result answered and what the model was shown in it; and the two are walked together in the
order they happened. A result carrying the right identifier and other bytes is a message the
model did not receive, and so is a message that never arrived at all, and the two are named
apart. Refusals are read the same way: they advance no protocol state and so exist only as text
the model saw, and the count the server kept is what says whether the transcript holds them all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

from examples.automationbench_cell.serve import REFUSAL_FILE, ROSTER_FILE, SERVED_PREFIX
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.kernel.messages import AttemptRecord, PresentedMessage
from shogym.serve.protocol_v2.records import Done, Task

#: The served tool that asks for work. It names no attempt, and it is counted on its own, because
#: how often an agent asked for work is a different fact from how much work it did. The tool that
#: says how much work is left names no attempt either, and asking is neither a pull nor work on a
#: task, so it is counted as neither; the answer it comes back with is a presented message like
#: any other and is reconciled as one.
PULL_TOOL = f"{SERVED_PREFIX}pull"

#: How a message names itself inside a result the agent was handed. Every record this protocol
#: presents carries its own identifier in the bytes, which is what tells a result holding some
#: of a message apart from a result holding none of it.
_MESSAGE_ID = re.compile(r'"message_id"\s*:\s*"([0-9a-f]{32})"')

#: How a refusal names itself. It is the canonical encoding of a protocol error, which carries a
#: code from a closed set and no identifier, because a refusal is never a message.
_PROTOCOL_ERROR = re.compile(r'"kind"\s*:\s*"protocol_error"')
_REFUSAL_CODE = re.compile(r'"code"\s*:\s*"([a-z_]+)"')

#: What one presented message came to when the transcript was read for it. Matched is the same
#: bytes in the same place; mismatched is the identifier under other bytes, which is a message
#: the model was not shown whatever the history delivered; missing is neither.
MATCHED = "matched"
MISMATCHED = "mismatched"
MISSING = "missing"

#: How much of one fault's own text is kept. A fault is whatever the harness put on the error
#: channel rather than a record with fields, so the text is the whole of it, and it is kept short
#: because a session that ended against a service that had stopped answering ends in a great many
#: of them and what a reader wants of one is which service said no.
FAULT_TEXT = 200

#: How the harness marks a call it refused itself. Claude Code wraps its own tool failures in this
#: tag: arguments that would not parse, a name it does not know, a file it could not open. Such a
#: call never left the session, so its error is the agent's own and says nothing about whether the
#: service on the other side of the boundary was answering. What the service failed comes back
#: unwrapped, as the text the transport put on the error channel.
LOCAL_ERROR = "<tool_use_error>"

_COLUMNS =("task", "position", "attempt", "score", "ending", "payload", "seen", "calls")


@dataclass(frozen=True)
class Handed:
    """One text item the harness wrote into the result of a call to a served tool.

    The digest is of exactly that item, because the protocol's bytes travel as an item of their
    own: a message that came of some other call is delivered behind what that call landed with,
    and hashing the result whole would compare the message against the message plus whatever it
    rode in with. ``ids`` is what the item names itself with, which is how a result holding a
    clipped or rewritten message is told apart from one holding no message at all.
    """

    digest: str
    ids: FrozenSet[str]


@dataclass(frozen=True)
class Transcript:
    """What the agent's own transcript holds: its calls, and the messages it was handed.

    ``pulls`` and ``tasks`` are two facts rather than one. A pull is a call the model wrote, and
    writing one is not receiving work: the request can be refused, redirected or answered with an
    error, and the call stands in the transcript either way. So the tasks a run actually served
    are counted off the results those calls came back with.

    ``done_count`` is how many of those results were the record that ends the queue. It is the
    only place a session says the agent was told there is no more work: an agent decides for
    itself when to stop asking, so a session that stopped with the queue half served ends exactly
    as one that emptied it does.

    ``faults`` is what came back on the error channel from a served tool carrying no protocol code
    and no mark of the harness's own, in the order it came back. The three are different facts. A
    refusal is the protocol saying no to one call. A fault is the service failing to answer one at
    all. An error the harness marked as its own is the agent's call never having left the session,
    which is a mistake the agent made rather than anything about the service, and those are kept
    apart in ``local_errors``.

    ``unresolved_faults`` is how many faults stand with nothing served answered after them. It is
    not a count of what the transcript ends in, because a service that has stopped answering does
    not go quiet: the calls that follow a failed one are refused for overlapping with it, so the
    run ends in refusals with the fault that caused them further up. A served answer is what
    resolves a fault, and nothing else is.
    """

    per_attempt: Dict[str, int]
    pulls: int
    tasks: int
    unserved: int
    handed: Tuple[Handed, ...]
    refusals: Tuple[str, ...]
    done_count: int
    faults: Tuple[str, ...]
    local_errors: Tuple[str, ...]
    unresolved_faults: int


@dataclass(frozen=True)
class Checked:
    """One message the generation presented, and what the agent's transcript holds of it."""

    kind: str
    message_id: str
    attempt_id: Optional[str]
    status: str


@dataclass(frozen=True)
class Counted:
    """The transport's own count of the refusals it issued, or why this run has no count.

    The two are kept apart because they are answered differently. A count is compared with the
    transcript. An absence is a check nobody can make, which a run presented as finished may not
    have and a launch that says it stopped early is allowed.
    """

    count: Optional[int]
    absent: Optional[str]


def read_roster(run_dir: Path) -> Dict[str, object]:
    """Return the roster the serving process wrote into ``run_dir``."""
    return json.loads((Path(run_dir) / ROSTER_FILE).read_text(encoding="utf-8"))


def read_transcript(transcript: Path) -> Transcript:
    """Return the calls ``transcript`` records and the message identifiers it received.

    The transcript is Claude Code's own stream, one JSON object to a line, and a line this does
    not understand is skipped rather than raising: a transcript from an interrupted run is still
    a transcript, and a read that refused one would lose the run's only record of how much the
    agent did.

    Calls are read from what the model asked for and messages from what it was answered with, so
    the two halves come from different lines. Calls to anything but the served tools are counted
    together. What that number says is whether the agent was working the task through the
    affordances the cell gave it, which is the fact a rerun of a cell wants beside the score.

    A result counts as something the cell handed over only when it answers a call to one of the
    cell's own tools, which is the call it names itself under. A model can write a message into a
    call of its own and read one back out of a file, and neither is the protocol handing it
    anything: what the cell delivered is what came back from the cell.

    A pull is counted where the model wrote it and a task is counted where one came back, by that
    same identifier, and it is a task only if it is not an error and decodes as a Task: a pull the
    transport refused, redirected or answered with a protocol error is a call that received no
    work, and a run whose every pull ended that way served nothing however many of them the model
    wrote.

    The bytes are kept as a digest of each text item rather than as the item, because a run over
    a whole roster hands the model more text than a read of it needs to hold, and the whole of
    what the comparison asks is whether two byte strings are the same one.

    An error result is a refusal or a fault rather than a message, so it is read for the refusal
    code it carries. That code is the only record of a refusal there is: it advances no protocol
    state, so nothing in the generation counts it, and this transcript is where the model saw it.
    An error carrying no code is one of two other things. It is a fault, a service that would not
    answer, unless the harness marked it as its own refusal of the call, which it does by wrapping
    it. A call the harness would not make never reached the service: a malformed argument list is
    the agent writing a bad call, and reading one as a service failure would put the agent's own
    mistake on the other side of the boundary. So a fault is an error result from a served tool
    that carries no protocol code and no such mark, and it is kept as the text it came as rather
    than counted, because a run that ended in these ended against something and the text is what
    says what.

    Faults stay unresolved until something is served after them. A service that has failed a call
    does not go quiet: the next calls are refused for overlapping with the one it never finished,
    so a run that ended against it ends in refusals rather than in the fault that caused them. A
    refusal is therefore not evidence that the service came back, and neither is an error the
    harness raised without asking it. Only a served answer is.

    The record that ends the queue is counted where a task is, off the results a pull came back
    with. It is what says the agent was told there is no more work, and it is the one thing a
    session that stopped early does not hold.

    It is read a line at a time rather than all at once. A session that worked a whole roster with
    partial messages on writes a transcript far larger than the run it describes, and every launch
    reads this at the end to find out whether the agent ever asked for work.
    """
    per_attempt: Dict[str, int] = {}
    pulls = 0
    tasks = 0
    done_count = 0
    unserved = 0
    served: Dict[str, str] = {}
    asked: set = set()
    handed: List[Handed] = []
    refusals: List[str] = []
    faults: List[str] = []
    local_errors: List[str] = []
    unresolved = 0
    with Path(transcript).open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if event.get("type") == "user" and block.get("type") == "tool_result":
                    call = block.get("tool_use_id")
                    if not isinstance(call, str) or call not in served:
                        continue
                    texts = _result_texts(block.get("content"))
                    if block.get("is_error"):
                        asked.discard(call)
                        codes = _refusal_codes(texts)
                        if codes:
                            # The refusals that follow a failed call are what a service that has
                            # stopped answering says, so they leave the fault standing.
                            refusals.extend(codes)
                        elif _is_local(texts):
                            local_errors.append(_fault_text(texts))
                        else:
                            faults.append(_fault_text(texts))
                            unresolved += 1
                        continue
                    unresolved = 0
                    if call in asked:
                        asked.discard(call)
                        if any(_is_task(text) for text in texts):
                            tasks += 1
                    if served[call] == PULL_TOOL:
                        done_count += sum(1 for text in texts if _is_done(text))
                    handed.extend(
                        Handed(
                            digest=sha256(text.encode("utf-8")).hexdigest(),
                            ids=frozenset(_MESSAGE_ID.findall(text)),
                        )
                        for text in texts
                    )
                    continue
                if event.get("type") != "assistant" or block.get("type") != "tool_use":
                    continue
                name = str(block.get("name", ""))
                if not name.startswith(SERVED_PREFIX):
                    unserved += 1
                    continue
                call = block.get("id")
                if isinstance(call, str):
                    served[call] = name
                if name == PULL_TOOL:
                    pulls += 1
                    if isinstance(call, str):
                        asked.add(call)
                    continue
                arguments = block.get("input")
                attempt = arguments.get("attempt_id") if isinstance(arguments, dict) else None
                if isinstance(attempt, str):
                    per_attempt[attempt] = per_attempt.get(attempt, 0) + 1
    return Transcript(
        per_attempt=per_attempt,
        pulls=pulls,
        tasks=tasks,
        unserved=unserved,
        handed=tuple(handed),
        refusals=tuple(refusals),
        done_count=done_count,
        faults=tuple(faults),
        local_errors=tuple(local_errors),
        unresolved_faults=unresolved,
    )


def _is_task(text: str) -> bool:
    """Whether a result's text is the canonical bytes of a Task rather than something like one.

    The protocol's own decoder answers this, so a work order is what the wire calls a work order:
    the exact field set, each field's own shape, and the kind fixed on it. A refusal, a redirect's
    body, a truncated result or any other record the agent was handed instead is not a task, and a
    ``pull`` answered with one of those is a pull that served no work.
    """
    try:
        Task.from_wire(json.loads(text))
    except (json.JSONDecodeError, WireFormatError):
        return False
    return True


def _is_done(text: str) -> bool:
    """Whether a result's text is the canonical bytes of the record that ends the queue.

    The protocol's own decoder answers this for the reason it answers what a task is. The word
    names a served tool as well, the one an agent ends an attempt with, and it stands in the
    arguments the model writes to that tool and in whatever the tool answers with. None of those
    is the generation saying there is no more work, and only the record is.
    """
    try:
        Done.from_wire(json.loads(text))
    except (json.JSONDecodeError, WireFormatError):
        return False
    return True


def read_refusals(run_dir: Path) -> Counted:
    """How many calls the transport refused, or why this run does not say.

    The count is the cross-check on the transcript and never the record: it is kept by the party
    that issued the refusals, so a refusal sent and never delivered is a difference between the
    two rather than something neither holds.

    No count is not agreement. The server writes the number where it changes and again before it
    serves anything, so a run that has none is a run whose server never served, and a launch that
    presents such a run as finished is presenting an episode whose refusals nobody can account
    for. The reason is carried rather than flattened to a number, because what the read may do
    with the absence depends on what the launch claimed.
    """
    path = Path(run_dir) / REFUSAL_FILE
    if not path.is_file():
        return Counted(count=None, absent=f"no {REFUSAL_FILE}, so no server wrote a count")
    try:
        written = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Counted(count=None, absent=f"{REFUSAL_FILE} is not a record this read can decode")
    count = written.get("refusals") if isinstance(written, dict) else None
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return Counted(count=None, absent=f"{REFUSAL_FILE} holds no count of refusals")
    return Counted(count=count, absent=None)


def reconcile(presentations: List[PresentedMessage], transcript: Transcript) -> List[Checked]:
    """Say, of every message this generation presented, what the transcript holds of it.

    The two records are walked together rather than compared as sets. A presentation is looked
    for from where the last one was found, so what is being checked is that the model was shown
    these bytes in this order: a result carrying a message's bytes somewhere else in the
    conversation is not that message arriving where the generation delivered it.

    A message the transcript never carries the bytes of is missing, and one it carries the
    identifier of under other bytes is mismatched. Both are messages the model did not receive,
    and they are named apart because they fail differently: bytes that never arrived say the
    result was lost, and bytes that arrived changed say the result was rewritten on the way.
    """
    carried = {identifier for item in transcript.handed for identifier in item.ids}
    at = 0
    checked: List[Checked] = []
    for message in presentations:
        found = next(
            (
                index
                for index in range(at, len(transcript.handed))
                if transcript.handed[index].digest == message.visible_bytes_sha256
            ),
            None,
        )
        if found is not None:
            at = found + 1
            status = MATCHED
        else:
            status = MISMATCHED if message.message_id in carried else MISSING
        checked.append(
            Checked(
                kind=message.kind,
                message_id=message.message_id,
                attempt_id=message.attempt_id,
                status=status,
            )
        )
    return checked


def disagreements(
    checked: List[Checked],
    transcript: Transcript,
    refused: Counted,
    *,
    certified: bool,
) -> List[str]:
    """Every way this transcript and this generation's own record fail to say the same thing.

    A run with none of these is one whose analysis may say the agent received what the generation
    delivered. A run with any of them is one where it may not, and the read says so rather than
    leaving a warning beside a table that otherwise reads as whole.

    A missing refusal count is one of them when the launch calls the run finished. The check
    exists to catch a refusal the server sent and the model never saw, and a run with no count
    fails that check by having nothing to check against: reading the absence as agreement would
    accept exactly the case the count is kept for. A launch that says it did not finish is the
    one place the absence is allowed, and there it is reported as a check nobody could make
    rather than counted as one that passed.
    """
    out = [
        f"{check.status} {check.kind} {check.message_id}"
        + ("" if check.attempt_id is None else f" on attempt {check.attempt_id}")
        for check in checked
        if check.status != MATCHED
    ]
    if refused.count is None:
        if certified:
            out.append(f"this run says it finished and {refused.absent}")
    elif refused.count != len(transcript.refusals):
        out.append(
            f"the server refused {refused.count} calls and this transcript holds "
            f"{len(transcript.refusals)}"
        )
    return out


def _refusal_codes(texts: List[str]) -> List[str]:
    """The refusal codes an error result carries, which is one code or none.

    A fault is not a refusal and carries no code, so a result that is merely an error contributes
    nothing here: what is counted is the protocol saying no, which names itself as one.
    """
    codes = []
    for text in texts:
        if not _PROTOCOL_ERROR.search(text):
            continue
        found = _REFUSAL_CODE.search(text)
        codes.append(found.group(1) if found else "unnamed")
    return codes


def _is_local(texts: List[str]) -> bool:
    """Whether an error result is the harness refusing the call rather than the service failing it.

    The harness wraps its own, and what the service answered arrives unwrapped, so the mark is
    what tells them apart. The distinction is worth making because they say opposite things about
    a run: an argument list that would not parse is the agent writing a bad call and hearing about
    it at once, and the service on the other side of the boundary was never asked.
    """
    return any(text.strip().startswith(LOCAL_ERROR) for text in texts)


def _fault_text(texts: List[str]) -> str:
    """One error result as a record of the run keeps it: its text, as much as is worth holding.

    The items are joined here where a message's are not. A fault carries no identifier and is
    compared with nothing, so what it is for is being read, and a service that answered an error
    across two items said one thing.
    """
    return " ".join(text.strip() for text in texts if text.strip())[:FAULT_TEXT]


def _result_texts(content: object) -> List[str]:
    """Return one tool result's text items, whichever of the two shapes the harness wrote it in.

    They stay a list. A message delivered behind the call that produced it travels as an item of
    its own beside what that call landed with, so joining them would hash the message together
    with its neighbour and match nothing.
    """
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [str(item.get("text", "")) for item in content if isinstance(item, dict)]
    return []


def rows(
    roster: Dict[str, object],
    records: List[AttemptRecord],
    transcript: Optional[Transcript] = None,
    checked: Optional[List[Checked]] = None,
) -> List[List[str]]:
    """Return one row per roster position, in the order the roster assigned them.

    A position the run never reached is printed as the position it is rather than dropped: a cell
    that stopped early is a cell that stopped early, and a table that silently shortened itself
    would read as a shorter roster.
    """
    by_attempt = {record.attempt_id: record for record in records}
    per_attempt: Dict[str, List[Checked]] = {}
    for check in checked or []:
        if check.attempt_id is not None:
            per_attempt.setdefault(check.attempt_id, []).append(check)
    out: List[List[str]] = []
    entries = roster.get("tasks")
    for entry in entries if isinstance(entries, list) else []:
        attempt = str(entry["attempt_id"])
        record = by_attempt.get(attempt)
        out.append(
            [
                str(entry["task"]),
                str(entry["task_position"]),
                attempt,
                _score(record),
                _ending(record),
                _payload(record),
                _seen(record, None if checked is None else per_attempt.get(attempt, [])),
                _calls(attempt, transcript),
            ]
        )
    return out


def unconfirmed(rows_: List[List[str]]) -> int:
    """Return how many rows hold a message the transcript does not confirm the model was shown."""
    return sum(1 for row in rows_ if row[_COLUMNS.index("seen")] not in ("ok", "-"))


def format_table(rows_: List[List[str]]) -> str:
    """Return the rows as a table one terminal line wide per position."""
    if not rows_:
        return "no roster"
    widths = [max(len(row[i]) for row in [list(_COLUMNS), *rows_]) for i in range(len(_COLUMNS))]
    lines = [_line(list(_COLUMNS), widths), *(_line(row, widths) for row in rows_)]
    return "\n".join(lines)


def _line(cells: List[str], widths: List[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()


def _score(record: Optional[AttemptRecord]) -> str:
    """The score the history committed. An attempt nobody sealed has none, and says so."""
    if record is None or record.score is None:
        return "-"
    return f"{record.score:g}"


def _ending(record: Optional[AttemptRecord]) -> str:
    """How the attempt ended, or that the run never got to it.

    A position whose task was never delivered is one the agent never saw, and its state is the
    generation's bookkeeping rather than an ending: a row left planned behind a run that stopped
    reads as not reached. An attempt floored without ever being served says which floor it was
    instead, because that row was scored and a row nobody reached was not.

    A sealed row says who filed it. This env's horizon is a graded ending, so an attempt that
    spent its whole budget is sealed and scored exactly like one the agent finished, and the
    state on its own cannot tell a workflow the agent called `done` on from one that ran out of
    calls. Those are two behaviours a cell counts separately, so the horizon is named where it
    was what filed.
    """
    if record is None:
        return "not reached"
    if record.final_failure:
        return record.final_failure
    if not record.task_delivered:
        return "not reached"
    if record.terminal_source == "horizon":
        return f"{record.state} (horizon)"
    return record.state


def _payload(record: Optional[AttemptRecord]) -> str:
    """What the history delivered against the attempt, and under what rule.

    Delivery is what this column reports, which is the exact bytes handed to the transport that
    carries them, and what the model made of those bytes is the column beside it.

    Three absences are three different facts and each is named. A row that was never going to
    have a payload says so and names the reason. A row whose attempt ended without a filing owes
    nothing any more, because the ending resolved the obligation where it stood. A row the run
    never reached has no delivery state at all, and printing one would read as a payload this
    generation failed to deliver.
    """
    if record is None:
        return "-"
    if not record.creates_payload_obligation:
        return f"none ({record.payload_disposition or 'withheld'})"
    if record.payload_delivered:
        return f"delivered ({record.payload_policy})"
    if record.final_failure:
        return f"none ({record.final_failure})"
    if not record.task_delivered:
        return "-"
    return f"owed ({record.payload_policy})"


def _seen(record: Optional[AttemptRecord], checked: Optional[List[Checked]]) -> str:
    """Which of this attempt's delivered messages the agent's transcript actually holds.

    The history attests that bytes were handed to the transport, and the transcript is where the
    harness wrote what came back to the model. A run where the two agree is the ordinary case. A
    message the history delivered and the transcript does not hold exactly is a message the
    analysis must not read as one the agent saw, so the first one is named here rather than left
    to the delivery column to imply.

    Every kind this attempt was presented is in the comparison, not the three the row has columns
    for: a SealReject the model was shown and the transcript lost is a turn the analysis would
    read as never having happened.
    """
    if record is None or checked is None or not checked:
        return "-"
    failed = next((check for check in checked if check.status != MATCHED), None)
    return "ok" if failed is None else f"{failed.status} {failed.kind}"


def _calls(attempt: str, transcript: Optional[Transcript]) -> str:
    """How many of the attempt's own tools were called, with no transcript read as no answer.

    A transcript that holds no call against an attempt is a transcript saying the agent made
    none, which is a number. Having no transcript to read is the absence of one, and the two
    would be the same cell if a missing count were printed as a dash.
    """
    if transcript is None:
        return "-"
    return str(transcript.per_attempt.get(attempt, 0))


__all__ = [
    "MATCHED",
    "MISMATCHED",
    "MISSING",
    "PULL_TOOL",
    "SERVED_PREFIX",
    "Checked",
    "Counted",
    "Handed",
    "Transcript",
    "disagreements",
    "format_table",
    "read_refusals",
    "read_roster",
    "read_transcript",
    "reconcile",
    "rows",
    "unconfirmed",
]
