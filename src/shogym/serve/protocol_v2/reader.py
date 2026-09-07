"""Read a run's records back out of the authority that holds them.

A generation's history is the record. The workflow holds every attempt's assignment, filing,
score and presentations, and answers for them through a Query, so reading a run means bringing
that history back up and asking it rather than parsing anything a run wrote down as it went.

That is what this module does. It opens a run directory, brings the history it holds back up on
a copy of itself, runs a Worker long enough to answer one Query, and returns the records of the
generation the manifest names.

Bringing a history up is not a passive act, so the copy of it that comes up belongs to the
read. A service fires the durable timers it finds waiting, and a Worker is a poller that runs
whatever the queue hands it, so a run left with an expired attempt deadline would be ended by
the act of looking at it. The run's own file is never opened by any of that: what a read costs
is paid in a directory that goes away with the process, and the history a run directory holds
is the same bytes after a read as before it.

Costing the run nothing is not the same as answering with the run's own rows, which is the
second thing a read owes. A copy the read moved would answer with rows the read made: an
overdue clock or an environment's answer applied in the scratch history says that an attempt
ended, while the run itself still holds it open and its owner is who decides what it comes to.
So the copy's history is marked before the Worker starts and again once it has stopped, and a
read that moved what it was reading is refused rather than answered.

``SHOGYM_TEMPORAL_ADDRESS`` names a service somebody else runs, and there is nothing to copy
there: that history belongs to whoever runs it. So a read of one starts nothing at all. It asks
the deployment as a client, and a deployment with no Worker serving that generation answers
nothing, which is a refusal here. Starting a Worker to answer for it would put this read on the
live generation's own queue, where the first thing it would be handed is the run's own
unfinished work.

:func:`write_records` puts those rows in the run directory as JSON Lines, and the generation's
failed operations in a second file beside them, in commit order. Both are derived views and
never an authority: they are rewritten from the history every time they are asked for, nothing
here or anywhere else reads either back to decide anything, and a note beside them says so to
whoever finds the directory later.

A directory that holds no history is not an error to raise a traceback over. It is a directory
with nothing to read, and it is reported as :class:`NothingToRead` with the reason in it. Every
way a read could only answer by moving something is the other answer, :class:`ReadRefused`.

A run that served committed source cells is read on two axes rather than one. What became of the
receipt is :func:`receipt_lifecycle`, seven values tested in a fixed order with the first match
winning, and whether the evidence behind it is still there is :func:`receipt_availability`, which
reads beside that answer and never in place of it. One list could not carry both: a committed row
whose objects later go missing is committed and unavailable, both of which are true, and a
withheld row that loses a body has no selection for an unavailable predicate to be about. Exactly
one lifecycle value matches every row a generation projects.

Committed there is a protocol commitment and nothing more. Whether a model then read those bytes
is the harness's own fact, so :func:`observed_receipt` answers it against what the harness wrote
down: the message identity has to be the one the generation committed and the digest of the bytes
in the transcript has to be the digest the generation stands behind. Where both hold the receipt
was observed, and where the commitment stands alone it was committed and not observed.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from temporalio.client import Client, WorkflowExecutionDescription
from temporalio.service import RPCError, RPCStatusCode

from shogym.serve.protocol_v2.kernel.messages import (
    REFUSED_OPERATION,
    UNRECOVERABLE_OPERATION,
    AttemptRecord,
    GenerationRecords,
    OperationFailure,
    PresentedMessage,
    SourceProvenance,
)
from shogym.serve.protocol_v2.kernel.runtime import (
    STREAM_DATABASE_FILE,
    TEMPORAL_ADDRESS_ENV,
    durable_client,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel.workflow import (
    ELIGIBLE,
    MATERIALIZED,
    OFFERED,
    PRESENTED,
    StreamWorkflow,
)
from shogym.serve.protocol_v2.rundir import RunDirectory, open_run_directory

#: The derived view a run directory holds, and the note that says it is one.
RECORDS_FILE = "records.jsonl"
NOTE_FILE = "records.jsonl.note"
#: The generation's failed operations, in commit order, beside the attempts they are about.
OPERATIONS_FILE = "operation_failures.jsonl"

# How long a Query gets to reach a Worker on a service somebody else runs. There is nothing to
# start there, so the only thing that answers is a deployment already serving that generation,
# and an unbounded wait for one that is not would hang a read rather than report it.
_A_SERVING_WORKER_ANSWERS_WITHIN = timedelta(seconds=15)

# What the service says when that wait ran out with nobody having answered. It is a refusal
# here rather than a fault: the generation is there, and what is missing is a Worker for it.
_NOBODY_ANSWERED = (RPCStatusCode.CANCELLED, RPCStatusCode.DEADLINE_EXCEEDED)

_NOTE = """\
{records} is a derived view, rebuilt by `shogym results` from the durable history of
{workflow} every time it is asked for.

The history is the record. Nothing reads this file back to decide anything, and an edit to it
changes no fact about the run: delete it and the next read writes it again.

A row whose seal could not go on explains itself here and not in the printed table:
`failure_activity` and `failure_activity_id` name the step the service gave up on,
`failure_kind` is what it failed as, `failure_message` is a bounded line of what it said, and
`failure_retry_state` is why the retries stopped. A row ended by a result the seal could not
vouch for names no step and no retry state, because its step succeeded: it carries the kind and
the message alone, and those are the check that refused the answer.

A row that served a committed source cell carries `source_provenance`, which is what that source
was and which cell of it this row was served, and two answers read off the row rather than held
in it. `receipt_lifecycle` is what became of the receipt and `receipt_availability` is whether
its evidence is still there; the second reads beside the first and never in place of it, so a
row can be committed and unavailable, which says the commitment happened and the objects behind
it went missing afterwards.

{operations} is the other half of that second answer, rebuilt from the same history at the same
time: every operation of this generation that could not produce the evidence it depended on, in
commit order, with the operation, the phase, the reason, the outcome, the generation whose
operation it was, the epochs it was refused and answered under and the objects it could not
produce. A refusal and the recovery that answered it both stand there, and an operation that
failed before any receipt sealed is a row here and a row in no attempt's line.
"""

# What became of one attempt's receipt. Seven values, tested in this order, first match winning.
#
# Legacy is first for a reason: an attempt from before any of this existed can end in a seal
# failure like any other, and absence there predates the question rather than answering it, so it
# reads as legacy and never as a receipt that failed.
#
# Withheld is the roster's own answer and not a missed delivery. A position given no payload
# obligation still seals, still grades and still records its source, so there is no selection for
# it to be missing: the absence is structural.
RECEIPT_LEGACY = "legacy"
RECEIPT_FAILED_BEFORE_CAPTURE = "failed_before_capture"
RECEIPT_ASSIGNED_BUT_UNSEALED = "assigned_but_unsealed"
RECEIPT_WITHHELD = "withheld"
RECEIPT_BUILT = "built"
RECEIPT_OFFERED = "offered"
RECEIPT_COMMITTED = "committed"
RECEIPT_LIFECYCLES: Tuple[str, ...] = (
    RECEIPT_LEGACY,
    RECEIPT_FAILED_BEFORE_CAPTURE,
    RECEIPT_ASSIGNED_BUT_UNSEALED,
    RECEIPT_WITHHELD,
    RECEIPT_BUILT,
    RECEIPT_OFFERED,
    RECEIPT_COMMITTED,
)

# And whether the evidence behind it is still there. This is the second axis and it replaces no
# lifecycle value: a committed row that later loses an object is committed and unavailable, what
# was committed having happened and no later loss unhappening it.
RECEIPT_AVAILABLE = "available"
RECEIPT_UNAVAILABLE = "unavailable"
RECEIPT_AVAILABILITIES: Tuple[str, ...] = (RECEIPT_AVAILABLE, RECEIPT_UNAVAILABLE)

# The states an obligation is in while its candidate exists and nobody has been offered it.
_OWED = (MATERIALIZED, ELIGIBLE)

_COLUMNS = (
    "task",
    "attempt",
    "state",
    "seal",
    "score",
    "ending",
    "filed",
    "decode",
    "delivered",
    "policy",
    "profile",
    "decided",
)


# What one read comes back with, before it is a :class:`RunRecords`: the attempts, and the
# messages this generation committed to deliver in the order it committed them. Both halves come
# back from one Query, so they describe the generation at one moment rather than at two.
_Answer = GenerationRecords


class NothingToRead(RuntimeError):
    """A run directory with no records to read, carrying the reason it has none."""


class ReadRefused(RuntimeError):
    """A run whose history could not be read without a read being what moved it."""


@dataclass(frozen=True)
class RunRecords:
    """What one run directory answered with: which generation, its attempts, its commitments.

    ``presentations`` is what the generation committed to deliver, in order. It is read beside
    the records rather than instead of them, because the two answer different questions: a record
    says what an attempt came to, and a presentation says which bytes were committed for it.

    A commitment is where the claim stops. It is accepted before the transport has a result to
    give anybody, so whether those bytes reached the transport at all, and whether a model then
    read them, are facts about the harness. A harness that keeps the model's transcript is the
    one that can say either, by reconciling its own writing against these rows; a harness that
    keeps no transcript can ignore them.

    ``operation_failures`` is the third list, read off that same moment. An attempt whose evidence
    went missing reads as delivered in one answer and as refused in another, and which of the two
    is true is a question about one point in the run. The rows are kept in commit order, and the
    last one naming an attempt is the latest word about it.

    All three are answered by one Query, so a row and the commitments beside it are the same
    generation at the same point rather than two reads of one that moved between them.
    """

    root: Path
    workflow_id: str
    records: List[AttemptRecord]
    presentations: List[PresentedMessage] = dataclasses.field(default_factory=list)
    operation_failures: List[OperationFailure] = dataclasses.field(default_factory=list)


async def read_records(root: Union[str, Path]) -> RunRecords:
    """Return the records of the generation ``root`` holds.

    The directory is read first, so a directory that names no generation, or names one under a
    version this code does not serve, is refused before anything is started or connected to.
    Where the answer comes from after that is the one thing that differs between a run keeping
    its own history and a run served against a service somebody else runs.
    """
    run = open_run_directory(root)
    _require_authority(run.root)
    if os.environ.get(TEMPORAL_ADDRESS_ENV):
        answer = await _read_through_a_deployment(run)
    else:
        answer = await _read_off_a_copy(run)
    return RunRecords(
        root=run.root,
        workflow_id=run.manifest.workflow_id,
        records=list(answer.attempts),
        presentations=list(answer.presentations),
        operation_failures=list(answer.operation_failures),
    )


async def _read_off_a_copy(run: RunDirectory) -> _Answer:
    """Answer out of a copy of this run's own history, or refuse because the read moved it.

    The copy is what makes a read cost the run nothing. Neither half of a read can be asked to
    change nothing: a service fires the durable timers it finds waiting the moment it comes up,
    so a run left with an expired attempt deadline is ended by having a service pointed at it
    whether or not anybody polls, and a Query is answered by a Worker, which is a poller, so
    whatever the queue has ready goes to it. The copy is a plain copy of the files the service
    keeps, which is consistent because a directory being read is a directory nothing is serving:
    two services on one database is not an arrangement this supports.

    The Worker that answers registers no Activity. Answering a Query needs the workflow, which
    is this package's and is deterministic. The seal, the grade and the payload bundle are the
    environment's, and an environment that seals and grades for itself installs its own
    implementations under the same Activity names. A reader that registered the kernel's
    stand-ins would answer in their place, so a generation stopped with a seal in flight would
    come back scored by whoever read it. Nothing here may produce a fact about an attempt, so
    nothing that could is registered.

    The generation is described before that Worker is started, because a generation carrying a
    workflow task nobody has applied hands it to whoever polls next and the rows that came back
    would be the read's. A generation that is merely open is not refused: one waiting on a pull
    and one stopped with its seal in the environment's hands both have nothing for a poller to
    run, and reading those is what this exists for.

    That check is one round trip and the window after it is not empty. A clock that falls due a
    moment later makes work ready that nobody was holding when the question was asked, and the
    Worker started next would run it. So the copy's history is marked here, the Worker is
    stopped before the mark is taken again, and a history that moved between the two says the
    rows just read are the read's own. There is no salvaging that by reading again: what moved
    is this run's next step, and its owner is who decides what that step comes to.
    """
    with TemporaryDirectory(prefix="shogym-read-") as scratch:
        for path in sorted(run.root.glob(f"{STREAM_DATABASE_FILE}*")):
            shutil.copy2(path, Path(scratch) / path.name)
        workflow_id = run.manifest.workflow_id
        async with durable_client(run_directory=Path(scratch)) as client:
            arrived = await _described(client, workflow_id)
            if arrived is not None:
                _refuse_unapplied_work(workflow_id, arrived)
            async with stream_worker(client, task_queue=run.manifest.task_queue, activities=[]):
                answer = await _query(client, workflow_id)
            _refuse_a_moved_history(
                workflow_id,
                _where_it_was(arrived),
                _where_it_was(await _described(client, workflow_id)),
            )
            return answer


async def _read_through_a_deployment(run: RunDirectory) -> _Answer:
    """Ask a service somebody else runs, as a client of it and never as a Worker on it.

    There is no copy to take here. The history belongs to whoever runs the service, and a run
    served against one is read where it lives, so every protection a copy gave is gone and the
    read has to be the kind of thing that cannot move a history at all. A Worker started here
    would poll the live generation's own task queue, and the first thing the server would hand
    it is that generation's own unapplied work: an environment's answer would be applied and a
    digest recorded, an overdue deadline would end an attempt whose owner may yet resume it.
    Both would be committed in the authority itself rather than in a copy.

    So nothing is started, and what answers is whatever is already serving that generation. A
    Query needs a Worker to replay against, so a deployment with none simply does not answer,
    and that is reported as a refusal rather than waited on for ever or made true by starting
    one. A run whose deployment is up is read the moment it is asked.
    """
    workflow_id = run.manifest.workflow_id
    async with durable_client() as client:
        handle = client.get_workflow_handle_for(StreamWorkflow.run, workflow_id)
        try:
            # One Query for both halves. A live generation moves between two of them, and a
            # read that asked twice would answer with rows from one moment and commitments from
            # another, which is a table describing no generation that ever existed.
            return await handle.query(
                StreamWorkflow.generation_records,
                rpc_timeout=_A_SERVING_WORKER_ANSWERS_WITHIN,
            )
        except RPCError as error:
            if error.status is RPCStatusCode.NOT_FOUND:
                raise NothingToRead(
                    f"the service at {os.environ[TEMPORAL_ADDRESS_ENV]} holds no generation "
                    f"{workflow_id}, which is the one this run's manifest names"
                ) from error
            if error.status not in _NOBODY_ANSWERED:
                raise
            raise ReadRefused(
                f"no Worker serving {workflow_id} answered within "
                f"{_A_SERVING_WORKER_ANSWERS_WITHIN.total_seconds():.0f}s, and a read may not "
                f"start one on a service it does not own, so this run is read while whoever "
                f"runs that service is serving it"
            ) from error


def write_records(run: RunRecords) -> Path:
    """Write ``run``'s records into its directory as JSON Lines, and return the path.

    One object per attempt, in the record's own field order, so two runs of this produce two
    files that compare as text. The note is written beside it every time, because a file whose
    reader has to be told it is derived should say so where it is found rather than only where
    it was documented.

    The two receipt axes are written after the record's own fields, because they are read off the
    row rather than held in it: what became of the receipt and whether its evidence is still
    there are an answer about this moment, and the operation failures they are computed against
    are a list of the generation's and not a column of the attempt's.

    Those failures are written out as their own file, in commit order, member by member. An
    attempt row can only carry the one word the availability axis comes to, and the episode is
    more than that word: which operation failed, at which phase, for which reason, under which
    epoch, which objects it could not produce and whether anything answered it afterwards. A
    failure before any receipt sealed names no attempt at all and has no row here to be part of.
    So the list is exported beside the records rather than folded into them, and an external
    reader holding both files holds what one consistent Query answered with.

    One Query, and one writer at a time. The two files are written one after the other and
    nothing here makes the pair atomic, so two exports of one run racing each other could leave
    a reader holding one file from each. What is promised is that a snapshot is exported whole,
    and a deployment wanting concurrent exporters owes them a publication rule this does not have.
    """
    path = run.root / RECORDS_FILE
    rows = [json.dumps(_row(record, run.operation_failures)) + "\n" for record in run.records]
    path.write_text("".join(rows), encoding="utf-8")
    episodes = [json.dumps(_failure_row(row)) + "\n" for row in run.operation_failures]
    (run.root / OPERATIONS_FILE).write_text("".join(episodes), encoding="utf-8")
    (run.root / NOTE_FILE).write_text(
        _NOTE.format(
            records=RECORDS_FILE, operations=OPERATIONS_FILE, workflow=run.workflow_id
        ),
        encoding="utf-8",
    )
    return path


def format_records(records: List[AttemptRecord]) -> str:
    """Return the records as a table one terminal line wide per attempt.

    The score column is the one a reader came for, so an unsealed attempt reads as a dash and
    never as a zero. The ending is printed beside it, because an attempt that ended without a
    filing scores the floor and the number on its own does not say that is what it is. And who
    filed is printed beside that, because a generation over an environment whose horizon is a
    graded ending files for an attempt that ran out of world work, so a sealed row no longer
    says the agent chose to stop there. The attempt identifier is printed whole: it is what a
    row is named by, and a shortened one is a row somebody has to go back to the file for.

    These columns are what a person reads, and they are not everything a row holds. This is the
    answer to what a generation came to, one line wide and read at a glance, so the derived
    ``records.jsonl`` carries every field of the record and this stays the columns that fit.

    The policy column says what the agent was allowed to be told about that score. A run
    recorded before a generation carried one reads as the legacy placeholder, which is what its
    bodies were, and never as an honest receipt nobody can now check.

    The profile and the column beside it say who decided that. The same policy is the right
    answer for an ordinary run the platform stamped and for an experiment cell somebody
    registered, so a reader asking whether a run was blinded by design or by accident is asking
    these two columns rather than the policy.
    """
    if not records:
        return "no attempts"
    rows = [_cells(record) for record in records]
    widths = [max(len(row[index]) for row in [_COLUMNS, *rows]) for index in range(len(_COLUMNS))]
    lines = [_line(_COLUMNS, widths), *(_line(row, widths) for row in rows)]
    return "\n".join(lines)


def receipt_lifecycle(record: AttemptRecord) -> str:
    """Return what became of one attempt's receipt, as one of the seven values.

    The order is the definition. A pre-artifact attempt that ended in a seal failure is legacy
    and not a receipt that failed, because a generation that declared no contract was never asked
    for a source; an attempt under a contract that ended before it captured one is the second
    answer, and the seal's own failure fields say at which stage. Everything from the fourth
    value on holds a descriptor, so what separates them is the roster and the obligation: no
    obligation at all is withheld, a candidate nobody has been offered is built, a reservation is
    offered, and a presentation is committed.

    Exactly one value describes any row a generation projects, and a row that describes none is
    reported rather than labelled: the whole point of two axes was to stop a row that matched
    nothing from being filed under whichever value was closest.
    """
    for value, matches in _LIFECYCLE_TESTS:
        if matches(record):
            return value
    raise ValueError(
        f"attempt {record.attempt_id} is in no state this generation projects: it holds "
        f"{'a' if record.source_provenance is not None else 'no'} committed source, its "
        f"obligation is {record.payload_state!r} and its ending is {record.final_failure!r}"
    )


def receipt_availability(
    record: AttemptRecord, failures: Sequence[OperationFailure]
) -> str:
    """Return whether the evidence behind one attempt's receipt is still there.

    It reads beside the lifecycle and never in place of it. A committed row that later loses an
    object is committed and unavailable, what was committed having happened and no later loss
    unhappening it, and a withheld row that loses a body is withheld and unavailable, which is a
    state a predicate needing a selection could not reach at all.

    The last row naming this attempt is what answers, because the rows are an episode: a refusal
    and the recovery that answered it both stand, and the latest word is the one at the end. So an
    attempt reads available while an earlier pull of its own is still unresolved, where a later
    operation on another of its objects recovered after it. The ordered list keeps that earlier
    refusal, and the delivery depending on those bytes checks them again.
    """
    naming = [row for row in failures if row.attempt_id == record.attempt_id]
    if naming and naming[-1].outcome in (REFUSED_OPERATION, UNRECOVERABLE_OPERATION):
        return RECEIPT_UNAVAILABLE
    return RECEIPT_AVAILABLE


def observed_receipt(record: AttemptRecord, witness: Mapping[str, str]) -> bool:
    """Say whether the harness's own transcript witnesses this attempt's committed receipt.

    ``witness`` is what the harness wrote down, by message identity, as the digest of the bytes
    it wrote. A commitment is where this generation's claim stops: it is accepted before the
    transport has a result to hand anybody, so an acknowledgement can be lost and the owner
    replaced before the bytes reach a model. What upgrades it is the harness's own durable entry
    for that message, matched on the identity and on the digest, and both have to hold.

    So a row that never committed is never observed, a message the transcript has no entry for is
    committed and not observed, and an entry whose digest is not the one this generation stands
    behind is not a witness for it. Nothing here imputes a dose either way: what comes back is
    whether the two records agree.
    """
    if receipt_lifecycle(record) != RECEIPT_COMMITTED:
        return False
    written = witness.get(record.payload_message_id)
    return written is not None and written == record.payload_visible_sha256


# The seven predicates, in the order they are tested. Each is written whole rather than leaning
# on the ones above it, so the values are disjoint as definitions and not only as an order: a
# reader checking that exactly one describes a row is asking about the definitions.
_LIFECYCLE_TESTS: Tuple[Tuple[str, Callable[[AttemptRecord], bool]], ...] = (
    (
        RECEIPT_LEGACY,
        lambda row: row.source_provenance is None and row.receipt_contract_id is None,
    ),
    (
        RECEIPT_FAILED_BEFORE_CAPTURE,
        lambda row: row.source_provenance is None
        and row.receipt_contract_id is not None
        and row.final_failure is not None,
    ),
    (
        RECEIPT_ASSIGNED_BUT_UNSEALED,
        lambda row: row.source_provenance is None
        and row.receipt_contract_id is not None
        and row.final_failure is None,
    ),
    (
        RECEIPT_WITHHELD,
        lambda row: row.source_provenance is not None
        and row.score is not None
        and not row.creates_payload_obligation,
    ),
    (
        RECEIPT_BUILT,
        lambda row: row.source_provenance is not None
        and row.source_provenance.selected_cell is not None
        and row.creates_payload_obligation
        and row.payload_state in _OWED,
    ),
    (
        RECEIPT_OFFERED,
        lambda row: row.source_provenance is not None
        and row.creates_payload_obligation
        and row.payload_state == OFFERED,
    ),
    (
        RECEIPT_COMMITTED,
        lambda row: row.source_provenance is not None
        and row.creates_payload_obligation
        and row.payload_state == PRESENTED
        and row.payload_delivered
        and bool(row.payload_visible_sha256),
    ),
)


def _require_authority(root: Path) -> None:
    """Refuse a directory whose history is not here, saying which half is missing.

    A run served without a directory kept its history in a temporary file that went away with
    the process, and a directory that was never served has none yet. Both look the same from
    here: blobs and a manifest, and nothing that can answer for an attempt.
    """
    if os.environ.get(TEMPORAL_ADDRESS_ENV):
        return
    if (root / STREAM_DATABASE_FILE).is_file():
        return
    raise NothingToRead(
        f"{root} holds no {STREAM_DATABASE_FILE}, so the history its records would be read out "
        f"of is not here"
    )


async def _described(client: Client, workflow_id: str) -> Optional[WorkflowExecutionDescription]:
    """What the service says about this generation, or nothing when it holds no such thing.

    A generation nobody ever started is left alone here rather than reported: :func:`_query` is
    where an absent one gets its own answer, which is the one that names the manifest.
    """
    handle = client.get_workflow_handle_for(StreamWorkflow.run, workflow_id)
    try:
        return await handle.describe()
    except RPCError as error:
        if error.status is not RPCStatusCode.NOT_FOUND:
            raise
        return None


def _refuse_unapplied_work(workflow_id: str, described: WorkflowExecutionDescription) -> None:
    """Refuse a generation holding a workflow task nobody has applied.

    The server gives a pending workflow task to the next Worker that polls, and it answers a
    Query against a generation that has one by handing over both together, so there is no
    Worker that could answer here without running that task. What is pending is the run's own
    work: the result its environment has already returned and whatever the generation does
    next with it. Applying it costs the run nothing now that the read has its own copy, but the
    rows that came back would be the ones the read produced rather than the ones the run holds,
    and an owner resuming it later is who decides what that work comes to.

    So the answer is that this run cannot be read yet.
    """
    if not described.raw_description.HasField("pending_workflow_task"):
        return
    raise ReadRefused(
        f"{workflow_id} stopped holding a workflow task nobody has applied, and answering a "
        f"Query needs a Worker that would apply it, so this run is read once its own owner has "
        f"resumed it and that work has landed"
    )


def _where_it_was(
    described: Optional[WorkflowExecutionDescription],
) -> Optional[Tuple[str, int]]:
    """Which execution this generation is in and how far it has got, as one mark.

    Both halves come out of one Describe, which is what makes them one moment rather than two.
    The pair is the mark rather than the count alone, because a generation may run in more than
    one execution: it continues as new before the durable service's limits bound it, under the
    same identifier, and the count starts again in the execution that follows. A count on its
    own would fall at that boundary, and a count that fell would be read as a history going
    backwards, which is not a thing that happens.
    """
    if described is None:
        return None
    info = described.raw_description.workflow_execution_info
    return info.execution.run_id, info.history_length


def _refuse_a_moved_history(
    workflow_id: str, before: Optional[Tuple[str, int]], after: Optional[Tuple[str, int]]
) -> None:
    """Refuse the read whose own Worker took the generation past where it found it.

    The question is whether the copy that answered is the copy the read arrived at, and the mark
    answers it by being equal or not. Equality rather than order is the test, and it has to be:
    the count can rise, because the Worker started for the read ran work that was ready, and it
    can start again in another execution, because the generation crossed a boundary while the
    read was in it. Both mean the same thing, that the rows this read answered with are the
    read's own next step rather than the run's.

    Reading again does not repair it. What moved is the step, and the owner that resumes the run
    is who decides what it comes to, so the answer is the same refusal an unapplied task gets.
    """
    if before is None or after == before:
        return
    raise ReadRefused(
        f"reading {workflow_id} moved it: it was at {_mark(before)} before the Query and "
        f"{_mark(after)} after, so the rows this read answered with are the read's own rather "
        f"than the run's, and this run is read once its own owner has resumed it and that work "
        f"has landed"
    )


def _mark(where: Optional[Tuple[str, int]]) -> str:
    """One end of the comparison, in words, for the refusal that reports it."""
    if where is None:
        return "no execution at all"
    run_id, length = where
    return f"{length} events of execution {run_id}"


async def _query(client: Client, workflow_id: str) -> _Answer:
    """Ask one generation for its rows and its commitments, or say this history holds neither.

    One Query answers with both, so the two halves are the projection as it stood in one handler
    call. Asking twice would be asking a generation that can commit a presentation in between,
    and the pair would describe a run that was never in either state.
    """
    handle = client.get_workflow_handle_for(StreamWorkflow.run, workflow_id)
    try:
        return await handle.query(StreamWorkflow.generation_records)
    except RPCError as error:
        if error.status is not RPCStatusCode.NOT_FOUND:
            raise
        raise NothingToRead(
            f"the history here holds no generation {workflow_id}, which is the one this run's "
            f"manifest names"
        ) from error


def _row(record: AttemptRecord, failures: Sequence[OperationFailure]) -> Dict[str, Any]:
    """Return one record as the JSON object a line holds, in the record's field order.

    The nested provenance is converted here rather than left to the encoder. What the builder
    hands over is the field's own value, and the source provenance is a record and not a string,
    so an object would reach the encoder as an object it has no rule for. Naming the members is
    also what keeps the file's shape a decision rather than a consequence of a dataclass.
    """
    row = {field.name: getattr(record, field.name) for field in dataclasses.fields(record)}
    row["source_provenance"] = _provenance_row(record.source_provenance)
    row["receipt_lifecycle"] = receipt_lifecycle(record)
    row["receipt_availability"] = receipt_availability(record, failures)
    return row


def _failure_row(row: OperationFailure) -> Dict[str, Any]:
    """Return one failed operation as the object its line holds, member by member.

    The members are named rather than read off the dataclass for the reason the provenance is:
    what this file's shape is should be a decision somebody made, and a field added to the record
    should reach an external reader because somebody put it here.
    """
    return {
        "operation": row.operation,
        "phase": row.phase,
        "reason": row.reason,
        "outcome": row.outcome,
        "generation": row.generation,
        "refused_epoch": row.refused_epoch,
        "attempt_id": row.attempt_id,
        "payload_position": row.payload_position,
        "references": list(row.references),
        "request_id": row.request_id,
        "recovered_epoch": row.recovered_epoch,
        "protocol_version": row.protocol_version,
    }


def _provenance_row(source: Optional[SourceProvenance]) -> Optional[Dict[str, Any]]:
    """Return one attempt's committed source as the object its line holds, member by member."""
    if source is None:
        return None
    return {
        "source_commitment": source.source_commitment,
        "bundle_digest": source.bundle_digest,
        "environment_task_id": source.environment_task_id,
        "source_attempt_id": source.source_attempt_id,
        "source_seal_id": source.source_seal_id,
        "execution_ordinal": source.execution_ordinal,
        "canonicalization_version": source.canonicalization_version,
        "renderer_configuration": source.renderer_configuration,
        "canonical_submission_sha256": source.canonical_submission_sha256,
        "kernel_submission_digest": source.kernel_submission_digest,
        "selected_cell": source.selected_cell,
        "selected_body_reference": source.selected_body_reference,
        "selected_policy_digest": source.selected_policy_digest,
        "protocol_version": source.protocol_version,
    }


def _cells(record: AttemptRecord) -> Tuple[str, ...]:
    """Return one record as the strings its row is printed from."""
    delivered = [
        name
        for name, shown in (
            ("task", record.task_delivered),
            ("ack", record.ack_delivered),
            ("payload", record.payload_delivered),
        )
        if shown
    ]
    return (
        str(record.task_position),
        record.attempt_id,
        record.state,
        _optional(record.seal_ordinal),
        "-" if record.score is None else f"{record.score:.3f}",
        _optional(record.final_failure),
        _optional(record.terminal_source),
        _optional(record.decode_state),
        " ".join(delivered) or "-",
        _optional(record.payload_policy),
        record.profile,
        _optional(record.payload_resolution_source),
    )


def _optional(value: Any) -> str:
    return "-" if value is None else str(value)


def _line(cells: Tuple[str, ...], widths: List[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()
