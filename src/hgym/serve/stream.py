"""Serve a *queue* of env tasks over one MCP endpoint.

:class:`~hgym.serve.episode.ServedEpisode` serves one episode. :class:`TaskStream` is the
next abstraction up: it holds a materialised queue of ``(env, task_idx)`` refs, hands out one
task at a time, routes the env's native tool calls to the episode that task is running in,
**seals and scores that episode itself** (never the agent), and appends one provenance row per
dispensed task.

The stream owns scoring, so what it hands out is deliberately *redacted*: a
:class:`DispensedTask` carries ``{env, instructions, budget, tools}`` and has no field that
could hold the task index or the target.

The same rule holds for the *whole* agent-visible surface, not just the framing. A queue may
repeat a task index, so anything that identifies a task is a correlation channel: learn an index
once and a later occurrence can be recognised and replayed against a scorer. So the
:class:`ResultRow` a seal produces — its lease, queue position, index and the env's raw
feedback — stays in ``results.jsonl`` and :attr:`TaskStream.results`, and a terminating call
tells the caller only that the task ended. For the same reason the server masks tool exceptions:
an env that raises while loading a task can name it in the exception text, and MCP would
otherwise relay that verbatim.

Failures are redacted on that same boundary and reported on the other one. Everything that can
go wrong while a task ends — a row that cannot be recorded, a summary the record cannot
headline, an env that raises on its way out — answers the agent with that same fixed payload.
An exception reaching a ``tools/call`` is a *different shape* from a result, and a shape that
varies with the outcome is the verdict channel again by another route: an env that publishes a
clean summary on one verdict and a malformed one on the other would hand the agent its verdict
through whether the call succeeded, without one byte of the payload changing. So the failure
goes where every other one here goes — the row already in ``results.jsonl``, the stream's
:attr:`~TaskStream.stopped` flag, and the exception :meth:`TaskStream.aclose` raises — and
never to the caller. Loud to the harness, silent to the agent.

The same holds one call along. A stop refuses every later dispense, so an agent's next
``get_task`` would raise where an unstopped run hands back a task: the identical leak, one call
later, which is why redacting only the terminating call would not have closed it. Over MCP a
stopped stream therefore answers ``get_task`` with the ``{"done": true}`` an exhausted queue
gets — the run is over either way, and which of the two it was is the harness's business. The
direct API still raises, so a harness driving the stream itself loses nothing. What no response
can hide is that the run ended: a stop truncates the queue, and an agent counting the tasks it
was promised can see that. That residue is one bit, delivered at the moment the agent has no
task left to spend it on, which is the most a stream that refuses to keep scoring can offer.

That answer is about the *queue* and about nothing else. ``done`` says no further task is
coming; it does not say the caller has finished the ones it already holds, and the ``in_flight``
beside it is reported rather than composed. A worker that is told nothing of its own is open
stops, and the task it was still entitled to answer is force-scored as an ordinary loss at the
drain — a wrong result with an earned-looking closure, which is precisely what the redaction
exists to prevent elsewhere. Nothing is hidden by reporting it: a stop seals no episode by
itself, so this count is the same one an exhausted queue shows, and ``queue_info`` answers it on
demand anyway. Which is also why an exhausted pull displaces nothing at all — a question about
the queue may not end work the agent is still entitled to finish.

That last point is deliberately *stricter* than a single served episode, which surfaces
episode-level feedback on the terminal result (:func:`~hgym.feedback.wire.select_inband`) and
returns the env's own terminal payload with it. That rule draws its line at the end of the
interaction: once the episode is over, what the agent learns there cannot reach its own
behaviour. A stream's terminal is *mid-run* — the next ``get_task`` can hand back an index
already played — so the identical principle, applied at the boundary this object owns, redacts
instead of surfacing. Both channels are closed together: neither the env's terminal response nor
the feedback sidecar it rides on crosses a terminating call, because ``correct`` is equally a
verdict in either. Whether the env's terminal response happens to *be* a verdict is not
something the serving layer can tell, so the boundary is the call, not the payload.

Ownership: ``env_for`` is a **factory**, not a shared instance. Each episode gets its own env
and closes it, because ``ServedEpisode.close()`` closes its env and ``ToolUsingEnv.close()``
ends *every* session that env tracks — a shared instance would let one sealing episode tear
down its siblings.

The tool contract is *frozen*, and frozen means checked. A server publishes one schema per tool
name at startup, so the manifest is read once, from a catalog instance, and every episode after
that is a different instance the factory built. The framing an agent is handed is therefore the
**published** manifest, not the live episode's, and every fresh episode's own manifest is
compared against it before the task is dispensed. A task whose framing disagrees with the
callable surface is not a usability problem — it is an unearned failure, so a stream that finds
one refuses to dispense it and stops rather than scoring the rest of the queue against a
contract the server does not serve.

**Outcomes are typed, and unscored is structural.** Every dispensed task ends in exactly one
:class:`ResultRow` whose ``closure`` says how it ended and whose ``score`` is ``None`` when the
outcome was not earned by the agent, so an infrastructure failure can never be averaged in as a
zero:

===============  ============================================  =========  ===============
``closure``      how it happens                                ``score``  stream continues
===============  ============================================  =========  ===============
``sealed``       the agent called the env's score terminal,     present    yes
                 or exhausted the budget
``aborted``      the agent called the reserved ``terminate``    present    yes
``drained``      the stream forced the terminal — the agent     present    yes
                 pulled the next task, or the stream closed
``timeout``      the per-episode ``deadline`` elapsed           ``None``   yes
``finalize_error`` the terminal transaction failed closed       ``None``   yes
``broker_abort`` dispensed, never sealed (crash / SIGKILL);     ``None``   n/a
                 produced by :func:`reconcile`, not in-process
===============  ============================================  =========  ===============

**Durability.** ``__aexit__`` drains in an orderly way, which cannot cover ``docker rm -f`` or
SIGKILL. So a dispense record is appended and fsync'd *before* the task is handed out; after a
crash, :func:`reconcile` pairs records with results and reports the unmatched ones as
``broker_abort``. That same record is what makes ``resume`` correct: it resumes by **queue
position**, so a queue holding the same task index twice replays both — and a position means
nothing without its queue, so resuming re-checks every recorded position against the queue in
hand and refuses a directory that was written by a different one.

``resume`` is also the *only* way to point a stream at a directory that already holds records.
A fresh stream numbers from the start of its own queue and appends, so it would otherwise file
its rows under positions the file already uses, with nothing on a row to say which run wrote it
— a record that double-counts a task the queue holds once, and that :func:`reconcile` reads as
a crash the run never had, while :attr:`TaskStream.results` shows only the newer run. Continuing
a record and rerunning into one by mistake are the same call otherwise, so the second is refused
at construction, before anything is spent.

A result row is durable before it is anything else — appended and fsync'd *before* it is
published on :attr:`TaskStream.results` — and a stream that cannot write one stops rather than
serving the rest of the queue over a record with a hole in it. Releasing the episode is a
separate concern from recording it, and happens whether or not the row landed.

A row's ``observed`` is the env's own output, verbatim and authoritative: the wire items
exactly as the episode published them — ``{name, value, level[, step]}``, ordered, every
occurrence — which is the form the JSONL trace and :class:`~hgym.evaluate.EvalResult` already
record feedback in. Keyed by name it would be a projection instead: the wire is a list and
permits a name more than once, so a mapping would decide by list order which occurrence
survives and drop the rest, and it would erase the level that says whether a value scored the
task or one step of it.

A :class:`Score`'s ``reward`` and ``success`` are a *summary* of that record, read from the
**episode-level** items, and only ever repeat a value the env already published as a number and
a bool respectively. Nothing is coerced into them — ``bool("false")`` is ``True``, so coercion
is how a record comes to overstate a benchmark — and nothing wrong-typed is quietly dropped
into ``None`` either, since that only trades the overstatement for an equally silent
undercount. A summary value that is of the wrong type, or published more than once, leaves the
row unscored, with ``diagnostic`` saying so, and stops the stream once that row is recorded. So
in a run that finished, a ``None`` inside a :class:`Score` means exactly one thing: the env
published no such field at episode level.

That summary is the *only* thing a published value decides. ``closure`` is classified without
the feedback in hand at all: how a task ended is the harness's own question, answered from the
deadline, the drain, the terminal the episode itself recorded, and the ``finalize_error`` the
core stamps onto its terminal payload. So a value published under a reserved name can be wrong,
or wrong-typed, without being able to reclassify the row it sits on — and the one place where a
published value does become a number validates it rather than coercing it.

**Leases.** With more than one episode live at once, a native call has to say *which* one it
belongs to. Native schemas are ``additionalProperties: false``, so the stream publishes wrapper
schemas that add a required ``lease``, validates it, and **strips it before
``ServedEpisode.call()``** — otherwise the routing capability would be recorded into the env's
trajectory. A lease is opaque, unguessable and never reused, and the registry binds
``(lease, env, native tool)``: a lease that is valid but denotes the wrong thing is refused.
Every refusal is a stream-level result, never an env step, so a routing mistake costs no budget
and enters no trajectory. A lease outlives the task it named — a call arriving late is told the
task is over, not that its lease was never real — but *only* the lease does: the registry entry
is retired the moment its seal is finished, so what a completed task leaves behind is its row and
a 32-character string, not the env, trajectory and sessions it ran on. "Never reused" spans the
whole *record*, not the process: a resumed run seeds the issued set from the dispenses already in
the directory, because a repeat there is one a result would silently answer twice (see
:func:`reconcile`).

A wrapper is only published for a schema the added argument is provably sound for — a plain root
object schema. A root that could reinterpret the addition (a ``$ref`` the arguments really live
behind, a composed ``allOf``/``oneOf``, a constraint on the object's names or size) is refused at
construction instead, because the *episode* keeps enforcing the env's own schema: advertising a
contract the seal does not enforce is how a call that conforms to the published schema becomes a
clean, earned-looking loss (see :func:`_leased_manifest`).
At ``max_in_flight == 1`` there is no lease and no wrapper, and native
tool registration *and* the arguments an env receives are exactly what they were: with one slot
every call is unambiguous, so nothing is routed on and nothing is taken out of ``arguments`` —
including an argument an env legitimately names ``lease`` itself.

Drive it directly (``get_task`` / ``dispatch`` / ``queue_info``) or wrap it for an agent with
:func:`build_stream_server`. The direct API needs no MCP at all, which is what makes the
lifecycle testable.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
)

from fastmcp import FastMCP
from fastmcp.tools import ToolResult

from hgym.core import Env
from hgym.serve.episode import ServedEpisode
from hgym.serve.server import build_tool
from hgym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from hgym.task import TaskSpec, ToolManifest

__all__ = [
    "Closure",
    "CompletedTask",
    "DispensedTask",
    "Provenance",
    "ProvenanceError",
    "ProvenanceSpan",
    "QueueInfo",
    "ResultRow",
    "Score",
    "TaskRef",
    "TaskStream",
    "build_stream_server",
    "reconcile",
    "read_dispenses",
    "read_results",
]

# Control tools the stream itself adds. An env tool of either name would be silently replaced
# by FastMCP, so a collision is rejected when the server is built.
_GET_TASK_TOOL = "get_task"
_QUEUE_INFO_TOOL = "queue_info"

# The *entire* response to a terminating call — the same bytes for every env, every task and
# every outcome. Serialized once, at import, so that invariance is structural rather than a
# convention the next edit could quietly break: nothing about the sealed episode can be read
# off a constant, including from which keys are present.
_TASK_OVER = json.dumps(
    {
        "content": "<task ended; the stream recorded the outcome>",
        "terminated": True,
        "hint": f"task over; call `{_GET_TASK_TOOL}` for the next one",
    }
)

# Feedback names the stream reads a headline score from, in order. Everything an env emits is
# kept verbatim, so these two are only the summary fields — and a missing one stays absent
# rather than becoming a fabricated zero. Read strictly — see `_pick_summary`.
_REWARD_NAMES = ("reward", "partial_credit")
_SUCCESS_NAMES = ("success", "correct")

# Every feedback name this record gives a meaning of its own, whether it reads one from
# feedback (the summary names above) or owns the name outright (`finalize_error`, which the
# core stamps and `_finalize_failed` takes only from that stamp). A value published under one
# of these is the one kind of env output that could move a number or a closure, and
# `EpisodeFeedbackValue` legally admits text and numbers — `bool("false")` is `True` — so none
# of them may ever be read by truthiness. `_pick_summary` is the single funnel: it is the only
# place a published value becomes a decision, and it validates rather than coerces. This tuple
# is what the guard test enumerates, so a name added here is covered by it automatically and a
# name that gets a meaning without being added here is the gap to look for.
_RESERVED_FEEDBACK_NAMES = (*_REWARD_NAMES, *_SUCCESS_NAMES, "finalize_error")

# The wrapper argument that names an episode. Env tool schemas are `additionalProperties: false`,
# so it has to be added to the published schema rather than smuggled alongside it.
_LEASE_ARG = "lease"

# How many times a dispense will redraw before it calls the source, rather than the draw, the
# problem (see `TaskStream._mint_lease`). Any bound at all is the point: two independent 128-bit
# draws colliding is not something a run meets, so a handful of repeats says the values are not
# independent, and the next thousand draws would say the same thing with the event loop held.
_LEASE_MINT_ATTEMPTS = 8

# The root keywords a schema may carry and still be one the lease can be added to (see
# `_leased_manifest`). An allow-list rather than a list of hazards: JSON Schema keeps gaining
# keywords, a keyword this module has never heard of is one it cannot have proved anything about,
# and the safe answer to an unknown constraint on the object being extended is to refuse it.
#
# Each of these is here for a reason of its own. `type`/`properties`/`required`/
# `additionalProperties` are the object vocabulary the wrapper reads and rewrites — adding a name
# to `properties` is exactly what makes a closed object accept it. `$defs`/`definitions` is a
# container of subschemas that constrains nothing itself; what it holds is reached from inside
# `properties`, which this wrapper does not touch. The rest are annotations: they carry no
# assertion about an instance in any dialect, so an added property cannot violate one.
_WRAPPABLE_ROOT_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "$defs",
        "definitions",
        "$schema",
        "$id",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)

_RESULTS_FILE = "results.jsonl"
_DISPENSES_FILE = "dispenses.jsonl"

# How far back a log is read to find its last committed record. Only a file whose final append
# died partway is scanned at all, and only until the first terminator, so this bounds what a
# repair reads rather than how long a record may be.
_TAIL_SCAN_BYTES = 1 << 16

# How the stream classifies the end of a dispensed task. See the table in the module docstring
# for which of these carry a score.
Closure = Literal[
    "sealed",
    "aborted",
    "drained",
    "timeout",
    "finalize_error",
    "broker_abort",
]

# The closures whose outcome the agent earned. Everything else records `score=None`, so an
# infrastructure failure is structurally unaggregatable rather than an earned zero.
_SCORED_CLOSURES = frozenset({"sealed", "aborted", "drained"})

# Where a task's `terminal_error` came from: a terminal the *stream* drove and the env failed on,
# or a call the *agent* made that never reached a result and was promoted because the stream then
# had to end the task itself (see `_Live.call_error`). Both leave the same unscored row and say
# different things about the run — only the first is a reason to stop serving the queue.
_TerminalErrorSource = Literal["terminal", "promoted_call"]


class _MalformedSummary(ValueError):
    """An env published a summary-named feedback value this record cannot honestly headline."""


class TaskRef(NamedTuple):
    """One queue entry: which env, which task index. Repeats are legal — the queue is a
    sequence, and a task's identity within a run is its *position*, not its index."""

    env: str
    task_idx: int


@dataclass(frozen=True)
class DispensedTask:
    """What the agent receives: enough to act, and nothing that identifies the task.

    Redaction is structural — there is no field the task index, the target, or the queue
    position could be written into."""

    env: str
    instructions: str
    budget: Optional[int]
    tools: Tuple[Dict[str, Any], ...]
    # The capability that names this episode, present only when more than one may be live. It
    # identifies the episode, never the task: it is random, and a new one is minted per dispense.
    lease: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        wire: Dict[str, Any] = {
            "env": self.env,
            "instructions": self.instructions,
            "budget": self.budget,
            "tools": [dict(tool) for tool in self.tools],
        }
        if self.lease is not None:
            wire["lease"] = self.lease
        return wire


@dataclass(frozen=True)
class QueueInfo:
    """Where the queue stands: how many tasks are left, how many were handed out, how many are
    live right now."""

    remaining: int
    consumed: int
    in_flight: int

    def to_wire(self) -> Dict[str, Any]:
        return {
            "remaining": self.remaining,
            "consumed": self.consumed,
            "in_flight": self.in_flight,
        }


@dataclass(frozen=True)
class Score:
    """An **earned** outcome, read off the sealed episode's own evidence.

    ``reward`` and ``success`` are the headline numbers when the env publishes them (a missing
    one stays ``None`` rather than becoming a zero, and a wrong-typed one is never coerced into
    either — see :func:`_pick_summary`); ``feedback`` is everything the env emitted, verbatim."""

    reward: Optional[float]
    success: Optional[bool]
    feedback: List[Dict[str, Any]]

    def to_wire(self) -> Dict[str, Any]:
        return {
            "reward": self.reward,
            "success": self.success,
            "feedback": [dict(item) for item in self.feedback],
        }


@dataclass(frozen=True)
class ResultRow:
    """One dispensed task's outcome — exactly one per dispense.

    ``score`` is ``None`` unless the closure was earned *and* the env's headline was readable,
    so aggregating ``score`` can never silently average in an infrastructure failure or a
    coerced verdict. ``observed`` keeps every item the env emitted, in wire form, for audit even
    on an unscored row; it is evidence, never a score. ``extensions`` is namespaced provenance
    from :class:`Provenance` extensions, never merged into the fields above.

    Each entry in ``extensions`` carries what one extension observed, and its members say which
    halves of the span were recorded:

    ============================  ==========================================================
    ``extensions[namespace]``     what it means
    ============================  ==========================================================
    ``{dispensed, sealed}``       both halves ran; ``sealed`` is what ``finalize`` returned
    ``{dispensed, error}``        ``finalize`` failed; ``error`` renders its failure
    ``{dispensed}``               nothing was recorded after the dispense — the row came from
                                  :func:`reconcile`, so ``closure`` is ``broker_abort``
    ============================  ==========================================================

    ``dispensed`` is always present, because a span that would not open refuses the dispense
    outright. So a row written in-process always has exactly one of ``sealed`` or ``error``
    beside it, and only a reconciled row has neither — but the discriminator a consumer should
    read is ``closure``, which is typed and says the same thing about the whole row."""

    seq: int
    lease: str
    position: int
    env: str
    task_idx: int
    closure: Closure
    score: Optional[Score]
    observed: List[Dict[str, Any]] = field(default_factory=list)
    diagnostic: Optional[str] = None
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_wire(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "lease": self.lease,
            "position": self.position,
            "env": self.env,
            "task_idx": self.task_idx,  # provenance only — never returned to the agent
            "closure": self.closure,
            "score": self.score.to_wire() if self.score is not None else None,
            "observed": [dict(item) for item in self.observed],
            "diagnostic": self.diagnostic,
            "extensions": dict(self.extensions),
        }

    @classmethod
    def from_wire(cls, row: Dict[str, Any]) -> "ResultRow":
        score = row.get("score")
        return cls(
            seq=int(row["seq"]),
            lease=str(row["lease"]),
            position=int(row["position"]),
            env=str(row["env"]),
            task_idx=int(row["task_idx"]),
            closure=row["closure"],
            score=(
                Score(
                    reward=score.get("reward"),
                    success=score.get("success"),
                    feedback=[dict(item) for item in (score.get("feedback") or [])],
                )
                if isinstance(score, dict)
                else None
            ),
            observed=[dict(item) for item in (row.get("observed") or [])],
            diagnostic=row.get("diagnostic"),
            extensions=dict(row.get("extensions") or {}),
        )


class ProvenanceError(RuntimeError):
    """An extension failed to open its span, so the task was not handed out."""


@dataclass(frozen=True)
class CompletedTask:
    """What an extension learns about the task it spanned: where it sat in the queue, how it
    ended, and what it scored.

    Redacted — an extension never receives the row, the episode, the env, the lease registry, or
    the target — and **detached**: this is built for one extension, from the summary re-parsed
    out of its own JSON, and is not the object the row is written from.

    ``frozen=True`` alone would not have been enough, and reading it as immutability is the
    mistake worth naming. It stops an attribute being rebound; it says nothing about the list
    behind :attr:`Score.feedback` or the dicts inside it. The row's ``observed`` and its
    ``Score.feedback`` are one list, so an extension handed the row's own summary could
    ``clear()`` it and rewrite both authoritative fields — or append a value that is not JSON
    and take the whole row down with the append. Detaching is what makes the guarantee true:
    whatever an extension does to what it is handed reaches nothing but its own copy, and not
    the next extension's either."""

    position: int
    closure: Closure
    score: Optional[Score]


class ProvenanceSpan(Protocol):
    """One extension's observation of one *dispense*.

    A span exists because ``TaskRef`` is not a dispense identity: the queue may hold the same
    task index twice, so a before/after pair keyed on the ref could not be correlated. The span
    is the correlation."""

    @property
    def dispensed(self) -> Dict[str, Any]:
        """What was observed before the task was handed out. Strict JSON.

        Read once, and made durable with the dispense record itself, so it reaches the row
        whether the task is sealed or the process is killed holding it."""
        ...

    async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
        """What was observed after the task was sealed, scored and classified. Strict JSON."""
        ...


class Provenance(Protocol):
    """An extension that records something extra about every dispensed task.

    Its output is nested under ``ResultRow.extensions[namespace]`` and never merged into the
    authoritative fields, so an extension can add to a row but never rewrite one."""

    namespace: str

    async def begin(self, ref: TaskRef) -> ProvenanceSpan:
        """Open a span for one dispense, before the task is exposed."""
        ...


@dataclass
class _Live:
    """A dispensed task: its episode and the bookkeeping needed to seal and record it."""

    lease: str
    seq: int
    position: int
    ref: TaskRef
    episode: ServedEpisode
    # The agent's clock, stamped where the task is exposed rather than where the entry is built:
    # everything between the two is the harness recording the dispense, and a deadline that
    # counted it would spend the agent's budget on storage the agent cannot see or wait out.
    # An entry only reaches the watchdog after the stamp, so the placeholder is never compared.
    started: float = 0.0
    sealed: bool = False
    # This task has been *ended* — its seal has begun. Unlike the claim above, this never becomes
    # false again: a seal that failed on the storage hands its claim back so a later drain can
    # retry the append (see `TaskStream._join_seal`), and by then the episode has been
    # force-terminated, the row composed and every span finalized. Whether a call may still be
    # routed here is a question about *that*, not about who currently holds the claim — read the
    # claim for it and the one task whose record already failed becomes the one task a late call
    # is let into, answered as though its call had ended something (see
    # `TaskStream._resolve`).
    ended: bool = False
    # Letting this task's episode go, held as a task. The entry outlives its own release — a seal
    # publishes the stop it owes afterwards — so more than one caller can reach it; the first
    # claims it and the rest await this rather than closing the episode again.
    releasing: Optional["asyncio.Task[None]"] = None
    row: Optional[ResultRow] = None
    # The tool that actually ended the task, so the stream knows whether the agent ended it
    # itself and how. Written by the one call that entered the terminal — never by a call the
    # episode tombstoned, which is `terminated` without having ended anything — and otherwise
    # taken from the episode at the seal, for a terminal whose own caller never came back.
    terminal_tool: Optional[str] = None
    # The core's sanitized terminal payload, always read off the episode: it carries the
    # `finalize_error` stamp the closure is classified from, and no env-supplied content may
    # stand in for it (see `_finalize_failed`).
    terminal_payload: Optional[Dict[str, Any]] = None
    # Set when a terminal the *stream* drove failed after ending the task, or could not end it
    # at all. Harness-owned — the exception a forced call raised, never anything the env
    # published — so the classifier may read it without reopening the channel `_classify`
    # closes. The row lands unscored with a diagnostic, and the stop follows the release, like
    # `summary_error` and for the same reason. A `call_error` promoted into this field is the one
    # that does not stop: it is the same finding about the row and a different one about the run.
    terminal_error: Optional[BaseException] = None
    # Which of those two this is. Kept as state because it cannot be read back off the exception:
    # one instance may be raised any number of times, so an env that raises a single object on a
    # call the agent made and raises that same object again on the terminal the stream drives
    # would have a genuine terminal failure mistaken for the promotion below — a row saying the
    # agent's call was lost when the env in fact failed on its way out of the task, and no stop,
    # so the rest of the queue is served against an env that ends no task at all.
    terminal_error_source: Optional[_TerminalErrorSource] = None
    # Set when a call the *agent* made failed before the episode had ended: the harness never got
    # a result out of it, so what the agent asked for is nowhere on this episode's record.
    # Recorded rather than acted on, because a mid-call failure is not yet an outcome — a task the
    # agent goes on to end itself is the agent's, whatever failed on the way there (see
    # `dispatch`). It becomes this entry's `terminal_error` only if the *stream* ends up forcing
    # the terminal, which is the case where a lost call would otherwise be recorded as a task the
    # agent played out and got wrong.
    call_error: Optional[BaseException] = None
    # Set when the env's headline could not be read off this task's feedback. The row still
    # lands (unscored, with a diagnostic); the stop is raised *after* the release, because a
    # row that landed makes the seal final and this entry is the only handle on its episode.
    summary_error: Optional[_MalformedSummary] = None
    # The single per-task finalization transition: the seal itself, held as a task. Whoever
    # creates it owns the seal and everyone else awaits *that* task rather than starting their
    # own, so a terminal call, the deadline and the drain — which all race for this — produce
    # one classification, one row and one finalize per span however their callers are cancelled.
    sealing: Optional["asyncio.Task[ResultRow]"] = None
    # The row this task's seal composed, retained from the moment it is built until the durable
    # append that commits it returns. A seal that failed hands its claim back so a later drain can
    # retry — and what is retryable is the *append*, nothing above it. Composing a second time
    # would call every extension's `finalize` again, and would re-read an episode the first
    # attempt already force-terminated, filing a task the stream drained as one the agent ended
    # itself. So the retry picks this up and goes straight to the write (see `_run_seal`).
    pending_row: Optional[ResultRow] = None
    # namespace -> the open span, and what it observed at dispense.
    spans: Dict[str, ProvenanceSpan] = field(default_factory=dict)
    dispensed_extensions: Dict[str, Any] = field(default_factory=dict)

    @property
    def released(self) -> bool:
        """This task's episode has been let go: its env is closed, its MCP sessions are gone,
        and nothing will ever call into it again.

        The release is claimed once and *is* the task (see :meth:`TaskStream._release`), so this
        asks whether that task has finished rather than whether one was started — an episode
        still closing is still holding what it holds.

        The entry can outlive this. A seal that failed on the storage keeps its composed row and
        hands its claim back so a later drain can retry the *append*, and the drain releases the
        episode in the meantime because a failed seal is not retried above the write. What is
        left then is a row still owed to the file and nothing else — which is not an episode in
        flight, and :meth:`TaskStream.queue_info` may not report it as one."""
        return self.releasing is not None and self.releasing.done()

    @property
    def settled(self) -> bool:
        """Nothing is owed for this task any more: its row is durable **and** the seal that
        landed it has finished — the episode released, and whatever stop the row implies
        recorded on the stream.

        A row on its own does not say that. The append commits it in the middle of the seal,
        with the episode still open behind it and a stop it may owe still unpublished, so an
        entry read as finished at that moment lets the queue move past a task that is still
        ending: the next dispense takes a slot the previous episode has not let go of, and it
        takes it over a stop nobody has recorded yet (see :meth:`_run_seal`).

        Neither does the claim on its own. A seal that failed, or one whose owner cancelled it,
        is ``done()`` with no row, and that task is still owed one — the claim is handed back
        for a later drain to retry."""
        return self.row is not None and self.sealing is not None and self.sealing.done()

    def failed_to_end(self, exc: BaseException, source: _TerminalErrorSource) -> None:
        """Remember that this task has no verdict behind it, and where that came from.

        The first failure explains the run, so a later one never overwrites it. The source is
        written *here*, beside the exception, because the two must never be able to disagree:
        a caller that set one without the other would leave the classifier and the stop reading
        a source that belongs to some earlier failure."""
        if self.terminal_error is None:
            self.terminal_error = exc
            self.terminal_error_source = source


@dataclass(frozen=True)
class _Stopped:
    """Why the stream stopped serving, and what to say about it.

    Two messages, because one failure is reported at two boundaries: the next dispense refuses
    with ``dispensing``, and :meth:`TaskStream.aclose` raises ``closing`` — the only place a
    stream driven entirely over MCP can report anything, since the harness never calls
    ``get_task`` itself and a tool exception is masked on its way to the agent."""

    cause: BaseException
    dispensing: str
    closing: str


def _wire_json(value: Any) -> str:
    """``value`` as the endpoint would put it on the wire — **proved**, not assumed.

    Two things the type of a value does not settle. Strict JSON is the first: ``json.dumps``
    accepts ``NaN`` and ``Infinity`` by default and writes them as bare tokens no JSON parser is
    obliged to read, so a schema carrying one is advertised as something else entirely (FastMCP
    sends ``null``) and the episode then refuses the value it advertised. UTF-8 is the second: a
    Python ``str`` may hold an unpaired surrogate, which is text to every ``isinstance`` check and
    a ``UnicodeEncodeError`` the moment a transport encodes it — and ``ensure_ascii`` would hide
    exactly that by escaping it into ASCII, so the encode is done the way an endpoint does it.

    Cheap to prove and expensive to assume: what fails here fails before anything is spent, and
    what fails later fails after a task is out and its row is owed an outcome."""
    text = json.dumps(value, allow_nan=False, ensure_ascii=False)
    text.encode("utf-8")
    return text


def _require_task_ref(ref: Any) -> TaskRef:
    """One queue entry, checked to be the identity it will be *recorded* as.

    A ``NamedTuple`` annotation is documentation, not validation, so anything at all reaches the
    queue — and these two fields are identity, not payload. They name the env whose rows are
    filed under them, and they are written into every row this run records. Nothing downstream
    re-checks them, but the episode *coerces* one: ``ServedEpisode.open_env`` takes ``int(task)``
    to load a task, while the row is appended carrying the caller's own value. A ``1.9`` therefore
    plays task 1 and is recorded as ``1.9`` — a record whose identity is not the task that ran,
    and one that no later reading can repair.

    Coercing here would only move the disagreement earlier: the caller asked for something this
    queue cannot hold, and a canonical identity invented after the fact is what put a number in
    the file that nothing else agrees with. So exact ``str`` and exact ``int`` — subclasses
    included, because a subclass is a value with its own ``__eq__`` sitting in a field every later
    comparison runs on, and the wire form is a plain scalar either way."""
    try:
        env, task_idx = ref
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"a queue entry must be a (env, task_idx) pair, got {type(ref).__name__}"
        ) from exc
    if type(env) is not str or not env:
        raise ValueError(
            f"a queue entry's env must be a non-empty string, got {env!r} "
            f"({type(env).__name__}); it is the key this run's rows are filed under"
        )
    if type(task_idx) is not int:
        # `bool` is an `int` subclass, and `True` would index task 1 while being recorded as a
        # boolean — the same disagreement in a shape `isinstance` would wave through.
        raise ValueError(
            f"a queue entry's task index must be a whole number, got {task_idx!r} "
            f"({type(task_idx).__name__}); it is written to every row, and what the queue plays "
            "has to be what the record says it played"
        )
    return TaskRef(env, task_idx)


def _frozen_manifest(env_name: str, tools: Sequence[ToolManifest]) -> List[ToolManifest]:
    """An env's own manifest as the wire carries it — the canonical copy everything this stream
    does with that contract is derived from.

    Frozen as soon as the contract has been validated, and before anything is *derived* from it.
    ``model_copy`` is shallow and ``describe`` hands back the env's own objects, so without this
    the signature the drift check compares, the score terminal the stream drives and every task's
    framing would all be the env's values, each one running the env's code at a moment nothing is
    guarding.

    That is not a leak of decoration: those are load-bearing paths. The score terminal's name is
    what this stream calls to end a task the agent did not, and the episode finds that terminal by
    looking the name up — so a name whose own ``__hash__`` answers differently once the run is
    under way makes the stream's own terminal uncallable. The abort answers instead, and the task
    is recorded as an ordinary wrong answer, with nothing anywhere saying a name was involved.

    Proved through the endpoint's own encoder (see :func:`_wire_json`), because a contract this
    endpoint cannot send is one it cannot serve, and refused here — at construction, before an
    episode exists and before any task has been handed out."""
    frozen: List[ToolManifest] = []
    for manifest in tools:
        try:
            wire = json.loads(
                _wire_json(
                    {
                        "name": manifest.name,
                        "description": manifest.description,
                        "input_schema": manifest.input_schema,
                    }
                )
            )
        except (ValueError, TypeError, UnicodeError) as exc:
            raise ValueError(
                f"env {env_name!r} advertises a tool this endpoint could not put on the wire "
                f"({_rendered_failure(exc)}); a tool's name, description and argument schema are "
                "what a server registers and what every task's framing carries, so they have to "
                "be values the endpoint can send"
            ) from exc
        for part in ("name", "description"):
            if not isinstance(wire[part], str):
                raise ValueError(
                    f"env {env_name!r} advertises a tool whose {part} is "
                    f"{type(wire[part]).__name__}, not text; a tool's name and its description "
                    "are what a server registers and what every task's framing carries, and both "
                    "are text on the wire"
                )
        frozen.append(manifest.model_copy(update=wire))
    return frozen


def _tool_signature(manifest: ToolManifest) -> Tuple[Any, ...]:
    """Everything about one tool that a server freezes when it registers it: the name it is
    called by, the description the model reads, the schema its arguments are validated against,
    and its role in the terminal lifecycle."""
    return (
        manifest.name,
        manifest.description,
        manifest.terminal_kind,
        json.dumps(manifest.input_schema, sort_keys=True),
    )


def _manifest_signature(tools: Sequence[ToolManifest]) -> Tuple[Any, ...]:
    """The part of a tool manifest a server freezes at startup: every tool's name, description,
    argument schema and terminal kind. Two task ids whose signatures differ cannot be served by one
    endpoint.

    The description is included because it is advertised once, at registration, from whichever task
    was inspected first — so a per-task description would leave the published tool contract
    disagreeing with what a later task's framing shows the agent."""
    return tuple(sorted(_tool_signature(m) for m in tools))


def _manifest_drift(published: Sequence[ToolManifest], fresh: Sequence[ToolManifest]) -> str:
    """Name the tools two manifests disagree about, for an error a maintainer can act on.

    The names are the env's, and naming them runs the env's ``repr`` — so they go through the
    guarded funnel like every other value this module writes down about a third party. This
    string is an *argument* to the stop the drift owes (see
    :meth:`TaskStream._require_published_manifest`): unguarded, a name that cannot be described
    replaces the refusal with its own exception and the stop is never published, so the queue is
    served on against an env whose episodes disagree with the contract the endpoint registered
    and ``aclose()`` reports a clean run. See :func:`_described`."""
    was = {m.name: _tool_signature(m) for m in published}
    now = {m.name: _tool_signature(m) for m in fresh}
    parts = []
    if added := sorted(set(now) - set(was)):
        parts.append(f"added {_described(lambda: repr(added))}")
    if removed := sorted(set(was) - set(now)):
        parts.append(f"removed {_described(lambda: repr(removed))}")
    if changed := sorted(name for name in set(was) & set(now) if was[name] != now[name]):
        parts.append(f"changed {_described(lambda: repr(changed))}")
    return "; ".join(parts) if parts else "the two manifests differ"


_UNRENDERABLE = "<unrenderable>"


def _failure_type(exc: BaseException) -> str:
    """The class name of a failure, when even *that* runs code this module did not write.

    ``__name__`` is an attribute of the class, and a metaclass may define it as a property, so
    this is guarded exactly like the message below and for the same reason. A name that comes
    back as something other than a string is refused rather than formatted, since formatting it
    is the call this exists to avoid making."""
    try:
        name = type(exc).__name__
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:  # noqa: BLE001 — see `_rendered_failure`
        return _UNRENDERABLE
    return name if isinstance(name, str) else _UNRENDERABLE


def _rendered_failure(exc: BaseException) -> str:
    """Describe a failure an env or an extension supplied, without running its code unguarded.

    Every message this module builds about a failure it has *caught* formats that failure — and
    formatting an exception runs code belonging to whoever raised it, a second time and outside
    the ``except`` that just contained it. ``__str__`` is theirs, and an accident is enough: a
    message built lazily from state that is gone by the time it is asked for raises here rather
    than at the raise site. The second exception is not the one the handler caught, so it does
    not stay caught — it walks out of the handler carrying the handler's job with it. Measured,
    at three sites in this module: the row a sealed episode is owed is never composed, the stop a
    failed terminal transaction owes is never published (so the stream serves the rest of the
    queue against the env that failed and reports a clean run), and the redacted constant a
    terminating call answers with becomes a traceback at the agent. A failure this module has
    already decided to contain may not be un-contained by the act of writing it down.

    So: the message is attempted, and on failure the type alone, and on failure a constant. What
    is never attempted twice is the caller's code — a fallback that formats the same object again
    would be the same bug one line down.

    ``CancelledError`` is caught here rather than let through. Nothing in this function awaits,
    so no cancellation can be *delivered* during it; one raised here was raised by the object
    being rendered, and letting that through would strand the seal exactly as a ``finalize`` that
    raises one does. ``SystemExit`` and ``KeyboardInterrupt`` still propagate, which is the line
    this module already holds for the callbacks themselves: an interpreter-level signal costs the
    row loudly rather than being swallowed inside a diagnostic."""
    name = _failure_type(exc)
    try:
        return f"{name}: {exc}"
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:  # noqa: BLE001 — a contained failure may not escape through its message
        return f"{name}: <unrenderable message>"


def _described(render: Callable[[], str]) -> str:
    """The env's own value, written into a refusal that is *about* that value — or, when asking
    for it raises, the failure that asking raised, through :func:`_rendered_failure`.

    ``repr`` is the env's code on the env's object, and this module calls it only while building
    the message that refuses the value. So the value gets to decide whether the refusal happens
    at all: unguarded, the row a sealed episode is owed is never composed and the stream reports
    a *storage* failure — the one thing that did not happen — with nothing on disk naming the
    env. The refusal is the point and the description is the decoration, so the description is
    what gives way.

    A ``str``/``int``/``float`` subclass is the reachable shape: the feedback wire accepts one
    (``isinstance``), and while the models coerce it away at construction they do not validate on
    assignment, so a post-construction mutation carries it through verbatim — the hole
    ``dump_item``'s own docstring names."""
    try:
        return render()
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 — the refusal outranks its own decoration
        return f"<a value this record cannot describe: {_rendered_failure(exc)}>"


class _Cancellation:
    """Whether cancellation has been requested against **this task** since this was taken.

    Two unrelated things arrive as ``asyncio.CancelledError``, and they are the same exception:
    this task being cancelled, and code this task *called* raising one — directly, or by awaiting
    something that was cancelled underneath it. Nothing about the object tells them apart.
    ``Task.cancelling()`` does: only the first moves it. So the count is taken before the calls
    and compared after each one.

    Taken once for a whole boundary rather than per call, because a request against this task
    outlives whichever callee it happened to interrupt: a callee that catches one and returns
    anyway must not be able to reset it for the callees after it. The bias is deliberate and it
    is the safe one — an outstanding request keeps counting until the boundary ends, so a
    doubtful case propagates rather than containing.

    A count rather than a flag, because ``asyncio.timeout`` expires a bound by cancelling this
    task and then calling ``uncancel()``: the count moves and comes back on a path that is
    neither of the two cases.

    With no task to cancel — sync code, or a bare callback — there is nothing a cancellation
    could have been *requested against*, so the conservative reading is kept and this reads as
    requested. A boundary that knows nothing can be delivered to it says so by passing ``None``
    to :func:`_must_propagate` instead of taking one of these."""

    __slots__ = ("_task", "_count")

    def __init__(self) -> None:
        task: "Optional[asyncio.Task[Any]]"
        try:
            task = asyncio.current_task()
        except RuntimeError:  # no running loop at all
            task = None
        self._task = task
        self._count = task.cancelling() if task is not None else 0

    def requested(self) -> bool:
        return self._task is None or self._task.cancelling() > self._count


def _must_propagate(exc: BaseException, cancellation: Optional[_Cancellation]) -> bool:
    """Whether a **containment boundary** must let ``exc`` out.

    A containment boundary is one whose contract is that nothing escapes it: it guards a call
    into an env or an extension whose failure is not the caller's outcome — a teardown, or a
    failure this module classifies and records rather than raises. Those boundaries catch
    ``BaseException`` and ask this. Everywhere else in this module an ``except Exception`` is the
    opposite thing, a deliberate passthrough: it guards this module's *own* coroutine, run on the
    caller's behalf and in the caller's task, where a ``CancelledError`` is that caller's own
    cancellation and has to end it. Which of the two a handler is follows from whose code can
    raise inside it, and that is the whole rule.

    At a containment boundary a ``CancelledError`` is third-party code failing, no different in
    kind from any other exception it raises — unless cancellation was requested against this task
    while the call was running, which is the one case where the exception belongs to whoever
    asked for it rather than to the callee. ``cancellation`` is ``None`` for a boundary with no
    ``await`` in it: nothing can be delivered to a task that never suspends, so one observed
    there was raised where it was observed.

    ``SystemExit`` and ``KeyboardInterrupt`` always propagate — the line this module already
    holds for the callbacks themselves, since swallowing an interpreter-level signal is worse
    than losing a row — and so does every other ``BaseException``, which an ``except Exception``
    would not have caught either."""
    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
        return True
    if isinstance(exc, asyncio.CancelledError):
        return cancellation is not None and cancellation.requested()
    return not isinstance(exc, Exception)


def _append_jsonl(path: Path, record: Dict[str, Any], *, durable: bool = False) -> None:
    """Append one JSON line. ``durable`` flushes and fsyncs it, for a record whose whole point
    is to survive the process that wrote it — and, the first time the file is created, fsyncs
    the directory too, since an unsynced directory entry can lose the whole file. A directory
    the write has to create is made reachable the same way, by :func:`_mkdir_durable`.

    **The terminating newline is this log's commit boundary, so a durable record is synced
    before the byte that commits it is even written.** Written together they carry no ordering:
    a row is one ``write`` — 46 KB of published feedback is still one — and a crash is free to
    persist the block holding the newline while losing one in the middle, so a torn write would
    read back as a committed record. Split, the terminator can only be on disk if everything it
    terminates already was, which is the whole of what :func:`_read_jsonl` infers from it: an
    unterminated tail is a write that never returned, and anything else that will not parse is
    corruption of a record something downstream was told about.

    **An append that raises leaves the log exactly as it found it.** That ordering settles what a
    *crash* may commit; an error settles nothing on its own, because the last fsync can fail with
    the terminator already flushed — visible, so every reader takes the record as committed,
    while the caller was told the record was not written, never published it, and will write it
    again on the next attempt. So a failed append is rolled back to the last record that did
    commit, and the rule the writer, the reader and the caller can all hold at once is: **a
    record is in this log if and only if this call returned.**"""
    if durable:
        _mkdir_durable(path.parent)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    # "Fresh" is a file holding no committed record yet, not merely one that does not exist: an
    # append that died partway through the file's *first* record leaves it created and the entry
    # naming it unsynced, because the sync below never ran. The next committed record is still
    # the first this file holds, so its directory entry has to become durable along with it.
    committed = _drop_uncommitted_tail(path) if durable else 0
    fresh = durable and committed == 0
    try:
        with path.open("a", encoding="utf-8") as handle:
            line = json.dumps(record, allow_nan=False)
            if not durable:
                handle.write(line + "\n")
            else:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        if durable:
            _undo_failed_append(path, committed)
        raise
    if fresh:
        _fsync_dir(path.parent)


def _undo_failed_append(path: Path, committed: int) -> None:
    """Drop whatever a failed append left behind, back to the last record that committed.

    Only ever removes bytes this call wrote: ``committed`` is the length of the log after the
    uncommitted tail was dropped and before the append began, and this writer is the only one
    appending to it. A crash between the truncation and its own durability simply leaves the
    record present and readable, which is the state the retry would have produced anyway.

    Best-effort. A rollback that cannot run must not mask the write error already on its way to
    the caller — that error is what stops the stream, and it stops it either way."""
    try:
        os.truncate(path, committed)
    except OSError:
        pass


def _drop_uncommitted_tail(path: Path) -> int:
    """Discard a trailing record whose terminator never landed, before appending past it, and
    report how many committed bytes are left (0 for a file that does not exist yet).

    A write that died partway leaves a prefix of its record with no terminator, and
    :func:`_read_jsonl` reads that as absent — nothing downstream was ever told about it. An
    append onto the end of it would fuse the fragment and the new record into a single
    terminated line: a *committed* record that cannot be parsed, which is the one thing recovery
    may not skip. So the fragment goes first, and the two sides read the same boundary. This is
    the state a resumed run finds after the crash it exists to continue from, and nothing
    committed is at risk: the truncation stops at the last terminator in the file."""
    try:
        with path.open("rb") as handle:
            end = handle.seek(0, os.SEEK_END)
            if end == 0:
                return 0
            handle.seek(end - 1)
            if handle.read(1) == b"\n":
                return end
            committed = 0
            probe = end
            while probe > 0:
                start = max(0, probe - _TAIL_SCAN_BYTES)
                handle.seek(start)
                chunk = handle.read(probe - start)
                cut = chunk.rfind(b"\n")
                if cut != -1:
                    committed = start + cut + 1
                    break
                probe = start
    except FileNotFoundError:
        return 0
    os.truncate(path, committed)
    return committed


def _mkdir_durable(directory: Path) -> None:
    """Create ``directory``, and make every entry on the path to it survive a host crash.

    Syncing a directory persists the entries *inside* it, never the entry that names it — that
    one lives in its parent. So each level has to be synced into the level above it, top down,
    and the walk runs the whole way up rather than stopping at the levels this call created.

    **An existing level is no evidence that anyone synced it.** ``mkdir`` makes a level visible
    immediately and durable never, so publishing only what this call created infers from a
    level's existence that someone else already did the work — and nothing anywhere promises
    that. The provenance directory a harness made a moment earlier, a path some unrelated program
    created last week, a writer that made the chain and died before its own sync: each one leaves
    this call finding everything present and returning success over a store whose directory entry
    is still only in the page cache. A crash then takes the whole record, rows and all, with
    every write having reported success — which is the case this exists to prevent, arrived at
    from the other side.

    Walking to the root costs one directory fsync per level, and a directory with nothing dirty
    behind it costs a syscall rather than a disk flush — small against the record's own file
    sync, which every row pays anyway, and bounded by the depth of the path. What it buys is that
    the publishing is unconditional. The syncs themselves stay best-effort (see
    :func:`_fsync_dir`, where a filesystem that refuses one may not fail the write it was
    protecting), so what is guaranteed is that nothing on the path goes unattempted, not that a
    hostile filesystem was talked into it. Top down, so a crash part-way through leaves a durable
    prefix rather than a durable entry inside an ancestor that is still missing."""
    directory.mkdir(parents=True, exist_ok=True)
    holders: List[Path] = []
    level = directory
    while level.parent != level:  # the filesystem root names no entry above itself
        holders.append(level.parent)
        level = level.parent
    for holder in reversed(holders):
        _fsync_dir(holder)


def _fsync_dir(directory: Path) -> None:
    """Fsync a directory entry. Best-effort: not every platform or filesystem permits it, and a
    refusal must never fail the write it was protecting."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Every *committed* record in a log :func:`_append_jsonl` wrote.

    A record is committed once its terminating newline is on disk, and that writer only puts it
    there once the record itself is. So an unterminated tail is a write that died before it
    returned: nothing was published on the strength of it, no counter moved, and reading it as
    absent is what lets a log survive the crash it exists to record — the alternative is that the
    last append takes every intact record before it down with it, and a dispense whose result was
    lost mid-write can never become the ``broker_abort`` it is owed.

    Everything else that will not parse is corruption of a record that *did* commit, and is
    raised naming the file and the line that holds it. The asymmetry is the point: recovery may
    skip a row nobody was ever told about, and may never quietly skip one somebody was."""
    if not path.exists():
        return []
    committed, terminator, _uncommitted = path.read_bytes().rpartition(b"\n")
    records: List[Dict[str, Any]] = []
    for number, line in enumerate((committed + terminator).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:  # JSONDecodeError, or a line that is not even UTF-8
            raise ValueError(f"{path} line {number} is not a JSON record: {exc}") from exc
    return records


def read_dispenses(prov_dir: Path) -> List[Dict[str, Any]]:
    """Every dispense record written under ``prov_dir``."""
    return _read_jsonl(Path(prov_dir) / _DISPENSES_FILE)


def read_results(prov_dir: Path) -> List[ResultRow]:
    """Every recorded result row under ``prov_dir``."""
    return [ResultRow.from_wire(row) for row in _read_jsonl(Path(prov_dir) / _RESULTS_FILE)]


def reconcile(prov_dir: Path) -> List[ResultRow]:
    """Pair dispense records with results and report the unmatched ones.

    A dispense with no result means the stream died between handing the task out and sealing it
    — a crash, a SIGKILL, a ``docker rm -f``. Recovery is this pairing, not a promise the stream
    could not keep: each unmatched dispense becomes a ``broker_abort`` row with **no score**, so
    it can be counted but never averaged. The rows are returned, not written: whether an
    abandoned queue position is replayed is the caller's policy (``resume=True`` replays it).

    **Provenance survives with it.** Whatever each extension observed before the task went out is
    in the dispense record (see :meth:`TaskStream._write_dispense`), so the row carries
    ``extensions[namespace] = {"dispensed": ...}`` — the half of the span that actually happened,
    and nothing else. There is no ``sealed`` member and no ``error`` one, because no ``finalize``
    result was ever committed for this dispense, and inventing either would put a value on the row
    that no extension produced. That absence is the shape, not an omission: an orderly row's entry
    always carries exactly one of ``sealed`` or ``error`` beside its ``dispensed``, so
    ``"sealed" not in entry and "error" not in entry`` holds for a reconciled entry and for no
    other. A consumer does not need even that much — ``closure == "broker_abort"`` is produced
    here and nowhere else in the module, and is the row-level fact the namespace shape follows
    from.

    It says the record has no sealed observation, never that ``finalize`` did not run: a kill
    between the callback and the durable append leaves the extension's side effect out in the
    world with nothing committed about it, which is the boundary an append-is-the-commit log has
    and not one this can read past.

    A dispense record written before provenance was recorded on it has no ``extensions`` member at
    all, and reconciles to an empty map — the same answer a run with no extensions gives."""
    prov_dir = Path(prov_dir)
    sealed = {row.lease for row in read_results(prov_dir)}
    return [
        ResultRow(
            seq=int(record["seq"]),
            lease=str(record["lease"]),
            position=int(record["position"]),
            env=str(record["env"]),
            task_idx=int(record["task_idx"]),
            closure="broker_abort",
            score=None,
            diagnostic="dispensed but never sealed; the stream did not exit in an orderly way",
            extensions={
                namespace: {"dispensed": observed}
                for namespace, observed in dict(record.get("extensions") or {}).items()
            },
        )
        for record in read_dispenses(prov_dir)
        if record["lease"] not in sealed
    ]


class TaskStream:
    """Serve a queue of env tasks: one episode per dispensed task, sealed and scored by the
    stream, one :class:`ResultRow` each.

    Args:
        env_for: builds a **fresh** env for a given env name. Called once per dispensed task
            (the episode owns and closes it) plus once per env name at construction, for the
            tool manifest the server publishes. Every instance it returns must publish that
            same manifest; one that does not stops the stream when its task comes up, because
            the endpoint was registered from the first instance and cannot be re-registered.
            Its envs are closed on the loop that built them and on no other: this constructor
            is synchronous, so if it is called inside a running loop and then fails, the
            catalog envs it built are closed *on that loop*, just after the error propagates
            (see :func:`_close_on_owning_loop`). Nothing is moved to a worker loop, and nothing
            about the failure is swallowed — what could not be finished is attached to the
            error being raised.
        tasks: the materialised queue. Non-empty; may repeat a task index.
        prov_dir: where ``dispenses.jsonl`` and ``results.jsonl`` are appended. Its rows
            carry the task index and the env's raw feedback, so it belongs to the harness —
            keep it off any filesystem the agent under test can read. One directory per run:
            one that already holds records is refused unless ``resume`` says to continue it.
        max_in_flight: how many episodes may be live at once. Above 1, ``get_task`` returns a
            lease and every native call must carry it. A pull beyond capacity seals the oldest
            live task, exactly as a pull does at capacity 1.
        deadline: per-episode wall-clock seconds from the moment the task is handed out until
            the stream **takes it over**. When it elapses the stream claims the seal itself and
            records ``closure="timeout"`` with no score; the queue keeps draining. Must be a
            finite positive number of seconds — ``None``, not an infinite or NaN one, is how the
            deadline is disabled, since either of those would leave a watchdog running that can
            never fire and a caller believing a clock was set.

            It is a deadline on the *agent*, so it starts where the agent's turn does. The
            durable dispense record is written first and the clock stamped after it, because
            storage latency is the harness's cost: a budget cannot be spent on a task the agent
            has not been shown yet, and a write slower than the deadline would otherwise hand
            out a task already out of time. From there it holds whether the agent is idle or has
            a call in flight — an env that is slow in a tool or in its finalizer cannot spend
            the clock and still be recorded as an ordinary seal. What it is not is a bound on the
            env: the seal it claims adopts an already-running finalization rather than forcing a
            second terminal over the top of it (see :meth:`_run_seal`), so the row lands when
            that finalization returns. It says ``timeout``, but it says it late, and a
            finalization that never returns leaves the task unrecorded and the drain waiting on
            it — there is no outcome to record and nothing here may invent one. Bounding the env
            itself is the env's own timeout to set.
        resume: replay only the queue positions that have no result row yet. The stored
            provenance must have been recorded against this same queue — every recorded
            position is checked against it, and a disagreement raises rather than skipping a
            task that never ran. This is also the only way to serve into a directory that
            already holds records: without it they would be appended to, not continued.
        provenance: extensions that record something extra per dispensed task, under
            ``ResultRow.extensions[namespace]``. Namespaces must be unique, and must be exact
            ``str`` — this is the key a row's provenance is filed under, so a value that could
            answer a later comparison differently is refused rather than recorded.
        provenance_timeout: how long this stream will *wait* for an extension's
            ``begin``/``finalize`` before treating it as failed. A callback that hangs must not
            wedge the queue, so at the bound the stream stops waiting: the callback is cancelled
            and let go, and the dispense or the seal carries on without it. It is a bound on the
            wait and not a kill — a callback that ignores its cancellation keeps running, and one
            that never yields to the event loop at all cannot be pre-empted by anything
            in-process (see :meth:`_with_timeout`). Finite and positive, for the same reason
            ``deadline`` is; ``None`` waits indefinitely.
    """

    def __init__(
        self,
        env_for: Callable[[str], Env],
        tasks: Sequence[TaskRef],
        *,
        prov_dir: Path,
        max_in_flight: int = 1,
        deadline: Optional[float] = None,
        resume: bool = False,
        provenance: Sequence[Provenance] = (),
        provenance_timeout: Optional[float] = 30.0,
    ) -> None:
        if not isinstance(max_in_flight, int):
            # A capacity is a count of slots, and everything downstream reads it as one: it
            # slices the live entries a pull displaces, and it is compared against 1 twice —
            # once to decide whether a lease is advertised at all, once to decide whether a call
            # must carry one. A value that is not a whole number passes each of those
            # differently, so the stream half-works rather than refusing: `1.5` dispenses a task
            # and then raises `TypeError` out of the *next* pull, and `nan` (which is `> 1` and
            # `== 1` neither) hands out a task with no lease and then refuses every call on it
            # as `missing_lease`. Neither is a capacity, and both are found here rather than one
            # dispense later.
            raise ValueError(
                f"max_in_flight must be a whole number of slots, got {max_in_flight!r}"
            )
        if max_in_flight < 1:
            raise ValueError(f"max_in_flight must be at least 1, got {max_in_flight}")
        if deadline is not None and not (math.isfinite(deadline) and deadline > 0):
            # NaN and infinity would both pass a `<= 0` check and then silently disable
            # enforcement: `now - started >= deadline` is false forever against either, so the
            # watchdog would run for the whole queue and time nothing out while the caller had
            # every reason to believe a clock was set. `None` is the way to say that on purpose.
            raise ValueError(
                f"deadline must be a finite positive number of seconds, got {deadline}; "
                "pass None to serve without one"
            )
        if provenance_timeout is not None and not (
            math.isfinite(provenance_timeout) and provenance_timeout > 0
        ):
            # The same hole, and the timer splits it in two: infinity bounds nothing, while NaN
            # expires immediately (measured, not assumed: `asyncio.wait` returns nothing done at
            # 0.0s against it), so every extension on every task would be filed as having timed
            # out. Either way the rows are wrong and the run reports fine.
            raise ValueError(
                "provenance_timeout must be a finite positive number of seconds, got "
                f"{provenance_timeout}; pass None to bound it at nothing"
            )
        # Read once, here, and kept: `namespace` is an ordinary attribute of an object the caller
        # owns and can rebind at any time, so re-reading it at every dispense would validate one
        # value and record under another. The names checked below are the names every row is
        # keyed by (see `self._provenance`).
        namespaces = [extension.namespace for extension in provenance]
        for ns in namespaces:
            # Exact `str`, subclasses included, for the reason `_require_task_ref` gives about a
            # queue entry: this is an identity field, and a subclass brings its own
            # `__eq__`/`__hash__`/`__len__` to every comparison later made on it. Both of the
            # checks that follow are such comparisons, and neither of them lasts. `not ns` is one
            # `__len__` call; the uniqueness check below is one round of hashing — while the
            # dict writes they protect happen once per dispense and once per seal, on the far
            # side of every containment boundary this module draws. An object that answers one
            # way here and another way later is the whole hole, and both halves of it are worse
            # than a refusal: two namespaces that pass as distinct and then compare equal file
            # one extension's output under the other's name, with the loser's `finalize` never
            # called, the row reporting success and nothing anywhere saying a name was involved;
            # and one whose `__hash__` starts raising mid-run raises where the row is being
            # *keyed*, outside the boundary that contains a failing extension, suppressing a row
            # whose score terminal had already succeeded.
            if type(ns) is not str or not ns:
                raise ValueError(
                    "every provenance extension needs a non-empty string namespace, got "
                    f"{ns!r} ({type(ns).__name__}); it is the key this extension's output is "
                    "filed under on every row, so it has to be a value whose identity cannot "
                    "change after it has been checked"
                )
        if len(set(namespaces)) != len(namespaces):
            raise ValueError(
                f"provenance namespaces must be unique, got {namespaces}; two extensions "
                "sharing one would overwrite each other's output"
            )
        # Checked, not merely rebuilt: these two fields are the identity every row of this run is
        # filed under (see :func:`_require_task_ref`).
        queue = [_require_task_ref(ref) for ref in tasks]
        if not queue:
            raise ValueError(
                "a stream needs a non-empty queue: its published tool manifest is derived "
                "from the queued tasks"
            )
        env_names = sorted({ref.env for ref in queue})
        if len(env_names) > 1:
            raise ValueError(
                f"a stream serves one env; the queue names {env_names}. Native tool names "
                "collide across envs, so multi-env serving needs prefixed registration"
            )

        self._env_for = env_for
        self._queue: List[TaskRef] = queue
        self._max_in_flight = max_in_flight
        self._deadline = deadline
        # The validated names, bound to the extensions they were validated for. Uniqueness and
        # non-emptiness are properties of *these* strings, and these are the ones the spans and
        # the rows are keyed by — an extension that renames itself afterwards renames nothing.
        self._provenance: Tuple[Tuple[str, Provenance], ...] = tuple(
            zip(namespaces, provenance)
        )
        self._provenance_timeout = provenance_timeout
        self.prov_dir = Path(prov_dir)
        self.results_path = self.prov_dir / _RESULTS_FILE
        self.dispenses_path = self.prov_dir / _DISPENSES_FILE
        # Checked here, ahead of the catalog: it is a statement about the arguments, and refusing
        # before a factory is called costs no env the caller would then have to see closed.
        self._require_fresh_provenance(resume)

        # One long-lived env per env name, used only to read the published contract (it never
        # begins a session, so closing it releases nothing an episode owns). Constructing it is
        # also what provisions an env whose data is fetched lazily.
        #
        # Everything from here to the end of construction runs under a cleanup guard: a factory
        # may provision real resources, and a constructor that raises hands back no object, so
        # nothing else could ever close what it built. That covers a partly-built catalog as
        # well as any later check that refuses the queue.
        self._catalog: Dict[str, Env] = {}
        try:
            for name in env_names:
                self._catalog[name] = env_for(name)
            # Validated up front: the server publishes one schema per tool name at startup, but
            # `Env.describe(task_id)` is a *per-task* contract. A queue whose tasks disagree
            # cannot be served, and the contract is public from the first registration — so a
            # disagreement found later can only ever be refused, never accommodated. Catching
            # this one here means it costs no task at all; the per-dispense check below refuses
            # the disagreements this one cannot see, because it reads a single instance.
            # Frozen as soon as it is validated, and frozen *native* — before anything is derived
            # from it. Everything downstream is a copy of these values: the signature the drift
            # check compares, the score terminal this stream drives when the agent ends nothing,
            # and the contract every task is framed with. Freezing only what the endpoint
            # advertises would leave the rest carrying the env's own objects, on paths where
            # running the env's code loses a call and costs a task its score (see
            # :func:`_frozen_manifest`).
            self._manifest: Dict[str, List[ToolManifest]] = {
                name: _frozen_manifest(name, self._validate_manifest(name)) for name in env_names
            }
            # The frozen contract, in comparable form. Every episode this stream starts runs on
            # a *different* instance the factory built, so this is what each one's own manifest
            # is checked against before its task is dispensed. Compared on the NATIVE names the
            # env publishes; what is advertised is derived from it afterwards.
            self._signature: Dict[str, Tuple[Any, ...]] = {
                name: _manifest_signature(tools) for name, tools in self._manifest.items()
            }
            self._score_terminal: Dict[str, Optional[str]] = {
                name: next((m.name for m in tools if m.terminal_kind == "score"), None)
                for name, tools in self._manifest.items()
            }
            # What the agent actually sees: at capacity 1 the env's own names and schemas, and
            # above 1 each schema wrapped with the required `lease` that names the episode. Both
            # are built from the frozen native contract above — already this stream's plain data
            # (see :func:`_frozen_manifest`) — so the lease argument is added to a copy of this
            # stream's own schema and never to an object the env still holds, and what the
            # endpoint registers above capacity 1 is this stream's schema throughout.
            #
            # The native contract stays unwrapped in `self._manifest`, because that is what an
            # episode's own manifest is compared against and what the terminal this stream drives
            # is named from. Neither goes through the wrapper: the wrapper is the agent's routing
            # surface, and an episode never sees the lease at all.
            self._advertised: Dict[str, List[ToolManifest]] = {
                name: [_leased_manifest(m) for m in tools] if max_in_flight > 1 else list(tools)
                for name, tools in self._manifest.items()
            }
            # Every lease ever issued, so one can never be reused — a recycled lease would let a
            # delayed call from a finished task act on, and be scored into, its successor. A
            # resumed run seeds it from the record below: "ever" spans the whole record, not the
            # process, or the guarantee lapses at exactly the boundary a crash creates.
            self._issued: set[str] = set()

            # Resume by queue POSITION, never by task index: the same index may be queued twice and
            # both occurrences must play. A position that was dispensed but never sealed has no
            # authoritative outcome, so it is replayed (its abandoned dispense stays visible through
            # `reconcile`, and the replay gets its own lease).
            #
            # A position only carries meaning next to the queue it was recorded against, so every
            # stored position is checked against the queue being served before any of it is trusted.
            #
            # `seq` continues past every number this directory already used, dispenses included.
            # A dispense is durable before its task is handed out, so an abandoned one holds a
            # `seq` no result row answers — and that is precisely the position a resume replays.
            # Numbering the replay from the results alone would hand it the abandoned dispense's
            # own number, leaving `reconcile`'s `broker_abort` and the replay's real outcome
            # sharing one identifier: two rows for one position, indistinguishable in order, in
            # the record whose whole purpose is to keep them apart.
            #
            # Leases continue past this directory's record for the same reason `seq` does, and
            # for a stricter one. `reconcile` pairs a dispense with a result **by lease alone**,
            # so a resumed run that minted one this record already holds would have the earlier
            # run's result answer the later run's dispense: the position dispensed and never
            # sealed reconciles to nothing, and the `broker_abort` a crash owes is simply absent
            # from a record that still reads as complete. A row goes missing, quietly. The odds
            # of a 128-bit CSPRNG repeating are not what makes that safe — this module promises a
            # lease is never reused, and it is the run that has to keep the promise.
            self._done_positions: set[int] = set()
            self._seq = 0
            if resume:
                for record in read_dispenses(self.prov_dir):
                    self._require_position_matches(
                        int(record["position"]),
                        TaskRef(str(record["env"]), int(record["task_idx"])),
                        source="a dispense record",
                    )
                    lease = str(record["lease"])
                    if lease in self._issued:
                        # Already ambiguous on disk, before this run adds anything: the pairing
                        # `reconcile` does cannot answer two dispenses with one result, so
                        # whichever of them was sealed reports both as sealed. Continuing would
                        # append to a record that has already lost a row.
                        raise ValueError(
                            f"{self.dispenses_path} records two dispenses under one lease "
                            f"({lease!r}); a result is paired with a dispense by its lease, so "
                            "one of them can never be answered and a crash on it would be "
                            "invisible to reconciliation"
                        )
                    self._issued.add(lease)
                    self._seq = max(self._seq, int(record["seq"]))
                for row in read_results(self.prov_dir):
                    self._require_position_matches(
                        row.position, TaskRef(row.env, row.task_idx), source="a result row"
                    )
                    # A row's lease is the dispense's, so this adds nothing to a record whose two
                    # files agree — and it is what keeps the guarantee if they ever do not.
                    self._issued.add(row.lease)
                    self._done_positions.add(row.position)
                    self._seq = max(self._seq, row.seq)

            self._position = 0
            self._consumed = 0
            self._live: Dict[str, _Live] = {}
            # The leases of tasks that are over, and nothing else about them (see
            # `_retire_settled`). A lease outlives its task because a late call must be told the
            # task ended rather than that its lease was never real — but that answer needs the
            # string and none of what the entry points at, so the entry goes and the string
            # stays. Bounded by the queue, and by one 32-character key per task.
            self._settled_leases: Set[str] = set()
            self._results: List[ResultRow] = []
            # A SHORT registry lock: it guards the live/queue bookkeeping and is never held across
            # an episode call, an extension callback, or a seal.
            self._lock = asyncio.Lock()
            # Serialises dispensing, which the registry lock cannot: a dispense opens spans and
            # starts an episode, both of which must happen with the registry free.
            self._dispense_lock = asyncio.Lock()
            self._closed = False
            self._stopped: Optional[_Stopped] = None
            self._catalog_closed = False
            self._watchdog: Optional[asyncio.Task[None]] = None
            self._releasing: Optional[asyncio.Task[None]] = None
        except BaseException as error:
            for note in self._close_catalog_now():
                error.add_note(note)
            raise

    # ----- construction-time validation -----

    def _require_fresh_provenance(self, resume: bool) -> None:
        """Refuse a provenance directory another run already recorded into, unless ``resume``
        says to continue it.

        A stream that is not resuming numbers from the start of its own queue — ``position``
        from 0, ``seq`` from 1 — and appends. Pointed at a directory that already holds records
        it would therefore file its own under keys that directory already uses, with nothing on a
        row naming the run that wrote it, while :attr:`results` holds only this run's. Scoring
        the file then counts a task the queue holds once twice over, and scoring the object does
        not: two faithful readings of the same run that disagree.

        The dispense file is checked for the same reason as the result file, and it is the likelier
        half: a crash is exactly when a directory is reused, and an abandoned dispense the rerun
        does not know about becomes a :func:`reconcile` ``broker_abort`` beside the result row the
        rerun earned for the same position — a crash reported against a task that ran.

        ``resume`` is what tells the two apart, and it is not a flag this can infer: continuing a
        record and rerunning into one by mistake are the same call otherwise. So the ambiguity is
        refused at the one moment where nothing has been spent yet."""
        if resume:
            return
        recorded = [
            path
            for path in (self.results_path, self.dispenses_path)
            if path.is_file() and path.stat().st_size > 0
        ]
        if recorded:
            raise ValueError(
                f"{self.prov_dir} already holds records ("
                + ", ".join(path.name for path in recorded)
                + "); a stream that is not resuming numbers from the start of its own queue and "
                "appends, so it would file a second run's records under the first run's "
                "positions with nothing to tell them apart. Serve this queue into a fresh "
                "provenance directory, or pass resume=True to continue this one."
            )

    def _close_catalog_now(self) -> List[str]:
        """Release the catalog envs from *sync* code, which only a failed constructor needs:
        there is no stream to call :meth:`aclose` on, and nothing else holds these envs.

        Returns what could not be finished, for the caller to attach to the error it is already
        raising. Cleanup must not mask that error — but it must not be silent either: a swallowed
        close is an env left open with nothing left holding it, which is the whole failure this
        path exists to prevent.

        Nothing in the loop below awaits, so no cancellation can be *delivered* into it: one
        observed here was raised by the env's own close, and it masks the error being raised
        exactly as any other failure would. Hence ``None`` for the cancellation (see
        :func:`_must_propagate`)."""
        notes: List[str] = []
        for name, env in self._catalog.items():
            try:
                if _close_on_owning_loop(env):
                    notes.append(
                        f"the catalog env for {name!r} is being closed on the loop that built "
                        "it; a synchronous constructor cannot await that, so the close finishes "
                        "just after this error propagates"
                    )
            except BaseException as exc:  # noqa: BLE001 — never mask the failure being raised
                if _must_propagate(exc, None):
                    raise
                notes.append(
                    f"the catalog env for {name!r} could not be closed while this error was "
                    f"being raised ({_rendered_failure(exc)}); it may still hold resources"
                )
        self._catalog.clear()
        return notes

    def _require_position_matches(self, position: int, ref: TaskRef, *, source: str) -> None:
        """A stored position must name the same task the current queue holds there.

        Resume trusts recorded positions — it skips them, and `reconcile` pairs them — so a
        position recorded against a *different* queue would silently retire a task that was never
        run, or report a crash against the wrong task. The provenance directory cannot say which
        queue it belongs to, so the queue itself is the check, and a disagreement is a caller
        error worth failing on before anything is spent rather than a row to quietly ignore or
        quietly replay."""
        found = self._queue[position] if 0 <= position < len(self._queue) else None
        if found == ref:
            return
        raise ValueError(
            f"{self.prov_dir} holds {source} for queue position {position} naming "
            f"{ref.env!r} task {ref.task_idx}, but this queue holds "
            + (
                f"{found.env!r} task {found.task_idx} there"
                if found is not None
                else f"only {len(self._queue)} positions"
            )
            + "; resuming needs the queue the provenance was recorded against, or a fresh "
            "provenance directory"
        )

    def _validate_manifest(self, env_name: str) -> List[ToolManifest]:
        """Every queued task of ``env_name`` must publish the same tool manifest; return it."""
        env = self._catalog[env_name]
        baseline: Optional[Tuple[Any, ...]] = None
        baseline_ref: Optional[TaskRef] = None
        tools: List[ToolManifest] = []
        for ref in dict.fromkeys(r for r in self._queue if r.env == env_name):
            spec = env.describe(str(ref.task_idx))
            signature = _manifest_signature(spec.tools)
            if baseline is None:
                baseline, baseline_ref, tools = signature, ref, list(spec.tools)
            elif signature != baseline:
                assert baseline_ref is not None
                raise ValueError(
                    f"env {env_name!r} publishes a different tool manifest for task "
                    f"{ref.task_idx} than for task {baseline_ref.task_idx}; a stream registers "
                    "one schema per tool name for the whole queue, so its tasks must agree"
                )
        return tools

    def _require_published_manifest(self, env_name: str, spec: TaskSpec) -> None:
        """Refuse an episode whose own manifest is not the one this stream serves.

        Construction validates the *catalog* instance across the queue's task ids, but every
        episode runs on a different instance the factory built, and the constructor takes any
        factory. So a per-instance manifest — one that varies with load order, a cached fetch, a
        feature flag, the wall clock — passes construction and then disagrees with the endpoint
        that was registered from the catalog instance. Nothing downstream reconciles the two: a
        tool the episode adds is framed but *unknown* to the server, and the episode validates a
        score terminal's arguments against its **own** schema, so a call shaped to one contract
        is rejected or silently mis-graded against the other.

        That makes it an eval-integrity failure rather than a usability one — the task is scored,
        the row looks ordinary, and the agent could not have passed. So the task is not dispensed
        and the stream stops: the drift belongs to the *factory*, not to this task, so every task
        after it would be scored the same way.

        **Reading the episode's manifest is itself the env's data.** Comparing it serializes the
        schema it advertises and describing it names the tools it publishes, so both run on
        values an env supplied, and both are contained here. The finding is an *argument* to the
        stop: a manifest that cannot be read would otherwise replace this refusal with its own
        exception and skip the stop, leaving the queue served against the very contract this
        check exists to refuse. A manifest this stream cannot compare is a manifest it cannot
        confirm, and it is not a third thing either — the published signature is only ever a
        schema that serialized, so one that does not is a different manifest, found here like
        any other."""
        try:
            if _manifest_signature(spec.tools) == self._signature[env_name]:
                return
            drift = _manifest_drift(self._manifest[env_name], spec.tools)
        except BaseException as exc:  # noqa: BLE001 — see above; the refusal outranks its detail
            if _must_propagate(exc, None):
                raise
            drift = f"the two manifests could not be compared ({_rendered_failure(exc)})"
        cause = RuntimeError(
            f"env {env_name!r} published a different tool manifest for this episode than the "
            f"one this stream serves ({drift}). A stream registers one schema per tool name "
            "when it is built, from a fresh instance of the env, so every later instance must "
            "publish that same manifest — otherwise a task is framed with tools the endpoint "
            "does not expose, or schemas it does not honour"
        )
        self._stop(
            cause,
            dispensing=(
                f"this stream stopped: env {env_name!r} published a tool manifest for a new "
                "episode that differs from the one this stream serves, so no further task can "
                "be scored against it"
            ),
            closing=(
                f"this stream stopped before its queue was served: env {env_name!r} published a "
                "tool manifest for a new episode that differs from the one this stream serves"
            ),
        )
        raise cause

    def _require_framable(self, env_name: str, spec: TaskSpec) -> Tuple[str, Optional[int]]:
        """Refuse an episode whose task this endpoint could not hand over, and hand back the two
        values it confirmed.

        A framing is the published contract plus two values off the episode's own spec, and the
        contract half is already plain data by the time it is advertised (see the constructor).
        These two are not. A model validates a field when it is built and not when it is
        assigned, so an env that edits its spec afterwards can publish anything at all as
        ``instructions`` or ``horizon``, and nothing between here and the wire looks at them:
        they are carried verbatim into the framing and serialised by whoever answers
        ``get_task``.

        **Where it is found decides what it costs.** The position is consumed and the episode
        registered before the framing is handed back, so one that fails after that point is a
        dispensed task the agent was never answered for — the episode is live, the drain ends it,
        and the row it lands says the agent played the task out and got it wrong. That is a wrong
        number where a missing one was the truth. Found here the bad state is unreachable rather
        than recoverable: nothing is recorded, the position is still owed, and the episode is let
        go by the same handler that answers a drifted manifest.

        Confirming is the whole of it, because there is nothing left to detach: ``str`` and
        ``int`` are immutable, so a confirmed one aliases nothing an env can reach through, and
        every other field of the framing is this stream's own or the published contract's.
        Reading them is still the env's code — the spec is the env's object — so the read is
        contained like every other read of an env's values here, and a value that cannot be read
        is not a different finding from one that cannot be carried.

        They are handed back rather than read again where the framing is built, because *this* is
        the read that was checked. A spec is the env's object and an attribute of one can be
        anything an env cares to make it, so a second read is a second value — and the one that
        would reach the agent is the one nothing looked at.

        The stream stops, on the same line the rest of this module draws. An env that *fails*
        while a task is being opened is refused and nothing more — the next dispense builds a
        fresh episode and may well get one. This is the other kind: the env did not fail, it
        published a value its own contract does not admit, and it will publish it again when it
        is asked again. Refusing without stopping would leave the position unadvanced and the
        refusal repeating for the rest of the run, with ``aclose`` reporting a clean run over a
        queue it never served."""
        try:
            instructions = spec.instructions
            budget = spec.horizon
            if not isinstance(instructions, str):
                defect = "its instructions are not the text an agent is framed with"
            elif budget is not None and (
                isinstance(budget, bool) or not isinstance(budget, int)
            ):
                # `bool` is an `int` subclass, so an unguarded test would advertise `True` as a
                # budget of one step.
                defect = "its budget is not a whole number of steps"
            else:
                # The types above are what the framing *declares*; this is whether the endpoint
                # can actually send it. A `str` is not automatically text a transport can encode
                # — an unpaired surrogate is a legal Python string and a `UnicodeEncodeError` on
                # the way out — and that failure lands where it costs most: the position is
                # already consumed, so the answer the agent was owed never arrives, the drain
                # ends the task, and the row says it played the task out and got it wrong. Proved
                # here instead, on exactly the two values this confirms (see :func:`_wire_json`).
                try:
                    _wire_json({"instructions": instructions, "budget": budget})
                except (ValueError, TypeError, UnicodeError) as exc:
                    defect = f"the endpoint could not put it on the wire ({_rendered_failure(exc)})"
                else:
                    return instructions, budget
        except BaseException as exc:  # noqa: BLE001 — the refusal outranks its detail
            # Nothing here awaits, so a cancellation observed was raised where it was observed
            # (see :func:`_must_propagate`).
            if _must_propagate(exc, None):
                raise
            defect = f"reading it raised {_rendered_failure(exc)}"
        cause = RuntimeError(
            f"env {env_name!r} published a task this stream cannot hand out ({defect}). A task's "
            "framing is what the agent acts on and what `get_task` answers with, so it has to be "
            "something the wire can carry — and it has to be that before the dispense is "
            "committed, because after that a task is out and its row is owed an outcome"
        )
        self._stop(
            cause,
            dispensing=(
                f"this stream stopped: env {env_name!r} published a task framing this stream "
                "cannot hand out, so the queue could not be served past it"
            ),
            closing=(
                f"this stream stopped before its queue was served: env {env_name!r} published a "
                "task framing this stream cannot hand out"
            ),
        )
        raise cause

    def _deliverable_framing(
        self, ref: TaskRef, instructions: str, budget: Optional[int], lease: str
    ) -> DispensedTask:
        """The task the agent will be handed, proved deliverable **as a whole** before the
        dispense is committed.

        The framing is assembled from more than the spec, and only the spec's half was ever
        proved: ``instructions`` and ``budget`` (see :meth:`_require_framable`). This stream's own
        env key and the frozen contract were added afterwards, past the point where the position
        is consumed and the episode registered — so a value that cannot be encoded there costs a
        task: the endpoint answers ``get_task`` with a serialization error, the agent is handed
        nothing, and the drain then ends a task nobody ever received and records it as one the
        agent played and lost. An env key is an ordinary non-empty ``str`` as far as every check
        upstream of here is concerned, and one holding an unpaired surrogate is exactly that.

        So what is proved is the object itself, through the encoder the endpoint uses (see
        :func:`_wire_json`). Field by field it would be the same check with a list to keep in
        step; whole, a field added later is covered the day it is added. The ``lease`` is that
        field: it arrives already minted so the proof covers it too, rather than being attached
        to a framing this had already signed off on.

        Nothing of this run's record is durable yet when this runs, and no position has been
        consumed, so a refusal costs no task. The stop is the run's, on the line this module
        already draws — an env key and a frozen contract are properties of the stream, not of this
        task, so the next dispense would fail the same way."""
        framing = DispensedTask(
            # This stream's key for the env, which is what its rows are recorded under — not the
            # name the instance calls itself, which nothing here is keyed by.
            env=ref.env,
            # The two values `_require_framable` confirmed, not a fresh read of the spec.
            instructions=instructions,
            budget=budget,
            # The *published* manifest, never the live episode's: this is the contract the server
            # registered, and the check above is what makes the two the same thing. Detached from
            # it, and per dispense, for the reason :func:`_detached_manifest` gives: what a task
            # is framed with is a reading of the frozen contract, not a handle on it, and one
            # task's framing is not the next one's either. Its name and description are already
            # this stream's own plain copies, taken when the contract was frozen.
            tools=tuple(
                {
                    "name": m.name,
                    "description": m.description,
                    "input_schema": copy.deepcopy(m.input_schema),
                }
                for m in self._advertised[ref.env]
            ),
            # Only when more than one episode may be live: at capacity 1 the wire contract is the
            # env's own, so there is no lease to publish and none to route on. The entry still
            # carries one — it is the registry's key either way — but it is not the agent's.
            lease=lease if self._max_in_flight > 1 else None,
        )
        try:
            # On the wire form the endpoint answers with, not on the object: `to_wire` is what
            # `get_task` returns, and it is where a field that cannot be encoded would surface.
            _wire_json(framing.to_wire())
        except BaseException as exc:  # noqa: BLE001 — the refusal outranks its detail
            # Nothing here awaits, so a cancellation observed was raised where it was observed
            # (see :func:`_must_propagate`).
            if _must_propagate(exc, None):
                raise
            cause = RuntimeError(
                f"env {ref.env!r} has a task framing this endpoint could not answer with "
                f"({_rendered_failure(exc)}). What `get_task` returns has to be something the "
                "wire can carry, and it has to be that before the dispense is committed, because "
                "after that a task is out and its row is owed an outcome"
            )
            self._stop(
                cause,
                dispensing=(
                    f"this stream stopped: the framing for env {ref.env!r} could not be put on "
                    "the wire, so the queue could not be served past it"
                ),
                closing=(
                    f"this stream stopped before its queue was served: the framing for env "
                    f"{ref.env!r} could not be put on the wire"
                ),
            )
            raise cause from exc
        return framing

    def _stop(self, cause: BaseException, *, dispensing: str, closing: str) -> None:
        """Stop the stream at the first integrity failure, keeping that one's explanation: the
        failures that stop a stream all mean the run is no longer a record of the queue, and the
        first is the one that explains the rest."""
        if self._stopped is None:
            self._stopped = _Stopped(cause=cause, dispensing=dispensing, closing=closing)

    # ----- public surface -----

    @property
    def results(self) -> Sequence[ResultRow]:
        """The rows recorded so far, in seal order.

        A detached view of the recorded rows, rebuilt per read (see :func:`_detached_row`) — for
        the reason :attr:`tools` is one, and with the same consequence: what a reader is handed
        is a reading of the record, never a handle on it."""
        return tuple(_detached_row(row) for row in self._results)

    @property
    def stopped(self) -> bool:
        """True once an integrity failure stopped the stream: a task that could not be recorded
        as dispensed, a row that could not be recorded, a summary the record cannot headline, an
        env that raised while ending a task, or an episode framed with a contract the endpoint
        does not serve.

        Nothing the agent sees reports any of them. This flag, the rows already in
        ``results.jsonl``, and the exception :meth:`aclose` raises are where a harness learns of
        it — and a long run can poll this rather than wait for the drain."""
        return self._stopped is not None

    @property
    def max_in_flight(self) -> int:
        return self._max_in_flight

    @property
    def tools(self) -> Sequence[ToolManifest]:
        """The tool manifest this stream advertises — the env's own schemas at capacity 1, and
        the lease-carrying wrappers above it — as the frozen contract the constructor made of it.

        A detached view of that contract, rebuilt per read (see :func:`_detached_manifest`) —
        reading what this endpoint serves may not be a way to change it."""
        return tuple(
            _detached_manifest(manifest)
            for manifest in self._advertised[next(iter(self._advertised))]
        )

    def queue_info(self) -> QueueInfo:
        """Where the queue stands right now (see :class:`QueueInfo`).

        ``in_flight`` counts *episodes*, not registry entries. The two stopped being the same
        thing when an entry began outliving its own release: a seal that failed on the storage
        keeps its composed row, hands its claim back so a later drain can retry the append, and
        is released in the meantime — so its entry reads unsealed with its env already closed.
        Counting it would report a stopped, closed stream as still serving a task, to a harness
        that has nothing left to wait for and to an agent that has nothing left to answer. What
        that entry is owed is a row, and ``in_flight`` is not where a row is reported."""
        return QueueInfo(
            remaining=sum(
                1
                for position in range(self._position, len(self._queue))
                if position not in self._done_positions
            ),
            consumed=self._consumed,
            in_flight=sum(
                1 for live in self._live.values() if not live.sealed and not live.released
            ),
        )

    async def __aenter__(self) -> "TaskStream":
        self._start_watchdog()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Orderly drain: seal, score and record every episode **this stream** dispensed, then
        release the catalog envs. Idempotent. Orderly exit only — a SIGKILL or a ``docker rm
        -f`` runs none of it, which is what the dispense records exist for.

        The drain is *total*: an episode that cannot be sealed does not cost the others their
        seal, nor the stream its catalog envs and its watchdog.

        **Closing the stream and claiming its episodes are one step.** Every unsettled task's
        seal is taken in the same critical section that marks the stream closed, before any of
        them is waited on, so from the moment this stream reads closed there is no live episode
        left for a call to be routed to. A seal that then *fails* hands its claim back for a
        later drain to retry — and hands back only the claim: the task stays ended, which is what
        a call is refused on (see :attr:`_Live.ended`), so the window does not reopen behind a
        row that could not be written. What a call arriving after it gets is the refusal a
        finished task gets — no budget spent, no trajectory entry, no row — and the row the
        record ends up holding for that task is the one this drain forced (``drained``), never a
        score an agent earned after the shutdown began.

        **Releasing is not part of the drain, and a lost caller cannot take it with them.** The
        catalog envs and the deadline watchdog are held by this object and by nothing else, so a
        shutdown cancelled on its way out would leave an env holding MCP sessions and
        subprocesses, and a watchdog task running against a stream nobody is serving, with no
        later call obliged to arrive. So the release runs in a ``finally`` — whatever became of
        the drain — and as the stream's own task awaited through a shield, so this caller's
        cancellation reaches this caller and stops there (see :meth:`_settled`, which does the
        same for the tail of a seal, for the same reason).

        Raises if anything stopped the stream, here or earlier in the run — a dispensed task
        that went unrecorded, a task that could not be recorded as dispensed at all, a summary
        the record cannot headline, an env that raised while ending a task, or an episode that
        would have been framed with a contract the endpoint does not serve. Together with
        :attr:`stopped` and the rows themselves, this is where a stream driven entirely over MCP
        reports any of them: the harness never calls ``get_task`` itself, and nothing the agent
        sees says a run went wrong."""
        try:
            async with self._lock:
                # `_closed` alone must not end the drain. A shutdown cancelled mid-seal leaves a
                # dispensed task still unrecorded; returning early here would answer its durable
                # dispense with nothing at all, and recovery would call an orderly shutdown a
                # crash. So a retry joins the seal the cancelled attempt left running and
                # finishes the drain.
                self._closed = True
                # Anything still owed something, including a task whose seal another path has
                # already claimed — the claim below joins that same transition rather than racing
                # it. A row is not the end of that: the episode behind it is released, and any stop
                # it owes published, in the tail after the append, so a drain that stopped at the
                # row would return over an episode still letting go of its env (see
                # :attr:`_Live.settled`).
                #
                # **Every one of them is claimed here, before any of them is waited on**, in the
                # same critical section that closed the stream and with no await able to split the
                # two. Sealing is not a step this can take one task at a time: it drives an env
                # terminal, waits on its finalizer and runs every extension, any of which may block
                # for as long as it likes. Claiming inside the loop below would leave every task
                # after the blocked one unclaimed, and an unclaimed task is one :meth:`_resolve`
                # still routes calls to — so an agent could earn a scored, `sealed` row on a
                # stream that had already begun shutting down, for exactly as long as some
                # *other* task's env took to let go: a stop that did not happen, recorded as an
                # ordinary result.
                # The deadline claims ahead of its waiting for the same reason and is documented
                # there (see :meth:`_watch_deadlines`); this is the drain's half of it.
                claimed = [
                    (live, self._claim_seal(live, "drained"))
                    for live in self._live.values()
                    if not live.settled
                ]
            # Only the waiting happens with the registry free — it drives a terminal call and runs
            # extension callbacks, neither of which may hold the lock. A deadline firing at the
            # same moment meets the single per-task transition and takes its outcome instead of
            # racing it.
            for live, sealing in claimed:
                try:
                    await self._join_seal(live, sealing)
                except Exception:  # noqa: BLE001 — recorded on the stream; drain the rest
                    # A *failed* seal, unlike a cancelled one, is not retried, so this drain is
                    # the last chance to release what the entry still holds. (Cancellation is a
                    # BaseException and passes through untouched, leaving the task for the retry
                    # the claim hand-back exists for.)
                    await self._release(live)
        finally:
            await self._released()
        if self._stopped is not None:
            raise RuntimeError(self._stopped.closing) from self._stopped.cause

    async def _released(self) -> None:
        """Wait for the shutdown release — **without being able to abandon it**. Claimed once;
        every later arrival joins the same task, so it never runs twice."""
        releasing = self._releasing
        if releasing is None:
            releasing = self._releasing = asyncio.ensure_future(self._release_stream())
        await asyncio.shield(releasing)

    async def _release_stream(self) -> None:
        """Let go of what only this stream holds: its deadline watchdog, then its catalog envs.

        Stopped only after the drain has sealed everything, so a deadline that fires at the same
        moment finishes the seal it started instead of being cancelled halfway through it. Each
        env is dropped as it is closed, so a release that is interrupted anyway leaves the rest
        still owed rather than marked done.

        Nothing here may raise. This runs in :meth:`aclose`'s ``finally`` as the stream's own
        claimed task, so an exception escaping it would replace the run-level report the drain
        was about to make — here it would also leave every catalog env open, since they are let
        go after the watchdog — and, because the claim is the task and a failed one stays
        claimed, it would do the same on every later attempt. The watchdog already records what
        it could not seal (see :meth:`_watch_deadlines`); this is the backstop for a failure
        that did not, and it is recorded rather than swallowed so the drain still reports it.

        "Nothing may raise" includes ``CancelledError``, which an env's ``close`` can raise like
        any other exception. This task is joined through a shield and nothing in this module ever
        cancels it, so one arriving here is the env's, not a caller's — and letting it out would
        leave the release task *cancelled* while it is the claim, so ``_catalog_closed`` stays
        false, the envs it already popped are unreachable, and every later ``aclose`` re-awaits
        the same cancelled task and raises again. A shutdown with no orderly exit, for a teardown
        failure that is not the run's outcome."""
        cancellation = _Cancellation()
        watchdog, self._watchdog = self._watchdog, None
        if watchdog is not None:
            watchdog.cancel()
            try:
                await watchdog
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001 — reported by the drain, not from a finally
                self._stop(
                    exc,
                    dispensing=(
                        "this stream stopped: its deadline watchdog failed, so a dispensed "
                        "task's deadline may never have been enforced"
                    ),
                    closing=(
                        "this stream's deadline watchdog failed; a dispensed task's deadline "
                        "may never have been enforced"
                    ),
                )
        for name in list(self._catalog):
            env = self._catalog.pop(name)
            try:
                await env.close()
            except BaseException as exc:  # noqa: BLE001 — teardown is best-effort
                if _must_propagate(exc, cancellation):
                    raise
        self._catalog_closed = True

    async def get_task(self) -> Optional[DispensedTask]:
        """Dispense the next queued task, starting its episode. ``None`` once exhausted.

        Pulling a new task while one is still live abandons it, so the abandoned episode is
        sealed and scored first: every dispensed task lands exactly one row. One whose seal is
        already running is waited for rather than abandoned — a task is not finished with its
        slot until its episode has been released and any stop it owes recorded, and the next
        task is not dispensed until then.

        A pull the queue cannot answer displaces nothing. Whether another position exists is
        settled *before* any slot is taken, so the call that finds the queue empty leaves every
        live episode live: it is a question about the queue, and answering it may not end work
        the agent is still entitled to finish (nor record a forced outcome against a position no
        task was dispensed for).

        The tools it lists are the ones the endpoint actually serves, and the episode is checked
        against them before it is dispensed (see :meth:`_require_published_manifest`) — the
        framing an agent acts on and the surface it can call are the same contract or there is
        no task.

        The framing is also confirmed to be something the endpoint can *carry*, and confirmed
        before the dispense is committed (see :meth:`_require_framable`): a task that cannot be
        handed over has to be no task at all rather than a consumed position with no answer.

        Raises if anything stopped the stream, including a seal this call itself could not
        finish: that one is reported as the stop it already recorded, in the same words every
        other caller gets, rather than as the raw storage error one abandoned episode happened
        to raise."""
        self._start_watchdog()
        async with self._dispense_lock:
            async with self._lock:
                self._require_open()
                # **Exhaustion first, before a slot is taken from anyone.** A pull is a request
                # for a slot, and a slot is only worth taking if there is a task to put in it.
                # Asking afterwards makes the one call whose whole purpose is to learn that the
                # run is over into a forced terminal: the oldest unfinished episode is sealed and
                # scored as an ordinary agent-driven loss, over an answer its agent was still
                # free to submit and against a position no task was dispensed for. So a queue
                # with nothing left answers here, with every live episode untouched — the answer
                # is about the queue, and the caller's own open work is not part of it.
                if self._next_position() is None:
                    return None
                # There is a task, so the slot is worth taking. Below capacity nothing is
                # displaced; at capacity the OLDEST occupant is the one that gives way, so a
                # dispensed task always lands exactly one row instead of being silently
                # forgotten.
                #
                # A task whose seal is already running still occupies its slot: its episode and
                # env are open until that seal's release returns, and the stop an unheadlinable
                # summary or a failed terminal owes is published at the end of it, after the row.
                # So it is counted here and *joined* rather than stepped over — taking a slot
                # from under a running seal serves the queue past an integrity failure the stream
                # has already found. Joining is not restarting: a claimed seal is awaited, and
                # only one whose claim was handed back is retried (see :meth:`_seal`).
                live_now = sorted(
                    (live for live in self._live.values() if not live.settled),
                    key=lambda live: live.seq,
                )
                over_capacity = len(live_now) - self._max_in_flight + 1
                abandoned = live_now[:over_capacity] if over_capacity > 0 else []
            for live in abandoned:
                try:
                    await self._seal(live, forced="drained")
                except Exception:  # noqa: BLE001 — recorded on the stream; reported just below
                    # A *failed* seal, unlike a cancelled one, is not retried, so this is the
                    # last chance to release what the entry still holds — the same reason the
                    # drain releases here. (Cancellation is a BaseException and passes through
                    # untouched, leaving the entry for the retry the claim hand-back exists for.)
                    await self._release(live)

            async with self._lock:
                # Rechecked because a seal above may have stopped the stream: dispensing over
                # that would serve the rest of the queue against a record already missing an
                # outcome.
                self._require_open()
                position = self._next_position()
                if position is None:
                    # Unreachable: this same lock found a position before any slot was
                    # displaced, `_dispense_lock` keeps every other pull out of this body, and
                    # nothing a seal does consumes a queue position. Loud rather than `None`,
                    # because a `None` here would be this call reporting that no task is coming
                    # *after* it has just drained one — the outcome the check above exists to
                    # prevent, arrived at by a different route.
                    raise RuntimeError(
                        "this stream lost the queue position this pull had already found"
                    )
                ref = self._queue[position]

            # Everything below is outside the registry lock, and none of it has been exposed
            # yet: if a span refuses to open, the episode never starts and the position is
            # still owed. Spans first, so nothing needs cleaning up when one fails.
            spans, dispensed_extensions = await self._begin_spans(ref)
            episode = await ServedEpisode.open_env(
                self._env_for(ref.env), env_name=ref.env, task=ref.task_idx
            )
            try:
                spec = episode.describe()
                self._require_published_manifest(ref.env, spec)
                # Both checks sit above the bookkeeping that commits the dispense, and this one
                # for the second reason that check gives: a task nobody could be handed is not a
                # task, and finding that out after the position is consumed turns a task that was
                # never served into a row saying the agent served it badly (see
                # :meth:`_require_framable`). What it confirmed is what the framing below is
                # built from — a second read of the spec would be a second value, and an
                # unchecked one.
                instructions, budget = self._require_framable(ref.env, spec)
            except BaseException:
                # Nothing has been exposed, so this is the same cleanup the closed-mid-dispense
                # path below does: release the episode and drop the spans without finalizing —
                # no task was dispensed, so there is no outcome to close them against. The check
                # sits *above* `_write_dispense` deliberately: a durable record of a task that
                # was never handed out would read back through `reconcile` as a crash.
                #
                # The baseline is taken here, inside the handler, so it asks only about the
                # close: a cancellation delivered *before* it is already the exception being
                # re-raised below, and one the env's close merely raised would otherwise replace
                # the refusal every caller of `get_task` is told to expect.
                cancellation = _Cancellation()
                try:
                    await episode.close()
                except BaseException as exc:  # noqa: BLE001 — teardown must not mask the failure
                    if _must_propagate(exc, cancellation):
                        raise
                raise
            # Both of these are refusals of the same shape as a drifted manifest — the position
            # is still owed, no row is due, the episode is released here, and the spans opened
            # for it are dropped without finalizing: no task was dispensed, so there is no
            # outcome to close them against. Which is why the mint sits inside this guard too:
            # it can refuse (see :meth:`_mint_lease`), and a refusal that skipped the close would
            # leave an env and its MCP sessions held by an episode nobody has a handle on.
            try:
                # Minted before the framing rather than with the entry, because above capacity 1
                # the lease is a *field of the framing* and the proof below is a proof of the
                # whole object: a lease attached after it would be the one field nothing checked,
                # which is the hole that proof exists to close. Minting is not registering —
                # nothing is keyed by this string until the entry is published under the lock
                # below, and a dispense refused between here and there simply retires it, which
                # is what `_issued` is for. Safe outside the registry lock because `get_task`
                # holds `_dispense_lock` for its whole body and nothing else mints.
                lease = self._mint_lease()
                # Built and proved before the position is consumed and the episode registered:
                # what the agent is handed is this object, and every field of it has to be one
                # the endpoint can answer with (see :meth:`_deliverable_framing`).
                framing = self._deliverable_framing(ref, instructions, budget, lease)
            except BaseException:
                cancellation = _Cancellation()
                try:
                    await episode.close()
                except BaseException as exc:  # noqa: BLE001 — teardown must not mask the failure
                    if _must_propagate(exc, cancellation):
                        raise
                raise

            undispensed: Optional[ServedEpisode] = None
            unrecorded: Optional[BaseException] = None
            async with self._lock:
                # Rechecked in the very critical section that publishes the live record, because
                # opening the spans and the episode above ran with the registry free: a shutdown
                # can start and *finish* inside that window. Publishing afterwards would hand out
                # a task the completed drain never saw and the stopped watchdog is not watching,
                # so its durable dispense could never be answered — an orderly exit that reads
                # back as a crash. Nothing has been exposed yet, so the dispense is abandoned and
                # the caller gets the same answer a closed stream gives up front. The spans opened
                # for it are dropped without finalizing: no task was ever dispensed, so there is
                # no outcome to close them against.
                if self._closed:
                    undispensed = episode
                else:
                    live = _Live(
                        # The lease the framing above was proved carrying: the string the agent
                        # is handed and the key the registry routes its calls on are one value,
                        # or a call would name a task no entry answers to.
                        lease=lease,
                        seq=self._seq + 1,
                        position=position,
                        ref=ref,
                        episode=episode,
                        spans=spans,
                        dispensed_extensions=dispensed_extensions,
                    )
                    # Durable BEFORE the task is exposed: after this point a crash is
                    # reconcilable, because the record says a task was handed out and no result
                    # answers it. And nothing is committed before it: a position stepped over a
                    # write that failed is a task quietly dropped from the queue, its episode
                    # absent from the registry that is the only handle on it, and a drain that
                    # reports a clean run over the hole.
                    try:
                        self._write_dispense(live)
                    except BaseException as exc:
                        # Nothing was handed out, so this position is still owed and no row is
                        # due — the same shape as the manifest refusal above. But a provenance
                        # directory that cannot be appended to is not a per-task problem: the
                        # next dispense record and every result row after it go to that same
                        # directory, so the run can no longer be a record of the queue. Serving
                        # on would spend the rest of the queue against a file that already lost
                        # a task, so the stream stops and says so at both boundaries.
                        # (Cancellation is excluded, as everywhere else here: nothing failed.)
                        if not isinstance(exc, asyncio.CancelledError):
                            self._stop(
                                exc,
                                dispensing=(
                                    "this stream stopped: a task could not be recorded as "
                                    f"dispensed to {self.dispenses_path}, so a crash from here "
                                    "on could not be told apart from a task that was never "
                                    "handed out"
                                ),
                                closing=(
                                    "this stream could not record a dispense to "
                                    f"{self.dispenses_path}; the queue was not served to the end"
                                ),
                            )
                        # Closed outside the lock, with the dispense that was refused before
                        # it: same window, same teardown, one path.
                        undispensed = episode
                        unrecorded = exc
                    else:
                        # Synchronous from the write down, so no cancellation point can
                        # separate the record on disk from the bookkeeping that answers for
                        # it — and so the agent's clock and the moment the task becomes
                        # visible are the same instant. Started here rather than where the
                        # entry was built because the durable dispense sits between the two:
                        # a slow volume would otherwise charge its own latency to the agent,
                        # which cannot see the task until this returns, and a write slower
                        # than the deadline would hand out a task already out of time.
                        live.started = time.monotonic()
                        self._position = position + 1
                        self._seq = live.seq
                        self._consumed += 1
                        self._live[live.lease] = live
            if undispensed is not None:
                # Same shape as the manifest refusal above: this close is teardown, and its
                # failure — cancellation included — may not stand in for the answer the caller
                # is owed just below.
                cancellation = _Cancellation()
                try:
                    await undispensed.close()
                except BaseException as exc:  # noqa: BLE001 — teardown of a task nobody ever saw
                    if _must_propagate(exc, cancellation):
                        raise
                if unrecorded is not None:
                    raise unrecorded
                raise RuntimeError("this stream is closed")
            return framing

    async def dispatch(
        self,
        tool: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        lease: Optional[str] = None,
    ) -> ToolResult:
        """Route one native tool call to the episode its lease names, sealing it when it
        terminates.

        Above capacity 1 the lease is required; it may be passed as a keyword or (as the wrapper
        schemas advertise it) inside ``arguments``, and it is **stripped before the episode sees
        the call** so the routing capability never enters the env's trajectory. Every refusal
        below is a stream-level result, not an env step: a misrouted call costs no budget and is
        recorded nowhere.

        An ordinary call returns the env's own response: that *is* the agent's observation, and
        nothing but the env can produce it. A terminating call returns only the fact that the task
        is over — a fixed payload identical for every task and every outcome.

        Everything a terminal produces stays with the harness, not just the row the seal records
        (lease, position, task index, raw feedback). The env's terminal response is redacted too:
        for a ``score`` terminal it is the verdict this stream just recorded, and a queue that
        repeats an index would make it the signal that identifies the repeat. The feedback sidecar
        a served episode rides its terminal feedback out on is dropped for the same reason and
        must stay dropped — relaying it, as the single-episode server does, would reopen the
        channel this closes. A fixed payload also says nothing about how the stream classified the
        ending: a caller able to tell an unscored infrastructure failure from an earned zero has a
        reason to cause one.

        A call that ends the task answers with that payload *whatever happened while it ended*.
        An exception is a different response shape, and the shape is the channel: an env that
        published a clean summary on one verdict and a malformed one on the other would tell the
        agent its verdict through whether the call succeeded. So a failed seal — and an env that
        raises once it has already ended the episode — are recorded on the stream and answered
        with the same bytes as a clean one. Only a call that leaves the task *live* still raises:
        there the exception is the env's own answer to a call the agent can make again, no
        different in kind from the env text an ordinary call returns, and no task has ended for
        it to be a verdict about.

        At capacity 1 there is no routing lease to find: the env's own schemas are advertised
        verbatim, so a ``lease`` in ``arguments`` is the *env's* argument and is passed through
        untouched. Reading it here would take an env's own parameter away from it and refuse the
        call — with one slot the call is unambiguous, so nothing needs naming."""
        args = dict(arguments or {})
        if self._max_in_flight > 1:
            lease = lease if lease is not None else args.get(_LEASE_ARG)
            args.pop(_LEASE_ARG, None)
        resolved = await self._resolve(tool, lease)
        if isinstance(resolved, ToolResult):
            return resolved
        live = resolved
        # Outside the registry lock: the episode has its own lock, holding a stream-wide one
        # across an awaited env call would serialise the whole stream behind one tool, and the
        # deadline needs that lock to arbitrate — an env slow in a tool or in its finalizer must
        # not be able to spend the wall clock and still be recorded as an ordinary seal.
        cancellation = _Cancellation()
        try:
            call = await live.episode.call(tool, args)
        except BaseException as exc:  # noqa: BLE001 — see below; never re-raised at the agent
            # An env can raise `CancelledError` like anything else, and one raised *by the env*
            # is not this caller's cancellation (see `_must_propagate`). Told apart here rather
            # than by type, because letting it through skips everything below: the stop is never
            # recorded, so the queue serves on against an env that already lost an outcome; the
            # seal never runs; and the terminating call answers the agent with a traceback in
            # place of the constant every other ending returns.
            if _must_propagate(exc, cancellation):
                raise
            if not live.episode.terminated:
                # The call ended nothing, so it goes back as the env's own answer to a call the
                # agent can make again — but not on its own. It reached no result, so nothing the
                # agent asked for is on this episode's record, and the outcome the stream would
                # compose from that record is one the agent never got to play for. Left at a bare
                # re-raise this is the whole failure: the drain later drives the terminal itself
                # and files the task in a *scored* closure, so an agent whose submission the
                # harness dropped is recorded as one that answered wrong — a number a run's mean
                # would then average in. So the loss is kept on the entry and the seal decides —
                # kept and not acted on, because an agent that recovers and ends the task itself
                # has earned whatever that terminal says (see `_compose_row`). What it costs is
                # this task's score and nothing else: the stream serves on, since the next task
                # need not meet what this call met.
                if live.call_error is None:  # the first loss explains the row
                    live.call_error = exc
                raise
            # The episode ended and *then* the call failed: the terminal is committed and the
            # feedback the episode was about to hand over is what raised. So a row is still owed
            # — it lands carrying whatever feedback survived, which is what makes the loss
            # legible — and the stream stops. A row with no readable outcome is the same
            # eval-integrity failure whether the value is unusable or missing, and an env that
            # raises here raises for every task in the queue; without the stop a solved task is
            # recorded unscored and `aclose()` reports a clean run.
            # Rendered once, and guarded: an env's exception formats through the env's own code,
            # and letting that escape here would take the stop, the seal and the redacted answer
            # with it — see :func:`_rendered_failure`.
            rendered = _rendered_failure(exc)
            self._stop(
                exc,
                dispensing=(
                    f"this stream stopped: env {live.ref.env!r} raised while ending a task "
                    f"({rendered}), so that task's row carries no outcome "
                    "and no further task could be scored either"
                ),
                closing=(
                    f"this stream stopped before its queue was served: env "
                    f"{live.ref.env!r} raised while ending a task "
                    f"({rendered})"
                ),
            )
            await self._seal_redacted(live)
            return ToolResult(content=_TASK_OVER)
        if not call.terminated:
            return ToolResult(content=json.dumps({"content": call.content, "terminated": False}))
        if not call.tombstoned:
            # Only the call that actually ended the task may say how it ended. Every call that
            # arrives after the episode has ended is answered with a tombstone, and a tombstone
            # is `terminated` too — so a `terminate` racing an accepted `submit` returns first
            # (the submission is still in its finalizer), and taking the ending from whichever
            # response reaches the seal first would file the agent's earned, scored submission
            # as a task it aborted. Stamped synchronously, the moment the call returns, so
            # whoever claims the seal already sees it. The payload is not taken from here at
            # all: it is the core's to stamp, and `_record` reads it off the episode.
            live.terminal_tool = tool
        await self._seal_redacted(live)
        return ToolResult(content=_TASK_OVER)

    # ----- routing -----

    async def _resolve(self, tool: str, lease: Optional[Any]) -> Any:
        """Find the episode a call belongs to, or refuse it.

        A lease alone is not identity. At capacity above 1 every env tool is exposed at once, so
        a worker holding a valid lease for one task could name a tool belonging to another env or
        to no task at all. The registry therefore binds ``(lease, env, native tool)`` and checks
        all three **before** the call can reach the episode, where an unknown tool would consume
        a step of the budget and land in the trajectory.

        At capacity 1 a lease is ignored outright rather than merely unnecessary: only one
        episode can be live, so the call is unambiguous, and the word ``lease`` belongs to the
        env there — the stream advertises no wrapper to put a routing one in.

        A lease whose task is over is refused as *over*, whether its entry is still in the
        registry or has been retired down to the lease (see :meth:`_retire_settled`). The two
        answers are not interchangeable: ``unknown_lease`` says the stream never dispensed this,
        which would be a lie about a task the agent really was given, and about a task the row
        for it can be found under.

        Over includes ended by the *stream*: an orderly shutdown claims every unsettled task's
        seal in the critical section that closes the stream (see :meth:`aclose`), and the
        deadline claims an expired one the same way, so a call arriving after either finds a task
        that is over rather than a live episode. That is what keeps this refusal — which costs no
        budget and writes no row — from being a scored row the record files under a closure the
        agent earned.

        **Over is read from the task, not from who holds its claim.** A seal that failed on the
        storage hands its claim back so a later drain can retry the append (see
        :meth:`_join_seal`), and an entry read through the claim reads as live again the moment
        that happens — so the one task whose record has already failed becomes the one task a
        late call is routed to, on an episode this stream force-terminated, composed a row for
        and finalized every span of. During a shutdown that is a call accepted after the drain
        claimed the task, on the one entry the drain's own claim-everything-first rule cannot
        hold; outside one it is a second ending for a task already ended. It is also a *shape*
        the agent could read the record's failure off: every other finished task answers a late
        call with this refusal, and that one would answer with the terminating payload instead.
        So the read is :attr:`_Live.ended`, which a hand-back does not clear — and no check of
        ``_closed`` is needed beside it, because the drain claims every unsettled task in the
        critical section that closes the stream, and claiming is what sets that bit."""
        async with self._lock:
            if self._max_in_flight == 1:
                # One slot, so the call is unambiguous and the wire contract is unchanged. The
                # entry that answers is one whose task has not been ended — a stricter test than
                # an unheld claim, and the same one the lease branch below applies.
                live = next((it for it in self._live.values() if not it.ended), None)
                if live is None:
                    return _stream_error(
                        "no_active_task", f"no task is live; call `{_GET_TASK_TOOL}` first"
                    )
            elif lease is None:
                return _stream_error(
                    "missing_lease",
                    f"this call needs the `{_LEASE_ARG}` that `{_GET_TASK_TOOL}` returned, so "
                    "the stream knows which task it belongs to",
                )
            # Looked up once, and the entry that lookup found is the one used. `lease` is the
            # caller's own object and only has to be a `str` to arrive, so a membership test
            # followed by a subscript is two reads of a value nothing obliges to answer the same
            # way twice: the test passes, the subscript raises `KeyError`, and it raises where
            # nothing catches it — out of `dispatch`, dropping a call the agent made while its
            # task stays live for the drain to end and record as one the agent played and lost.
            # An unearned wrong answer, from a routing key. This is the read that was checked
            # (see :meth:`_require_framable` for the same rule about a spec's own values).
            elif (found := self._live.get(lease) if isinstance(lease, str) else None) is None:
                # A second read, and deliberately not the same hazard: it decides only which of
                # two refusals this is. Both cost no budget, enter no trajectory and write no
                # row, so a value that answers differently here mislabels a refusal rather than
                # standing in for an outcome.
                if isinstance(lease, str) and lease in self._settled_leases:
                    return _stream_error(
                        "sealed_lease",
                        f"that task is over; call `{_GET_TASK_TOOL}` for the next one",
                    )
                return _stream_error("unknown_lease", "no task was dispensed under this lease")
            else:
                live = found
                if live.ended:
                    return _stream_error(
                        "sealed_lease",
                        f"that task is over; call `{_GET_TASK_TOOL}` for the next one",
                    )
            if tool not in self._advertised_names(live.ref.env):
                # Either a tool of a different env, or one this task never advertised. Both are
                # the same failure: a lease that is valid but denotes the wrong thing.
                if any(tool in self._advertised_names(name) for name in self._advertised):
                    return _stream_error(
                        "wrong_env",
                        f"tool {tool!r} belongs to another env; this lease names a "
                        f"{live.ref.env!r} task",
                    )
                return _stream_error(
                    "tool_not_in_task", f"tool {tool!r} is not advertised by this task"
                )
            return live

    def _advertised_names(self, env_name: str) -> frozenset:
        return frozenset(m.name for m in self._advertised[env_name])

    def _mint_lease(self) -> str:
        """An opaque, unguessable lease that is never reused.

        **Bounded.** Redrawing until the value is fresh is the right loop and the wrong shape to
        leave unbounded: it is synchronous, so a source that cannot produce a fresh value spins
        inside the dispense lock with the event loop held — no task, no error, no deadline able
        to fire, and nothing for a harness to read. A run that cannot name its next task has to
        say so, and this is the only place that can. With a 128-bit CSPRNG and a set the size of
        a queue, one repeat is already beyond reach, so a whole run of them is not a collision:
        it is the source, and no further draw is going to fix it."""
        for _ in range(_LEASE_MINT_ATTEMPTS):
            lease = secrets.token_hex(16)
            if lease not in self._issued:
                self._issued.add(lease)
                return lease
        raise RuntimeError(
            f"this stream could not mint a lease no task has used: {_LEASE_MINT_ATTEMPTS} "
            "draws from `secrets.token_hex` all came back as values this run has already "
            "issued, so the source is not producing fresh ones"
        )

    # ----- the deadline -----

    def _start_watchdog(self) -> None:
        """Start the per-episode deadline watchdog (idempotent; needs a running loop, so it is
        started on first use rather than at construction)."""
        if self._deadline is None or self._watchdog is not None or self._closed:
            return
        self._watchdog = asyncio.ensure_future(self._watch_deadlines(self._deadline))

    async def _watch_deadlines(self, deadline: float) -> None:
        """Seal any episode that outlives the deadline. A hung agent never calls back, so this
        cannot be checked at the next request — it needs its own clock.

        It arbitrates through the stream's lock, which is why ``dispatch`` does not hold that
        lock across the env call it routes: with the lock held, the one case where a wall clock
        matters most — something is taking too long *right now* — is the one case this could
        never see.

        **Every expired task is claimed before any of them is waited on.** With more than one
        episode live, sealing is not a step this can take one task at a time: a seal drives the
        env's terminal, waits on its finalizer and runs every extension, any of which may block
        for as long as it likes. Claiming inside that loop would leave the tasks after the
        blocked one expired but unclaimed, and an unclaimed task is one :meth:`_resolve` still
        routes calls to — so an agent could earn a scored, ``sealed`` row on a task whose clock
        ran out, for exactly as long as some *other* task's env took to let go. So the claims are
        taken in the same critical section that finds them (see :meth:`_claim_seal`), which no
        await can split, and only the waiting happens afterwards.

        For the same reason the waiting does not happen *here*. A join is handed to its own task
        and the loop goes back to its clock, because a blocked seal must not stop the deadline
        being enforced on the episodes dispensed after it either — at capacity 1 a stuck seal
        blocks the next pull as well, so there is nothing left to time; above 1 the queue keeps
        moving, and a watch that stalled would leave every later task unclocked. Each task's
        deadline fires once, ``timed_out`` being the record of that: a seal that failed hands its
        claim back for the *drain* to retry, and a retry is not something a wall clock is owed.

        **A seal this cannot complete ends the watch, and is the stream's failure rather than
        this task's.** :meth:`_seal` has already recorded the stop, so the run is reported the
        way every other integrity failure is: by :meth:`aclose`, and by :attr:`stopped` before
        it. Letting it fail *this task* instead would report it nowhere the harness looks and
        break the release that awaits it — :meth:`aclose` would raise the raw error from its
        ``finally`` in place of the run-level explanation, having skipped part of what only the
        stream holds, and would do it again on every later attempt, since the failed task is
        what the release is claimed as. Ending the watch loses nothing: a stopped stream
        dispenses nothing more, and the drain finishes whatever is still live.

        The fallback ``_stop`` is belt and braces for a future edit that raises without recording
        one — the first cause wins, so it is a no-op on every path that exists today."""
        tick = max(0.005, min(0.25, deadline / 10))
        joining: Set["asyncio.Task[ResultRow]"] = set()
        timed_out: Set[str] = set()
        try:
            while True:
                await asyncio.sleep(tick)
                async with self._lock:
                    if self._closed:
                        return
                    now = time.monotonic()
                    for live in list(self._live.values()):
                        if live.sealed or live.lease in timed_out:
                            continue
                        if now - live.started < deadline:
                            continue
                        timed_out.add(live.lease)
                        joining.add(
                            asyncio.ensure_future(
                                self._join_seal(live, self._claim_seal(live, "timeout"))
                            )
                        )
                finished = {join for join in joining if join.done()}
                joining -= finished
                # Every one of them is read before any of them is acted on: leaving a finished
                # join un-inspected is asyncio's unretrieved-exception warning, and the loop
                # below leaves on the first failure it finds.
                failures = [
                    exc
                    for join in finished
                    if not join.cancelled()
                    for exc in [join.exception()]
                    if exc is not None and not isinstance(exc, asyncio.CancelledError)
                ]
                if failures:
                    self._stop(
                        failures[0],  # the first loss is the one that explains the run
                        dispensing=(
                            "this stream stopped: a dispensed task could not be sealed when "
                            "its deadline elapsed, so the run's record is missing an outcome"
                        ),
                        closing=(
                            "this stream could not seal every dispensed task; the run's "
                            "record is incomplete"
                        ),
                    )
                    return
        finally:
            # The seals themselves are shielded and unaffected; what is dropped here is only
            # this task's interest in them. Whatever is still owed is owed to the drain, which
            # joins every unsettled entry and reports what these would have.
            for join in joining:
                join.cancel()
            await asyncio.gather(*joining, return_exceptions=True)

    # ----- sealing -----

    async def _seal_redacted(self, live: _Live) -> None:
        """Seal a task whose terminating call is being answered with the redacted payload.

        The row :meth:`_seal` records is for the harness, not the caller — and so is the
        exception it raises instead. Every failure it can raise is already recorded on the stream
        before it leaves: whatever row did land is in ``results.jsonl``, :attr:`stopped` is set,
        the next dispense refuses, and :meth:`aclose` raises. Nothing is therefore lost by
        answering the agent with the constant, and raising instead would tell it precisely what
        this call is not allowed to tell it. Cancellation is a ``BaseException`` and still passes
        through: the caller it would have answered is already gone, and the claim hand-back needs
        it to.

        The fallback ``_stop`` is belt and braces for a future edit that raises without recording
        one — the first cause wins, so it is a no-op on every path that exists today."""
        try:
            await self._seal(live)
        except Exception as exc:  # noqa: BLE001 — reported on the stream, not to the agent
            self._stop(
                exc,
                dispensing=(
                    "this stream stopped: a dispensed task could not be sealed, so the run's "
                    "record is missing an outcome"
                ),
                closing=(
                    "this stream could not seal every dispensed task; the run's record is "
                    "incomplete"
                ),
            )

    async def _seal(self, live: _Live, *, forced: Optional[Closure] = None) -> ResultRow:
        """End the episode authoritatively, classify how it ended, record the row, and release
        the episode (and its env). Runs at most once per dispensed task.

        ``forced`` is set when the *stream* ended the task rather than the agent — an orderly
        drain or abandonment (``drained``) or the deadline (``timeout``).

        A terminal call, the deadline and the drain all race for this. They meet at one shared
        per-task transition — the seal itself, held as a task: the first claims it, the rest
        await that same task and take its outcome.

        The claim is the *running seal*, not merely a flag, because a caller can go away
        mid-seal. Its cancellation must not restart the work: a restarted seal calls each
        extension's ``finalize`` a second time, and it re-reads an episode the first attempt has
        already terminated, which reclassifies a task the stream drained as one the agent sealed
        or aborted itself — an outcome the agent never earned, recorded as one it did. So the
        caller's await is shielded: the seal runs to completion on its own and a later caller
        joins it. Only a seal that genuinely *failed* hands the claim back, so a later drain
        retries it rather than waiting forever on a transition that never completed — an entry
        left claimed with no row is invisible to a later drain, to ``get_task`` and to
        ``queue_info``, so its durable dispense would be reported as a crash that never
        happened.

        What that retry retries is the durable append and nothing above it. Both objections in
        the paragraph above are just as true of a seal that failed on the storage as of one whose
        caller was cancelled, so the composed row is retained on the entry across the hand-back
        and the retry starts at the write (see :meth:`_run_seal`).

        **What a later caller joins is the claim, never the row.** The row becomes durable in the
        middle of the seal — the episode is still open behind it and any stop it owes is still
        unrecorded — so answering with it there would let a drain report a shutdown complete
        while an episode was still releasing, and let the next dispense take a slot over a stop
        nobody had published yet. The claim covers the whole seal, tail included, so joining it
        is joining that tail too; once the seal is over, joining it is free and hands back the
        same row it recorded.

        A failed seal also stops the stream (see :meth:`_require_open`): the row it
        was going to write is lost, and every further task would be served over that hole. So
        does a row that landed but cannot be *summarized*: a ``success``/``reward`` that is
        wrong-typed, or published twice, is a property of the env rather than of the task, so it
        would recur for the whole queue.

        Claiming and joining are separable, and the deadline and the drain both separate them
        (see :meth:`_claim_seal`, :meth:`_watch_deadlines` and :meth:`aclose`): a caller that has
        several tasks to end must be able to take every claim before it waits on any of them.
        This method is the pair for a caller that has exactly one."""
        async with self._lock:
            sealing = self._claim_seal(live, forced)
        return await self._join_seal(live, sealing)

    def _claim_seal(
        self, live: _Live, forced: Optional[Closure]
    ) -> "asyncio.Task[ResultRow]":
        """Take this task's one seal, or hand back the claim someone else already holds.

        **Synchronous, and called with the registry lock held**, which is what makes it usable
        for more than one task at a time: the scan that decides a task must be sealed and the
        claim that stops anyone else acting on it happen in the same critical section, with no
        await between them. A claimant with several tasks to end therefore takes every claim
        first and waits afterwards, so one task's seal — which drives an env terminal, an env
        finalizer and every extension, and may block on any of them — cannot leave another's
        unclaimed and still answerable (see :meth:`_watch_deadlines` for the deadline's use of
        this and :meth:`aclose` for the drain's).

        It also records that this task has been *ended*, which the claim itself cannot say: the
        claim is handed back when a seal fails, and what has already happened by then — the
        forced terminal, the composed row, every span's ``finalize`` — has not been undone. That
        bit never clears, and it is what :meth:`_resolve` refuses a late call on."""
        sealing = live.sealing
        if sealing is None:
            sealing = live.sealing = asyncio.ensure_future(self._run_seal(live, forced))
            live.sealed = True
            live.ended = True
            # Bookkeeping only: a seal that fails while its caller is being cancelled has no
            # awaiter at that instant, and asyncio would log it as unretrieved (see
            # `_mark_retrieved`). The failure itself is still reported the usual way, by the
            # next caller to join this same task.
            sealing.add_done_callback(_mark_retrieved)
        return sealing

    async def _join_seal(
        self, live: _Live, sealing: "asyncio.Task[ResultRow]"
    ) -> ResultRow:
        """Wait on a claimed seal without being able to disturb it, and answer for it if it
        failed: hand the claim back so a later drain retries the append, and stop the stream.

        What goes back is the *claim* and nothing else. This task has been ended — its episode
        force-terminated, its row composed, its spans finalized — and no retry undoes any of
        that, so :attr:`_Live.ended` stays set and a call naming it is still refused as over.

        Split from :meth:`_claim_seal` so the two can happen at different moments; joining is
        what may block, and nothing here holds the registry lock while it does."""
        try:
            return await asyncio.shield(sealing)
        except BaseException:
            failure = (
                sealing.exception()
                if sealing.done() and not sealing.cancelled()
                else None
            )
            if failure is not None:
                async with self._lock:
                    if live.sealing is sealing and live.row is None:
                        live.sealing = None
                        live.sealed = False
                # The seal *failed* rather than being deferred, so a row is lost: stop
                # dispensing, or the rest of the queue is served over a record missing an
                # outcome the agent earned. A merely cancelled seal never reaches here — that
                # one is finished by whoever drains next.
                self._stop(
                    failure,  # the first loss is the one that explains the run
                    dispensing=(
                        "this stream stopped: a dispensed task could not be recorded to "
                        f"{self.results_path}, so the run's record is missing an outcome the "
                        "agent actually earned"
                    ),
                    closing=(
                        "this stream could not record every dispensed task to "
                        f"{self.results_path}; the run's record is incomplete"
                    ),
                )
            raise

    async def _release(self, live: _Live) -> None:
        """Let go of a task's episode (and its env), leaving its registry entry in place.

        Separate from recording it, and reached whether or not the row landed: the episode owns
        MCP sessions and an env, this entry is the only handle on either, and a seal that failed
        is not retried.

        **Claimed once, and every later arrival joins that same release** — the shape
        :meth:`_released` uses for the stream's own teardown and :meth:`_seal` for the seal,
        applied per episode. Two callers can reach here for one entry now that the entry outlives
        its release: a seal publishes the stop it owes *after* letting the episode go, so whoever
        joins that seal arrives behind a release already running, and a retry of a failed append
        runs the tail a second time. A flag would stop the second close but let its caller past a
        teardown still in flight — which is the same 'finished with the slot' mistake one level
        down, and would let a dispense proceed while an env was still closing. So the second
        caller waits on the first, and a cancelled one leaves it running rather than taking it
        along."""
        releasing = live.releasing
        if releasing is None:
            releasing = live.releasing = asyncio.ensure_future(_close_episode(live))
        await asyncio.shield(releasing)

    def _retire_settled(self, live: _Live) -> None:
        """Drop a finished task's entry from the registry, keeping only its lease.

        What the entry held was an episode — its env, its MCP sessions, the task payload, the
        trajectory and every tool result in it, the provenance spans and what they observed —
        and none of that is answerable to anyone once the seal's tail has run. The row is already
        in ``results.jsonl`` and on :attr:`results`; the episode is closed. Keeping the entry for
        the sake of the one answer still owed would make a run's memory grow with every task it
        had already finished, which is what :meth:`_run_seal` retires it to avoid.

        The one answer still owed is that a call naming this lease is naming a task that is over
        rather than one that never existed (see :meth:`_resolve`), and that needs the lease
        string alone. Everything else a settled entry was ever asked is a constant for it: it is
        sealed, it is released, and it is settled, so it counts toward no capacity, is drained by
        nobody, and is never the episode a capacity-one call resolves to. An absent entry answers
        all three the same way the entry did.

        **Synchronous on purpose.** It runs from :meth:`_run_seal`'s ``finally``, where a
        cancellation may already be pending, so it may not be allowed to await — a retirement
        that could be skipped there would leave exactly the entry it exists to remove. Taking the
        registry lock is what it would have to await for, and it does not need to: every scan of
        the registry is itself synchronous from its first read to its last, so two statements
        with no await between them cannot land in the middle of one."""
        self._live.pop(live.lease, None)
        self._settled_leases.add(live.lease)

    async def _run_seal(self, live: _Live, forced: Optional[Closure]) -> ResultRow:
        """The claimed seal: end the episode, classify, finalize the spans, record the row.

        Two phases, and only the second is retryable. Composing the row ends the episode, reads
        its verdict and runs every extension's ``finalize`` — each of those happens once per
        dispensed task and none of them can be undone. Recording it is a durable append that can
        fail on the storage and be retried by a later drain, which is what the claim hand-back in
        :meth:`_seal` exists for.

        So the composed row is retained on the entry across that hand-back. A retry that found no
        row would compose a second one, and both halves of that are wrong: every ``finalize``
        would run again — its snapshots and commits are already out in the world and no row says
        it ran twice — and the classification would be taken from an episode this attempt has
        already force-terminated, so a task the stream drained would be recorded as one the agent
        sealed or aborted itself, in the scored closures, exactly as a restarted seal would (see
        :meth:`_seal`). The retry therefore starts at the append.

        If no retry ever comes — the caller abandons the stream without closing it — the row is
        lost with the process, as it is today: nothing durable was written, so the dispense record
        goes unanswered and :func:`reconcile` reports the crash it actually was. What the retained
        row buys is that the extensions are not asked to produce their side effects a second time
        for a row that may never land.

        **The registry entry outlives the append**, which is the middle of the seal and not its
        end: the episode is still open behind the row, and the stop an unheadlinable summary or a
        failed terminal owes is recorded after the release, not before it. An entry that stopped
        being findable there would be a seal nobody can join — the drain would think it had
        drained everything and return while an episode was still releasing, and the next dispense
        would take a slot over a stop that has not been published yet, which is the whole point of
        stopping. What a caller joins is therefore the claim, and :attr:`_Live.settled` is how the
        rest of the stream tells a finished seal from one still in its tail.

        **It does not outlive the seal.** Once the tail has run there is nothing left to join,
        nothing left to release and no stop left to publish, and the only thing anyone can still
        ask about this task is whether its lease was real — which is a question about the lease
        and not about the episode behind it. So the entry is retired here and the lease alone is
        kept (see :meth:`_retire_settled`); holding the entry instead would keep a finished
        episode's env, trajectory, tool output, spans and session objects reachable for the
        length of the run, growing a long queue's memory by every task it has already scored."""
        try:
            row = live.pending_row
            if row is None:
                row = live.pending_row = await self._compose_row(live, forced)
            async with self._lock:
                # Durable before the row counts anywhere else. `reconcile` reads a missing result
                # as a crash, so a row that only reached the page cache would turn a sealed,
                # scored task into a `broker_abort` after a host crash — an outcome the agent
                # earned, reported as an infrastructure failure. Everything in here is
                # synchronous, so no cancellation point can split the write from the claim it
                # makes.
                _append_jsonl(self.results_path, row.to_wire(), durable=True)
                # What the run keeps is the row *the file now holds*, re-read from its own wire
                # form. That is the canonical snapshot every reader is shown a copy of (see
                # :attr:`results`), and taking it here is what makes those copies cheap and
                # certain: they copy plain data, run no env code, and cannot disagree with the
                # record. Held any other way this list would carry the env's own objects — the
                # feedback values it published, the one list that `observed` and `score.feedback`
                # both are, and whatever an extension put on `extensions` — and every view of it
                # would be a handle on them.
                #
                # It cannot fail here, and that is why it is here rather than a step earlier: the
                # append above just serialised these exact values, with the same encoder and the
                # same `allow_nan`, so a normalization that ran before the write could suppress a
                # row this run has already committed to. After it, there is nothing left to find
                # out.
                recorded = _recorded_row(row)
                live.row = recorded
                live.pending_row = None
                self._results.append(recorded)
                self._done_positions.add(live.position)
                # The entry stays in the registry through the tail below: the seal is not over at
                # the append, and a dispense or a drain that means to join it has to be able to
                # find it (see :attr:`_Live.settled`). It is retired in the `finally`, once there
                # is nothing left for anyone to find it for.
            # The row is committed and fsync'd; letting the episode go is best-effort, and it is
            # the same release a drain would run for an entry whose seal failed (see
            # :meth:`_release`), so whichever gets there first is the only one that runs.
            await self._release(live)
            # A promoted `call_error` is deliberately **not** one of these (see
            # :meth:`_compose_row`), and the source is what says which this is — never the
            # exception object, which an env may raise on both boundaries. What a lost call owed
            # the record is the row it already produced: unscored, saying the call was lost, which
            # is the whole property — an outcome nothing the agent did produced cannot be averaged
            # into a benchmark. Stopping on top of that spends the rest of the queue on one lost
            # call, and the failure it names is one the *next* task need not have: a mid-episode
            # call is where a transient fault lands, and a session that hiccups once would end a
            # 480-task run. A terminal that failed is the opposite case and still stops here —
            # there the env is on its way out of a task it had already ended, and every task of
            # that env leaves the same way.
            if live.terminal_error is not None and live.terminal_error_source == "terminal":
                exc = live.terminal_error
                # Guarded, and for more than tidiness: this runs *after* the append, so an
                # unguarded format would leave a durable row standing beside a stop that was
                # never published, and the queue would serve on against the env that failed.
                rendered = _rendered_failure(exc)
                self._stop(
                    exc,
                    dispensing=(
                        f"this stream stopped: env {live.ref.env!r} failed while the stream "
                        f"ended a task ({rendered}), so that task's row carries "
                        "no outcome and no further task could be scored either"
                    ),
                    closing=(
                        f"this stream stopped before its queue was served: env "
                        f"{live.ref.env!r} failed while the stream ended a task ({rendered})"
                    ),
                )
            if live.summary_error is not None:
                # The row landed first — `observed` carries every offending item verbatim, so the
                # evidence is durable rather than confined to an exception string — and the
                # episode is already released. Only then does the stream stop. The order is not
                # cosmetic: the row is what makes this seal final, so nothing above may raise
                # ahead of it and strand the episode. What the order does *not* license is
                # answering a joiner as soon as the row is there: this stop is still owed, and
                # until it is published a dispense that stepped over it would serve the rest of
                # the queue against an env whose headline this record has already refused.
                self._stop(
                    live.summary_error,
                    dispensing=(
                        f"this stream stopped: env {live.ref.env!r} published a summary value "
                        f"this record cannot headline ({live.summary_error}), so no further "
                        "task can be scored against it"
                    ),
                    closing=(
                        f"this stream stopped before its queue was served: env "
                        f"{live.ref.env!r} published a summary value this record cannot "
                        f"headline ({live.summary_error})"
                    ),
                )
                raise live.summary_error
            # The recorded snapshot, so one seal produces one row wherever it is read from.
            return recorded
        finally:
            # Only a task whose row is durable is finished with; one whose append failed keeps
            # its entry, because the claim is handed back and a later drain retries the write.
            if live.row is not None:
                self._retire_settled(live)

    async def _compose_row(self, live: _Live, forced: Optional[Closure]) -> ResultRow:
        """Everything a seal does exactly once: end the episode, read its verdict, classify it,
        close the spans, and build the row from all of it. Nothing here is retried."""
        episode = live.episode
        # A terminal call whose caller was cancelled leaves its finalization running: the episode
        # is already sealed while its verdict is still landing. Adopt that outcome rather than
        # forcing a second terminal over the top of it — a forced call on a sealed episode reads
        # the post-seal tombstone, which would file an earned, scored submission as a task the
        # stream drained and drop the real result on the floor.
        await episode.wait_finalized()
        drove = False
        if not episode.terminated:
            # The agent stopped short. Drive the env's own terminal on its behalf so the row
            # carries an authoritative outcome rather than a guess: the score terminal first
            # (a sealed partial state can still earn partial credit), falling back to the
            # reserved abort when the env's score terminal needs arguments we cannot invent.
            score_terminal = self._score_terminal.get(live.ref.env)
            refused: Optional[BaseException] = None
            if score_terminal is not None:
                drove, refused = await self._force_terminal(live, score_terminal)
            if not episode.terminated:
                ended, abort_refusal = await self._force_terminal(live, TERMINATE_TOOL_NAME)
                drove = drove or ended
                # The first refusal is the one that explains the run, and the fallback's own
                # answer may not stand in for it. An abort that succeeds does not make the score
                # terminal's raise a non-event: that terminal is the only call that can produce a
                # verdict for this env, so what the abort ended is a task with nothing behind it,
                # and letting the reassignment drop the refusal would record the abort's
                # fail-closed `correct=False` as an outcome — for a queue in which every task of
                # that env will refuse the same way.
                if refused is None:
                    refused = abort_refusal
            # A forced call can be tombstoned too: the agent's own terminal may have been waiting
            # on the episode's lock behind an ordinary call, so it sealed while this one queued
            # behind it. Then nothing was forced, its verdict is still landing, and the wait
            # above already returned.
            await episode.wait_finalized()
            if refused is not None:
                # A terminal this stream drove raised instead of answering, so no verdict exists
                # to read off the episode — whether the fallback then brought the task to an end
                # or nothing did. That is the same failure as a terminal that ended the task and
                # then raised, one step earlier, and the row is owed the same answer, because an
                # agent that ended nothing did not end this.
                live.failed_to_end(refused, "terminal")
        if drove:
            forced = forced or "drained"
            # The stream ended this task, and a call the agent made had already failed without
            # reaching a result (see :meth:`dispatch`). The closure about to be recorded is a
            # *scored* one, so the row would say the agent played the task out and got it wrong,
            # for a call the harness never carried. Same finding as a terminal that could not be
            # driven, one call earlier, and the row gets the same answer — the run does not, and
            # the source recorded beside it is what tells the two apart (see :meth:`_settle`). An
            # unscored closure needs none of this: it already says the outcome was not earned.
            if forced in _SCORED_CLOSURES and live.call_error is not None:
                live.failed_to_end(live.call_error, "promoted_call")
        elif forced == "drained":
            # Nothing was forced after all: the agent ended this task itself and only its verdict
            # was still landing when the drain reached it. Record how it actually ended — calling
            # it drained would file an earned outcome as one the stream imposed. (The deadline
            # keeps its claim: a task whose clock ran out is the deadline's to classify.)
            forced = None

        # How it ended, taken from the episode itself when the call that ended it never made it
        # back to `dispatch` — cancelled mid-flight, driven by the stream just above, or answered
        # to a caller whose own call ended nothing. The payload is always the episode's: it is
        # the core's stamped verdict, and nothing else may stand in for it.
        if live.terminal_tool is None and episode.terminated:
            live.terminal_tool = episode.terminal_tool or (
                TERMINATE_TOOL_NAME if episode.terminal_source == "abort" else None
            )
        live.terminal_payload = episode.terminal_payload

        # The env's items, in the order and at the levels it published them. Keyed by name this
        # would be a *projection*, not a record: an env may publish a name twice, and a mapping
        # silently keeps one of them — losing the evidence and, for a summary name, deciding the
        # headline by list order (see `_pick_summary`).
        observed = [dict(item) for item in episode.terminal_feedback]
        closure, diagnostic = self._classify(live, forced)
        # The summary is read as a unit and strictly (see `_pick_summary`): a wrong-typed
        # `success`/`reward` leaves the row unscored rather than coerced or quietly emptied.
        # `closure` is untouched — how the task *ended* is a different question from whether
        # the env's headline is readable, and `observed` keeps every offending item either way.
        # `diagnostic` is what makes this legible in the file itself, which is the one thing
        # this commit can say that the one below it cannot.
        score: Optional[Score] = None
        if closure in _SCORED_CLOSURES:
            try:
                score = Score(
                    reward=_pick_float(observed, _REWARD_NAMES),
                    success=_pick_bool(observed, _SUCCESS_NAMES),
                    feedback=observed,
                )
            except _MalformedSummary as exc:
                live.summary_error = exc
                diagnostic = f"the env published a summary value this record cannot headline: {exc}"
        # Extensions run here — after the outcome is decided, before the row is written, and
        # outside every lock. Whatever they return is namespaced; whatever they do wrong is
        # recorded in their own namespace and cannot stop the row. What they are *handed* is
        # built per extension and detached from `score`, which is the row's own summary object.
        extensions = await self._finalize_spans(live, closure, score)
        return ResultRow(
            seq=live.seq,
            lease=live.lease,
            position=live.position,
            env=live.ref.env,
            task_idx=live.ref.task_idx,
            closure=closure,
            score=score,
            observed=observed,
            diagnostic=diagnostic,
            extensions=extensions,
        )

    async def _force_terminal(
        self, live: _Live, tool: str
    ) -> Tuple[bool, Optional[BaseException]]:
        """Drive one terminal on the agent's behalf. Reports whether this call is what ended the
        task, and — only when it raised *without* ending it — what it raised, so the caller can
        fall back to the next terminal.

        **A call that raises after ending the episode has not ended nothing.** The terminal is
        committed and the env is what failed, so the task is over with no verdict standing behind
        it: the exception is kept on the entry, the row is classified as a failed terminal
        transaction, and the stream stops. Swallowing it instead reads the missing return value
        as evidence the *agent* ended the task, which files an infrastructure failure as a clean
        agent seal — with ``success`` null, which in a run that finished means the env published
        no such field — and lets the queue serve on against an env that will raise for every task
        in it. It is the same failure :meth:`dispatch` already refuses to let an env answer with;
        the only difference is who made the call.

        Only a raise that left the episode still OPEN is handed back. That one is the refusal
        this fallback exists for — an env whose score terminal needs arguments the stream cannot
        invent — and the reserved abort is what answers it.

        A ``CancelledError`` the env raises is that env failing and is classified with the rest
        (see :func:`_must_propagate`). Letting it through instead cancels the *seal task* this
        runs in — the seal is its own task precisely so a lost caller cannot restart it — leaving
        a claim nothing can ever complete: no row is composed, the entry keeps a cancelled seal,
        and every later drain re-awaits it and raises the same cancellation, so an orderly
        shutdown reconciles as a crash."""
        cancellation = _Cancellation()
        try:
            call = await live.episode.call(tool, {})
        except BaseException as exc:  # noqa: BLE001 — classified above, never raised at the agent
            if _must_propagate(exc, cancellation):
                raise
            if not live.episode.terminated:
                return False, exc
            live.failed_to_end(exc, "terminal")
            return True, None
        return (call.terminated and not call.tombstoned), None

    # ----- provenance spans -----

    async def _begin_spans(
        self, ref: TaskRef
    ) -> Tuple[Dict[str, ProvenanceSpan], Dict[str, Any]]:
        """Open one span per extension, before the task is exposed.

        An extension that raises, hangs, or hands back something that is not strict JSON stops
        the dispense: the task is never handed out and its queue position is still owed. Failing
        loudly here is the honest option — a run whose provenance is silently missing for some
        tasks is worse than one that stops."""
        spans: Dict[str, ProvenanceSpan] = {}
        dispensed: Dict[str, Any] = {}
        # See `_Cancellation` for why cancellation is counted rather than caught by type. Here
        # the task being watched is the caller's own — whoever is dispensing — so a cancellation
        # aimed at *it* still ends the dispense, and one an extension merely raised is that
        # extension failing to open its span, which is what `ProvenanceError` already says.
        cancellation = _Cancellation()

        def refused(namespace: str, cause: BaseException) -> ProvenanceError:
            # The cause is the extension's own object, so describing it is guarded: an unguarded
            # format would hand the caller the formatter's failure in place of the
            # `ProvenanceError` this promises, which is the one thing every caller of `get_task`
            # is told to expect from a span that would not open.
            return ProvenanceError(
                f"provenance extension {namespace!r} failed to open a span for "
                f"{ref.env}[{ref.task_idx}]: {_rendered_failure(cause)}"
            )

        # The namespace is the one the constructor validated, never a fresh read of the
        # extension's attribute: uniqueness and non-emptiness were checked against those strings,
        # and re-reading here would key the row by whatever the object says now. Two extensions
        # that end up agreeing on a name would silently overwrite each other's span — one key on
        # the row, one `finalize` never called — and an emptied one would key a row by "".
        for namespace, extension in self._provenance:
            try:
                span = await self._with_timeout(extension.begin(ref))
                observed = _strict_json_object(span.dispensed)
            except BaseException as exc:  # noqa: BLE001 — reported, never swallowed
                if _must_propagate(exc, cancellation):
                    raise
                raise refused(namespace, exc) from exc
            spans[namespace] = span
            dispensed[namespace] = observed
        return spans, dispensed

    async def _finalize_spans(
        self, live: _Live, closure: Closure, score: Optional[Score]
    ) -> Dict[str, Any]:
        """Close every open span and collect its namespaced output.

        The episode is already sealed and scored, so a failing extension cannot change the
        outcome: it gets a structured error under its own namespace and the row is written
        regardless. An extension can neither suppress a row nor create a second one.

        Each span is handed its **own** :class:`CompletedTask`, built from a detached copy of the
        summary. One shared object would be a hole in exactly that guarantee — ``frozen=True``
        does not freeze the list behind ``Score.feedback``, and that list *is* the row's
        ``observed`` — and a shared one would also let the first extension decide what the second
        one observes.

        **Cancellation is told apart by where it was requested, not by its type** (see
        :class:`_Cancellation`). The seal runs as its own task precisely so a lost caller cannot
        restart it, which means a ``CancelledError`` arriving here is one of two unrelated things:
        this seal task being cancelled, or an extension raising one. Unmoved count, the extension
        raised it and it is that extension's failure, recorded in its namespace like any other;
        moved, the seal itself is being cancelled and it must propagate — swallowing that one
        would write a row for a seal its owner cancelled. (Expiring a hung extension does not move
        it either: the callback runs in a task of its own and the bound cancels *that* one, never
        this — see :meth:`_with_timeout` — so a timed-out span arrives as the ``TimeoutError`` it
        is, through the same handler below.)

        Without the count, an extension could raise ``CancelledError`` and cancel the seal task
        out from under the row: ``_seal`` would see a cancelled claim, record no stop, and leave
        the entry sealed with no row, so every retry re-awaits the same cancelled task, the
        durable dispense is never answered, and an orderly shutdown reconciles as a crash."""
        extensions: Dict[str, Any] = {}
        cancellation = _Cancellation()
        for namespace, span in live.spans.items():
            entry: Dict[str, Any] = {"dispensed": live.dispensed_extensions.get(namespace)}
            try:
                # Built inside the boundary, not above it. Detaching the summary is the one step
                # here that touches the *row's* objects rather than the extension's, and above
                # the boundary a failure in it is uncontained: no row at all, and a second
                # `_compose_row` that finalizes again every span that had already closed. It is
                # detached by serialisation precisely so it cannot fail (see
                # :func:`_detached_summary`) — and it is built here so that a row does not
                # depend on that argument holding.
                completed = CompletedTask(
                    position=live.position, closure=closure, score=_detached_summary(score)
                )
                entry["sealed"] = _strict_json_object(
                    await self._with_timeout(span.finalize(completed))
                )
            except BaseException as exc:  # noqa: BLE001 — an extension may not break a row
                if _must_propagate(exc, cancellation):
                    raise
                entry["error"] = _rendered_failure(exc)
            extensions[namespace] = entry
        return extensions

    async def _with_timeout(self, awaitable: Any) -> Any:
        """Run one extension callback in a task of its own, and **stop waiting on it** at the
        bound. A callback that hangs must not wedge the queue.

        The bound is on how long the *harness* waits, which is the only thing a bound can be
        in-process. ``asyncio.wait_for`` is deliberately not what enforces it: it expires a
        callback by cancelling the task the callback is running in and then awaiting that
        cancellation to finish — and ``CancelledError`` is catchable, so asyncio documents that a
        callback which suppresses it has its value returned instead. A callback that catches it
        and carries on is therefore accepted past the deadline, and one that catches it and never
        returns holds the waiter for ever: at ``begin`` that is the dispense lock and so the whole
        queue, and at ``finalize`` it is the seal, so ``aclose`` never returns either.

        So the callback gets its own task and this waits on that task with a timeout that touches
        nothing else. On expiry the task is cancelled and let go (see :func:`_abandon`) instead of
        awaited, and the callback is recorded as the failure it is. Its own task is also the
        smallest surface that can be handed to arbitrary code: a callback can now only cancel
        *itself* through ``asyncio.current_task()``, and a cancellation requested against the
        seal reaches this caller rather than being catchable inside the extension — which is what
        :meth:`_finalize_spans` tells the two kinds of ``CancelledError`` apart by.

        **What is not bounded, stated exactly, because in-process nothing can bound it.** A
        callback that never yields to the event loop — a synchronous loop, or blocking I/O —
        stalls the loop itself, so no timer fires and there is nothing left running to pre-empt
        it; the same is true of the synchronous ``span.dispensed``. And an abandoned callback is
        cancelled, not stopped: it may catch the cancellation and run on, holding whatever *it*
        holds. That reaches one thing outside itself, and only one: ``asyncio.run`` awaits every
        outstanding task before it returns, so a callback that refuses to end delays the *loop's*
        shutdown after this stream has already answered and closed. That belongs to whoever owns
        the loop, and no bound this stream could set would change it. What an abandoned callback
        cannot do is reach anything of this stream's — see :func:`_abandon`."""
        call = asyncio.ensure_future(awaitable)
        # `asyncio.wait` reports on the callback without joining its fate to this caller's: it
        # never cancels this task, never raises what the callback raised, and a timeout leaves
        # the callback merely unfinished rather than waited on.
        try:
            done, _ = await asyncio.wait((call,), timeout=self._provenance_timeout)
        except BaseException:
            # Only this caller's own cancellation reaches here. The callback goes with it: it was
            # opened for work nobody is waiting for any more.
            _abandon(call)
            raise
        if not done:
            _abandon(call)
            raise TimeoutError(f"the callback did not return within {self._provenance_timeout}s")
        # Raises whatever the callback raised, in this task, without moving its cancel count —
        # a callback that raised `CancelledError` leaves its own task cancelled, and reading its
        # result here is what turns that into this task seeing the exception it raised.
        return call.result()

    def _classify(self, live: _Live, forced: Optional[Closure]) -> Tuple[Closure, Optional[str]]:
        """Decide how this task ended.

        Deliberately without the env's feedback in hand. How a task ended is the harness's own
        question — which of the deadline, the drain, the agent's terminal or the core's failed
        transaction closed it — and every one of those answers is already held on the entry or
        on the episode. The env's published items decide the row's *summary*, one step further
        down and behind :func:`_pick_summary`, and that separation is what keeps a reserved
        feedback name from being able to steer the closure (see :meth:`_finalize_failed`).

        A failed terminal transaction outranks everything except the deadline that caused it:
        the env published no verdict it stands behind, so the outcome was not earned however the
        task was ended. A terminal the stream drove and the env then failed is the same answer
        reached without a transaction to stamp it — see :meth:`_force_terminal`."""
        if forced == "timeout":
            return "timeout", "the per-episode deadline elapsed before the task was sealed"
        if live.terminal_error is not None:
            exc = live.terminal_error
            if live.terminal_error_source == "promoted_call":
                # The failure the entry kept from a call that never reached a result, promoted
                # once the stream had to end the task itself (see :meth:`_compose_row`). The env
                # failed *during* the task rather than on its way out, so the row says that
                # instead — the finding is the same either way, and it is the row that has to be
                # readable by whoever has to act on it.
                return (
                    "finalize_error",
                    f"env {live.ref.env!r} failed on a call the agent made "
                    f"({_rendered_failure(exc)}) and the stream then ended the task; the agent "
                    "never played it out",
                )
            return (
                "finalize_error",
                f"env {live.ref.env!r} failed while the stream ended the task "
                f"({_rendered_failure(exc)}); it published no verdict",
            )
        if self._finalize_failed(live):
            return (
                "finalize_error",
                "the terminal transaction failed closed; the env published no verdict",
            )
        if forced is not None:
            return forced, None
        if live.terminal_tool == TERMINATE_TOOL_NAME:
            return "aborted", None
        return "sealed", None

    @staticmethod
    def _finalize_failed(live: _Live) -> bool:
        """Did the terminal transaction fail closed?

        Answered by the core alone. Every seal-enabled episode's sanitized terminal payload
        carries a ``finalize_error`` stamped from :attr:`TerminalEvidence.finalize_error` — a
        real ``bool``, written over whatever the env's own verdict had under that name — so the
        stamp is present on every episode that ran a terminal transaction and is the value the
        finalizer committed. A non-seal episode runs no such transaction, has no stamp, and
        never fails this way.

        **An env's own ``finalize_error`` feedback item is deliberately not consulted.** It is a
        reserved name the core already owns, so it could only agree with the stamp or contradict
        it — and consulting it means reading an :data:`~hgym.types.EpisodeFeedbackValue`, which
        the wire legally permits to be text or a number. ``bool("false")`` is ``True``, so an env
        reporting a clean finalization in the wrong type would have its solved, scored task filed
        as an unscored infrastructure failure, with a diagnostic saying the transaction failed.
        That is the coercion :func:`_pick_summary` refuses for the summary names, for the same
        reason, and refusing it here too leaves ``_pick_summary`` as the **only** path from a
        published value to a number on this row. The item itself stays in ``observed`` verbatim,
        as evidence — it is simply not what decides how the task ended."""
        payload = live.terminal_payload or {}
        return payload.get("finalize_error") is True

    # ----- bookkeeping -----

    def _next_position(self) -> Optional[int]:
        """The next queue position to play, skipping ones a resumed run already sealed."""
        position = self._position
        while position < len(self._queue) and position in self._done_positions:
            position += 1
        self._position = position
        return position if position < len(self._queue) else None

    def _write_dispense(self, live: _Live) -> None:
        """The durable record that makes a crash reconcilable. fsync'd: a record that only
        reaches the page cache is exactly the record a hard kill loses.

        It carries what each extension observed **before** the task was handed out, because this
        record is the only artifact that survives a kill and that observation is the only half of
        a span a crash can leave behind. Without it the guarantee is lopsided: provenance for
        every orderly closure and none for the one closure that is *about* not exiting in an
        orderly way, so a snapshot an extension had already taken is disconnected from the
        dispense that caused it. On a resumed run that is the difference between an orphaned
        snapshot and one the ``broker_abort`` row names (see :func:`reconcile`).

        **The payload is captured before this record is composed, not while it is written.** It
        is ``live.dispensed_extensions``, built in :meth:`_begin_spans` — before the episode was
        opened, before the registry lock, and long before here — so no extension code runs at or
        after this point and nothing here can be blocked, delayed or refused by one. An extension
        that fails still fails where it always did, at ``begin``, where it costs a dispense that
        was never recorded rather than one that was.

        It also cannot make this append fail. :func:`_strict_json_object` already serialised each
        value with ``json.dumps(..., allow_nan=False)`` — the same encoder and the same setting
        :func:`_append_jsonl` uses — and returned the *re-parsed* result, so what is written here
        is a detached copy that the exact call about to encode it has already encoded once. The
        ordering below it is untouched: still one synchronous append inside the same critical
        section, still the last thing before the clock starts and the counters commit, with no new
        suspension point between the record landing and the bookkeeping that answers for it."""
        _append_jsonl(
            self.dispenses_path,
            {
                "lease": live.lease,
                "seq": live.seq,
                "position": live.position,
                "env": live.ref.env,
                "task_idx": live.ref.task_idx,
                "dispensed_at": time.time(),
                # namespace -> what that extension observed at dispense. The `{"dispensed": ...}`
                # wrapper the row carries is deliberately *not* here: it exists to sit beside
                # `sealed`/`error`, which are outcomes, and a dispense record has no outcome.
                # `reconcile` puts it on, so exactly one place knows the row's shape.
                "extensions": live.dispensed_extensions,
            },
            durable=True,
        )

    def _require_open(self) -> None:
        if self._stopped is not None:
            raise RuntimeError(self._stopped.dispensing) from self._stopped.cause
        if self._closed:
            raise RuntimeError("this stream is closed")


def _recorded_row(row: ResultRow) -> ResultRow:
    """A composed row, re-read from the wire form the results file just committed — the run's
    own canonical copy of what it recorded.

    Composing a row leaves the env's values on it: ``observed`` holds the items the episode
    published, and ``Score.feedback`` *is* that same list. So the row a seal builds is a handle
    on env objects, and this is where the run stops holding one. Taken through the same strict
    JSON the file holds — the same encoder and the same ``allow_nan`` :func:`_append_jsonl`
    committed it with — so what stays in memory and what a later :func:`read_results` reads back
    are the same row, and every view taken of it afterwards copies plain data (see
    :func:`_detached_row`).

    ``extensions`` comes through it too, and it has to: each namespace's halves are already
    strict JSON (:func:`_strict_json_object`), but the mapping holding them is this seal's own
    dict and ``to_wire`` copies only its top level. A snapshot that kept it would leave the run
    reporting an extension's output that the file it committed does not have.

    Called **after** the append, which is what makes it safe to run at all: those exact values
    have just been serialised, so nothing here can fail that the write did not already fail. The
    same call a step earlier would be a normalization that could suppress a row the run had
    otherwise earned."""
    return ResultRow.from_wire(json.loads(json.dumps(row.to_wire(), allow_nan=False)))


def _detached_row(row: ResultRow) -> ResultRow:
    """One recorded row, copied whole, for a reader that must not be able to reach the run's.

    :class:`ResultRow` is frozen and *shallow*: ``observed`` is a list of dicts, ``extensions``
    is a dict of them, and ``Score.feedback`` is a list of them too — so a reader handed the row
    itself can edit what the run reports without touching what the file says, and every later
    read would show the edit. A run's record is not a thing reading it may rewrite.

    ``deepcopy`` here, for the reason :func:`_detached_manifest` gives and not despite the one
    :func:`_detached_summary` gives: what is behind this is already plain data (see
    :func:`_recorded_row`), so copying it copies data and runs no code an env wrote. That is what
    makes the read total — a reader gets the row whatever the env published in it."""
    return copy.deepcopy(row)


def _detached_manifest(manifest: ToolManifest) -> ToolManifest:
    """One advertised tool on a copy of its own schema, for a reader that must not be able to
    reach the stream's.

    The frozen contract is only frozen if reading it cannot rewrite it, and a ``ToolManifest`` is
    a mutable model whose ``input_schema`` is a mutable dict — ``model_copy`` copies the model
    and keeps the dict. Shared, the same object is what every task's framing advertises and what
    a server registers, so a reader that edited it would change the contract the agent is shown
    from the next dispense on, and the endpoint would register the edit if it were built
    afterwards. The snapshot behind this is plain JSON (see the constructor), so copying it
    copies data and runs no env code."""
    return manifest.model_copy(update={"input_schema": copy.deepcopy(manifest.input_schema)})

def _detached_summary(score: Optional[Score]) -> Optional[Score]:
    """A private copy of a row's summary, for a caller that must not be able to reach the row's.

    Detachment has to reach the shallow layers, because those are the ones that alias: the row's
    ``observed`` list *is* the ``Score.feedback`` list, and each item in it is the dict the row
    carries. So it is round-tripped through the same strict JSON :func:`_strict_json_object`
    detaches *returned* provenance with — the same encoder and the same ``allow_nan`` setting
    :func:`_append_jsonl` commits the row with.

    **Not deep-copied, because a copy runs the copied object's own code.** ``deepcopy`` dispatches
    to ``__deepcopy__``/``__reduce_ex__``, and the values here are the env's: the feedback models
    are mutable and do not validate on assignment, so a value that is a *subclass* of a JSON
    scalar reaches ``observed`` and is written to the row like any other string or number. Its
    ``__deepcopy__`` raising would take down a row this same run records without provenance — the
    copy is built here, above the boundary that contains a failing extension, so nothing catches
    it, the seal fails, and ``_compose_row`` is retried, finalizing a second time every span that
    had already closed. One that blocks would wedge the seal where no extension bound applies.
    Merely turning provenance on would then suppress a row the stream would otherwise have
    scored, with every extension behaving perfectly.

    Serialising runs none of that code: ``json``'s encoder reads JSON scalars through their
    concrete C representation rather than through ``__repr__`` or ``__str__``, so a subclass gets
    no say. What it strips is exactly what the subclass was: scalar subclasses become the exact
    ``str``/``int``/``float``/``bool``, tuples become lists, non-string keys become strings.
    Nothing declared is lost — the wire type of a feedback value is ``float | bool | str`` — and
    nothing the *row* keeps is lost either, because the row is this same JSON. An extension now
    sees the summary in the form the file will hold it in, which is the form it should have been
    reasoning about.

    Nor can it refuse a summary the row can carry. ``hgym.feedback.wire`` validates every item on
    the way into ``observed`` and admits exactly the finite JSON scalars this call accepts, so
    there is no value that reaches here and fails here. The caller contains it anyway (see
    :meth:`TaskStream._finalize_spans`): that argument holds across a module boundary, and a row
    may not depend on one to survive.

    ``reward`` and ``success`` are handed over as they are, not copied. They are already exact
    primitives rather than anything the env passed in — :func:`_pick_float` returns ``float(v)``
    and ``bool`` cannot be subclassed — and they are immutable, so sharing them aliases nothing
    the row can be reached through."""
    if score is None:
        return None
    return Score(
        reward=score.reward,
        success=score.success,
        feedback=json.loads(json.dumps(score.feedback, allow_nan=False)),
    )


async def _close_episode(live: _Live) -> None:
    """The release itself, run as the entry's own claimed task (see
    :meth:`TaskStream._release`).

    Nothing here may raise. The claim *is* this task and a failed one stays claimed, so an
    exception would be handed to every later joiner — a seal publishing the stop it owes, a drain
    finishing what a failed seal left — for a teardown that is best-effort by nature and whose
    failure is not the run's outcome.

    ``CancelledError`` included, and it is the one that costs most. An env's ``close`` can raise
    it, this task is joined through a shield and nothing in this module cancels it, so one here
    is the env's. Out it would go through :meth:`_release`, into the middle of :meth:`_run_seal`
    — past the durable append, before the stop that seal still owes — so an unheadlinable summary
    or a failed terminal would go unpublished, the terminating call would answer the agent with a
    traceback instead of the constant, and ``aclose`` would report a clean run over it."""
    cancellation = _Cancellation()
    try:
        await live.episode.close()
    except BaseException as exc:  # noqa: BLE001 — the row is settled; teardown is best-effort
        if _must_propagate(exc, cancellation):
            raise


# Extension callbacks nobody is waiting on any more. Held only so the loop cannot collect a task
# that is still running — asyncio keeps no strong reference of its own — and dropped as each one
# ends. One that never ends stays here, which is the same fact as a callback that never ends.
_abandoned_calls: "set[asyncio.Task[Any]]" = set()


def _abandon(call: "asyncio.Task[Any]") -> None:
    """Stop waiting on an extension callback: cancel it, and let it go.

    Cancelled but deliberately **not awaited**, which is the whole difference between a bound and
    a wish. ``CancelledError`` is catchable, so a callback that catches it can run on or never
    return at all, and awaiting that is exactly the wedge the bound exists to prevent.

    What is left running holds only what it was handed: its own span object, the immutable
    :class:`TaskRef` at ``begin``, and at ``finalize`` that span's private re-parse of the
    summary (see :func:`_detached_summary`). It holds no lock, no registry entry, no episode and
    no file, and whatever it eventually returns is dropped on the floor — the row was written
    with this callback recorded as failed, and nothing reads it again. So an abandoned callback
    can spend CPU and its own resources; it cannot alter or suppress a row, or delay one.

    The done callback keeps a task nobody will ever await off asyncio's unretrieved list, the
    same bookkeeping :meth:`_seal` does for a seal whose caller left."""
    call.cancel()
    _abandoned_calls.add(call)
    call.add_done_callback(_abandoned_calls.discard)
    call.add_done_callback(_mark_retrieved)


def _mark_retrieved(task: "asyncio.Future[Any]") -> None:
    """Take a finished task's failure off asyncio's unretrieved list, without consuming it.

    A caller cancelled mid-seal leaves nobody awaiting the seal at the instant it fails, so
    asyncio logs ``Task exception was never retrieved`` for a failure that is not lost at all —
    the claim is still held, and the next drain joins the same task, reads the same exception and
    reports it. ``Future.exception()`` only clears the *warning*; the exception stays on the task
    for that later awaiter, so this changes what is logged and nothing else."""
    if not task.cancelled():
        task.exception()


def _stream_error(code: str, message: str) -> ToolResult:
    """A refusal by the stream, not a step in an env. It consumes no budget, enters no
    trajectory, and is never scored."""
    return ToolResult(
        content=json.dumps({"error": code, "message": message, "stream_error": True})
    )


def _leased_manifest(manifest: ToolManifest) -> ToolManifest:
    """The same tool, advertised with the required ``lease`` that names its episode.

    The env's schema is closed (``additionalProperties: false``), so the argument has to be part
    of the published schema — and the stream strips it again before the episode sees the call,
    or the routing capability would be recorded as part of the agent's action.

    **Only a schema this addition is provably sound for is wrapped.** Adding a name to root
    ``properties``/``required`` says what it means for a plain root object schema and for nothing
    else, while the *episode* keeps enforcing the env's own schema — so a shape this rewrite
    changes the meaning of leaves the endpoint advertising one contract and the seal enforcing
    another, which is an unearned result with an ordinary closure on it. A valid ``$ref``-rooted
    schema is the reachable case: the referenced object carries the native arguments and its own
    ``additionalProperties: false``, and a ``$ref`` sibling is either ignored — advertising a tool
    whose only permitted argument is the lease, so the one call a strict client can make seals as
    a clean wrong answer — or applied beside it, refusing the lease the root now requires and
    leaving the task impossible to finish.

    So the root is checked against an allow-list of keywords whose meaning the addition is known
    to preserve (see :data:`_WRAPPABLE_ROOT_KEYWORDS`), and anything else is refused *here* — at
    construction, before an env is opened or a position is spent — in the same shape
    :func:`_frozen_manifest` refuses a contract this endpoint cannot send. Transforming the whole
    schema instead would mean re-deriving an arbitrary author's semantics into a different
    document and hoping the two agree; a refusal is a statement the maintainer can act on, and
    the only thing it costs is capacity: at ``max_in_flight == 1`` nothing is wrapped and every
    one of these schemas is served exactly as the env wrote it.

    The refusal is loud about which keywords it could not carry, because that is what an env
    author has to change. It is not a claim that the schema is invalid.

    Read once and checked on the copy that is wrapped: the manifest reaching this point is
    already this stream's own plain data (see :func:`_frozen_manifest`), and checking the value
    that is then rewritten keeps it that way regardless of who else calls this."""
    schema = copy.deepcopy(manifest.input_schema)
    if not isinstance(schema, dict):  # pragma: no cover - manifests are objects
        raise ValueError(f"tool {manifest.name!r} has no object schema to wrap")
    unprovable = sorted(name for name in schema if name not in _WRAPPABLE_ROOT_KEYWORDS)
    if unprovable:
        raise ValueError(
            f"tool {manifest.name!r} has a schema the stream's {_LEASE_ARG!r} argument cannot be "
            f"wrapped around: its root carries {unprovable}, and adding a property beside "
            "any of those changes what the schema means rather than extending it. Serve this env "
            "at max_in_flight=1, where its schema is advertised exactly as written, or publish "
            "the tool's arguments as a plain root object schema"
        )
    if schema.get("type") != "object":
        raise ValueError(
            f"tool {manifest.name!r} has a schema the stream's {_LEASE_ARG!r} argument cannot be "
            f"wrapped around: its root is {schema.get('type')!r}, not an object, so it does not "
            "say that this tool takes named arguments a lease could be added to. Serve this env "
            "at max_in_flight=1, where its schema is advertised exactly as written, or publish "
            "the tool's arguments as a plain root object schema"
        )
    native_properties = schema.get("properties")
    native_required = schema.get("required")
    if not (native_properties is None or isinstance(native_properties, dict)) or not (
        native_required is None or isinstance(native_required, list)
    ):
        # Both are read *and rewritten* here, so a value of another type is not something a
        # lease can be added to: `[*"answer", _LEASE_ARG]` would publish a tool requiring six
        # single-letter arguments and a lease, and `dict("answer")` would raise from inside a
        # constructor that has envs open.
        raise ValueError(
            f"tool {manifest.name!r} has a schema the stream's {_LEASE_ARG!r} argument cannot be "
            "wrapped around: its root `properties` must be an object and its `required` an "
            "array, and the lease is added to both"
        )
    properties = dict(native_properties or {})
    if _LEASE_ARG in properties or _LEASE_ARG in (native_required or []):
        # Named in either place, because either is the env spending the word on an argument of
        # its own: one the stream would strip out of every call, and whose name it cannot
        # advertise twice.
        raise ValueError(
            f"tool {manifest.name!r} already takes an argument named {_LEASE_ARG!r}, which the "
            "stream needs to name the episode a call belongs to"
        )
    properties[_LEASE_ARG] = {
        "type": "string",
        "description": "The lease `get_task` returned for the task this call belongs to.",
    }
    schema["properties"] = properties
    schema["required"] = [*(native_required or []), _LEASE_ARG]
    return manifest.model_copy(update={"input_schema": schema})


def _strict_json_value(value: Any, enclosing: Tuple[int, ...]) -> Any:
    """One value on its way onto a row, **read exactly once** and rebuilt in plain containers,
    with every object name checked to be exact text.

    Only the containers are rebuilt. Every scalar is passed through untouched, so what decides
    whether a ``Path``, a datetime or a ``NaN`` is admissible is still the encoder the record is
    written with, and not a second opinion here that could drift from it.

    Reading once is the point of doing it here rather than checking and then encoding. A value
    only has to be a ``dict`` to arrive; the encoder asks a ``dict`` *subclass* for its
    ``items()``, so a check that walked one read and an encoder that walked another would put
    values on the row that nothing looked at. What comes back is what was inspected, in exact
    ``dict`` and ``list``, which is what the encoder then sees.

    ``enclosing`` is the identity of every container this one sits inside. A structure that
    contains itself is refused here, with the same finding the encoder would have reached, rather
    than recursing until the interpreter stops it."""
    if isinstance(value, dict):
        if id(value) in enclosing:
            raise ValueError("a JSON object may not contain itself")
        inside = enclosing + (id(value),)
        built: Dict[str, Any] = {}
        for name, item in value.items():
            if type(name) is not str:
                raise TypeError(
                    "a JSON object's names must be text, got "
                    f"{_described(lambda: f'{name!r} ({type(name).__name__})')}"
                )
            if name in built:
                # Only a mapping that answers for itself can get here — two equal names cannot
                # both be in a `dict` — and the encoder would write both, leaving the decoder to
                # keep whichever came last. Losing one silently is the failure this refuses.
                raise ValueError(f"a JSON object may not name {name!r} twice")
            built[name] = _strict_json_value(item, inside)
        return built
    if isinstance(value, (list, tuple)):
        if id(value) in enclosing:
            raise ValueError("a JSON array may not contain itself")
        inside = enclosing + (id(value),)
        return [_strict_json_value(item, inside) for item in value]
    return value


def _strict_json_object(value: Any) -> Dict[str, Any]:
    """A JSON object an extension handed back, proved to be strict JSON and detached from it.

    A type annotation does not stop a ``Path``, a datetime, a NaN or a mutable alias reaching
    the provenance file, so the value is actually serialised and re-parsed: what lands on the
    row is a detached copy that is provably writable.

    The encoder proves the *values* and is deliberately not trusted with the **names**. It
    coerces any key it is handed into text instead of refusing it — ``1``, ``True`` and ``1.0``
    are all written as names a JSON object can hold, and so is a ``str`` subclass that its own
    ``__eq__`` says is nothing like the plain key of the same text. Two keys a ``dict`` holds
    apart therefore become one name, and the value that survives is whichever was written last.
    Nothing raises anywhere: the extension is told its output was recorded, the row says as much,
    and one of the two values it handed over is simply not on it. "Strict JSON" has to mean the
    names too, so :func:`_strict_json_value` refuses everything but exact ``str``, the whole way
    down, and refuses a normalization that would put one name on an object twice."""
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
    named = _strict_json_value(value, ())
    round_tripped = json.loads(json.dumps(named, allow_nan=False))
    if not isinstance(round_tripped, dict):  # pragma: no cover - json.dumps of a dict
        raise TypeError("expected a JSON object")
    return round_tripped


# Scheduled catalog closes, held so the loop cannot collect one before it runs. A failure in
# one is deliberately left unretrieved: the loop's own unhandled-exception handler reports it,
# which is louder than anything this module could do with it after the constructor has raised.
_pending_closes: "set[asyncio.Task[None]]" = set()


def _close_on_owning_loop(env: Env) -> bool:
    """Close ``env`` from sync code **without moving it to a loop that does not own it**.
    Returns True when the close was scheduled rather than completed.

    ``Env.close`` is a coroutine and the contract says nothing about loop affinity, while the
    factory that built this env is explicitly allowed to provision resources — so an env built
    inside a running loop may hold objects belonging to *that* loop. Running its close on a
    private worker loop is therefore not a safe generalisation: at best it raises (a future
    attached to a different loop) and at worst it deadlocks, because the sync constructor
    waiting on the worker's result is blocking the very loop that close is waiting on.

    So the close runs where the env was built. With no loop running there is nothing to conflict
    with and this is a complete, synchronous close. Inside a running loop a *synchronous*
    constructor cannot await one, so the close is scheduled on that loop and completes as soon
    as the caller yields — after the error has propagated. That is the cost of validating in
    ``__init__``: the alternative is an async construction boundary, which is a different API.
    A caller that needs the close to be finished before it sees the error can construct outside
    a loop; a caller inside one is told, on the error itself, that the cleanup is still in
    flight."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(env.close())
        return False
    task = loop.create_task(env.close())
    _pending_closes.add(task)
    task.add_done_callback(_pending_closes.discard)
    return True


def _episode_items(feedback: Sequence[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    """Every **episode-level** item the env published under ``name``, in publication order.

    Episode level is this package's own definition of an outcome that belongs to the whole
    task: ``wire.select_inband`` withholds it until the terminal because it *is* the terminal
    score, while inference-level feedback is scoped to a single step and is recorded but not
    surfaced. A row is a per-task record, so headlining a step's value would leave ``reward``
    meaning "this task" on one row and "step 3" on the next, indistinguishably. Inference
    items stay in the row — they are evidence — but they are never the headline."""
    return [
        item
        for item in feedback
        if item.get("level") == "episode" and item.get("name") == name
    ]


def _pick_summary(
    feedback: Sequence[Dict[str, Any]],
    names: Sequence[str],
    accept: Callable[[Any], bool],
    expected: str,
) -> Any:
    """The first of ``names`` the env published at episode level, required to have been
    published exactly once and to already *be* the type it is summarized as. ``None`` when it
    published none of them.

    Never coerced. A feedback value is allowed to be a number, a bool or text
    (:data:`hgym.types.EpisodeFeedbackValue`, and ``hgym.feedback.wire`` validates exactly that
    set on both boundaries), so a wrong-typed one is a reachable env-authoring defect rather
    than an impossibility. Coercing it is the worst available answer: ``bool("false")`` is
    ``True`` and so is ``bool(0.25)``, so the env's own "not solved" becomes this record's
    "solved" — a benchmark overstated, and overstated directly beside the raw feedback that
    contradicts it. This package already settled that argument for the sibling case in
    ``wire.parse_meta``, which validates the terminate flag rather than coercing it, for this
    exact reason.

    Dropping it quietly is the other half of the same mistake, not the fix for it. ``None``
    already means "the env published no such field"; reusing it for "published something
    unusable" makes the two indistinguishable in the file, and a consumer counting truthy
    summaries would read the malformed run as a run of *failures*. That trades a silent
    overstatement for a silent understatement.

    So a wrong type raises and the caller stops the stream (see :meth:`TaskStream._seal`). The
    type an env gives a field is a property of the env, not of the task, so every later row
    would carry the same unusable headline; and stopping at the first one is what keeps
    ``None`` unambiguous everywhere else — in a run that finished, it always means the env
    published nothing under these names.

    **Published twice is the same problem without the type error.** The feedback wire is an
    ordered *list* and neither it nor ``FeedbackCollection`` enforces one item per name, so an
    env can publish ``success`` more than once. Nothing in this package says a later item wins:
    ``FeedbackCollection.get`` and ``EvalResult.value`` both answer with the *first* match, so
    picking by list order would make this record disagree with the two accessors it sits beside
    — and quietly, since whichever value it dropped would be gone from the row too. It is also
    exactly how a wrong-typed value escapes the paragraph above: ``success="false"`` followed by
    ``success=True`` reads as an honest solve. There is no answer that is both silent and right,
    so a summary name published more than once raises like a wrong-typed one, and the row keeps
    every occurrence.

    The duplicate scan covers every name in ``names``, not only the one that would be picked: a
    summary-named field is published at most once, whether or not a preferred name outranks it.
    """
    published: Dict[str, Any] = {}
    for name in names:
        items = _episode_items(feedback, name)
        if len(items) > 1:
            raise _MalformedSummary(
                f"{name!r} was published {len(items)} times "
                f"({_described(lambda: ', '.join(repr(item['value']) for item in items))}), so "
                "this record cannot say which one is the outcome"
            )
        if items:
            published[name] = items[0]["value"]
    for name in names:
        if name not in published:
            continue
        value = published[name]
        if not accept(value):
            raise _MalformedSummary(
                f"{name!r} must be {expected}, got "
                f"{_described(lambda: f'{value!r} ({type(value).__name__})')}"
            )
        return value
    return None


def _is_number(value: Any) -> bool:
    # `bool` is an `int` subclass, so an unguarded numeric test would accept `True` and record a
    # reward of 1.0 the env never published.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pick_float(feedback: Sequence[Dict[str, Any]], names: Sequence[str]) -> Optional[float]:
    value = _pick_summary(feedback, names, _is_number, "a number")
    return float(value) if isinstance(value, (int, float)) else None


def _pick_bool(feedback: Sequence[Dict[str, Any]], names: Sequence[str]) -> Optional[bool]:
    value = _pick_summary(feedback, names, lambda v: isinstance(v, bool), "true or false")
    return value if isinstance(value, bool) else None


def build_stream_server(stream: TaskStream, *, name: Optional[str] = None) -> FastMCP:
    """Wrap ``stream`` in a FastMCP server: ``get_task``/``queue_info`` plus the env's native
    tools, routed to whichever episode is live. Constructs the server; it neither runs it nor
    owns its transport."""
    # An exception's text is written by the env, not by the stream: an env that fails to load a
    # task routinely names it ("no such task 42 in split heldout"), and MCP relays a tool
    # exception to the caller verbatim. Mask it — the caller learns only that the call failed,
    # while the full exception is logged server-side and still raised on the direct API. Argument
    # validation is unaffected: the episode returns those as ordinary results, so a caller keeps
    # the feedback it needs to correct its own call.
    server: FastMCP = FastMCP(name=name or "hgym:tasks", mask_error_details=True)

    @server.tool(name=_GET_TASK_TOOL)
    async def get_task() -> Dict[str, Any]:
        """Take the next task off the queue and start it.

        Returns the task framing — ``{env, instructions, budget, tools}`` — and never the task
        index or the target. Returns ``{"done": true}`` once the queue is empty; any task
        already dispensed to you is still yours to finish, and ``in_flight`` says how many of
        them there are. Work the task with the native tools it lists; they route to it
        automatically."""
        try:
            dispensed = await stream.get_task()
        except Exception:  # noqa: BLE001 — a stopped stream is the harness's business
            # A stopped stream serves no further task, and *why* is not the agent's to read: an
            # integrity failure would otherwise be an error here where an ordinary run returns a
            # task — the same shape channel a terminating call closes, one call later, and
            # reachable by the same conditional env. So it reads as the end of the queue, which
            # is what it is for the caller. Anything that did not stop the stream still raises:
            # that refused *this* dispense, not the run, and the agent meets it again next call.
            if not stream.stopped:
                raise
            dispensed = None
        if dispensed is None:
            # `done` is a statement about the QUEUE — no further task is coming — and about
            # nothing else. `remaining` is composed rather than relayed, because a stopped
            # stream gets this same answer and its queue is *not* empty: passing that count
            # through would say `done: true` beside a non-zero `remaining`, the two halves of
            # one response contradicting each other and the integrity failure this redaction
            # keeps off the call written out in a field. Both cases promise the identical thing,
            # which is what makes them indistinguishable here.
            #
            # `in_flight` is reported as it stands, and is not part of that promise. It counts
            # the caller's OWN open episodes — tasks it was handed and has not ended — so
            # claiming zero beside a live lease is not a redaction but a false statement about
            # the caller's own work: a worker that believes it stops, and the task it was still
            # entitled to answer lands as a forced loss at the drain. It reveals nothing a stop
            # would: a stop seals nothing by itself, and `queue_info` already answers this same
            # count on demand. `consumed` moves for the same reason — a count of the tasks the
            # caller itself played is the one residue no answer here could hide from it.
            info = stream.queue_info()
            return {
                "done": True,
                "remaining": 0,
                "consumed": info.consumed,
                "in_flight": info.in_flight,
            }
        return dispensed.to_wire()

    @server.tool(name=_QUEUE_INFO_TOOL)
    async def queue_info() -> Dict[str, Any]:
        """Report ``{remaining, consumed, in_flight}`` for the task queue."""
        return stream.queue_info().to_wire()

    reserved = {_GET_TASK_TOOL, _QUEUE_INFO_TOOL}
    for manifest in stream.tools:
        if manifest.name in reserved:
            raise ValueError(
                f"env tool name {manifest.name!r} collides with the stream's reserved control "
                f"tool; an env served by a stream may not expose a tool named {manifest.name!r}"
            )
        server.add_tool(build_tool(manifest, stream.dispatch))
    return server
