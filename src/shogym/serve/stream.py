"""Serve a *queue* of env tasks over one MCP endpoint.

:class:`~shogym.serve.episode.ServedEpisode` serves one episode. :class:`TaskStream` is the
next abstraction up: it holds a materialised queue of ``(env, task_idx)`` refs, hands out one
task at a time, routes the env's native tool calls to the episode that task is running in,
**seals and scores that episode itself** (never the agent), and appends one provenance row per
dispensed task.

The stream owns scoring, so what it hands out is deliberately *redacted*: a
:class:`DispensedTask` carries ``{env, instructions, budget, tools}`` — plus, when the stream
serves several envs, the ``tool_naming`` note that says what those tools are called — and has no
field that could hold the task index or the target.

The same rule holds for the *whole* agent-visible surface, not just the framing. A queue may
repeat a task index, so anything that identifies a task is a correlation channel: learn an index
once and a later occurrence can be recognised and replayed against a scorer. So the
:class:`ResultRow` a seal produces — its lease, queue position, index and the env's raw
feedback — stays in ``results.jsonl`` and :attr:`TaskStream.results`, and a terminating call
tells the caller only that the task ended. For the same reason the server masks tool exceptions:
an env that raises while loading a task can name it in the exception text, and MCP would
otherwise relay that verbatim.

That is the **default** and it is a policy, not a law of the object: ``feedback=Never()``. The
whole of the next section is what that default buys and why an evaluation may not give it up;
:class:`Immediate` is the other policy this module ships, and :class:`EvalStream` is the
construction that takes the choice away.

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
episode-level feedback on the terminal result (:func:`~shogym.feedback.wire.select_inband`) and
returns the env's own terminal payload with it. That rule draws its line at the end of the
interaction: once the episode is over, what the agent learns there cannot reach its own
behaviour. A stream's terminal is *mid-run* — the next ``get_task`` can hand back an index
already played — so the identical principle, applied at the boundary this object owns, redacts
instead of surfacing. Both channels are closed together: neither the env's terminal response nor
the feedback sidecar it rides on crosses a terminating call, because ``correct`` is equally a
verdict in either. Whether the env's terminal response happens to *be* a verdict is not
something the serving layer can tell, so the boundary is the call, not the payload.

**The verdict channel is a policy, and the record says which one was in force.** Everything
above describes ``feedback=Never()``, the default, and it is the posture an evaluation needs.
It is not the posture an agent *improving* needs: a run whose whole point is that the agent gets
better between tasks has to tell it how each one went, and a stream that cannot do that is a
stream every training loop has to be written around. So the choice is named at the construction
site rather than buried:

- :class:`Never` — the default. A terminating call answers with the fixed payload above, the
  same bytes for every env, task and outcome.
- :class:`Immediate` — the terminating call carries the **sealed row's own** episode-level
  feedback, verbatim: the very items ``results.jsonl`` records under ``observed`` at
  ``level == "episode"``, and nothing else. Episode level, because that is the same line
  :func:`~shogym.feedback.wire.select_inband` already draws for a single served episode — a
  stream's terminal reveals no more than that terminal would. Routed by the seal, so at capacity
  above 1 what comes back is the feedback of the task the *call* ended and never a sibling's;
  and only on that call, because a task the stream ended (a drain, the deadline, a displacing
  pull) has no response to carry it.

The envelope is this module's and only the item list is the policy's, which is what keeps a
policy from becoming a channel of its own: it is handed the sealed row's episode-level items and
its answer is put inside the response shape above, under one added ``feedback`` member. It never
sees, and can never add, the lease, the position, the index, the closure, the queue counts or the
stop — so the answer to "what else can an agent learn from this?" is a property of these few
lines rather than of whatever policy is passed. That shape is also what makes ``Delayed(k)``,
``Batched(n)`` and ``Noisy(p)`` arrive later as policies rather than as new surface — written
here and added to the allow-list of exact policy types a stream will serve under, because a
regime is the record's claim about how a row was produced and a claim an arbitrary subclass makes
about itself is not one a reader can check.

Which policy served a task is written into the record itself — ``feedback_regime`` on every
dispense record and every result row — because "these scores were earned with no verdict
channel open" is a claim about how a row was produced, and a row that cannot make it is a row a
reader has to take on trust from prose. It is stamped on the *dispense*, before the task is
handed out and so before anything could have been revealed, and again on the row; a record with
no stamp was written before this existed, when every stream was ``Never``, so the reader idiom is
``row.get("feedback_regime", "never")``.

One record never mixes regimes, and that takes three checks rather than one, because the first
two are about the past and a record's second writer may not have one yet. A fresh stream refuses
a directory that already holds records; a resuming one refuses a directory whose records name a
different regime; and every stream, fresh or resuming, **claims the directory** with an
exclusive create before it builds anything, so that two streams pointed at the same empty
directory end with one serving and one refusing rather than with both appending. The claim names
the regime too — a run killed before its first dispense leaves no record to compare against, and
the claim is what a resume compares itself with there. It is released on an orderly close, so a
claim left on disk means a stream that never finished, which is exactly what ``resume=True``
asserts about it.

**The claim is re-checked inside every append, holding an exclusive lock on the directory across
the check and the write.** Ownership taken once covers a constructor; ownership re-read before a
write covers everything except the write itself, which is where the second writer lands. So the
two logs are appended to by one function, and it verifies inside the same critical section it
writes in — against the token that says *which stream*, the pid that says *which process* and a
witness that says *which object*, so that a stream inherited by ``fork`` fails the check its
parent passes rather than filing rows under numbers its parent is also using, and so does one
duplicated inside a single process. **A stream cannot be copied at all**: ``copy``, ``deepcopy``
and ``pickle`` are refused where the second object would be made, because an ownership identity
is not a value and a copy of one is two streams serving one record.

:class:`EvalStream` is that default made structural — it pins :class:`Never`, refuses a
``feedback`` argument outright, and enumerates what it does and does not guarantee.

Ownership: ``env_for`` is a **factory**, not a shared instance. Each episode gets its own env
and closes it, because ``ServedEpisode.close()`` closes its env and ``Env.close()``
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
occurrence — which is the form the JSONL trace and :class:`~shogym.evaluate.EvalResult` already
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

**Several envs at once.** Native tool names collide across envs — every env has
``terminate``, and ``done`` and ``submit_answer`` each appear in more than one env with
different schemas behind the same name — while a server registers one schema per name. So with
more than one env in the queue the stream advertises ``<env>__<tool>``, and routing uses an
explicit map rather than splitting the public string back apart. That map is also the collision
check: it refuses to hold one public name twice, whichever pair of halves produced it. A
single-env stream advertises the env's own names, unchanged.

That join is where an env key stops being an internal one. Until there are two envs it is a
caller's private label and may be any string; joined, it is *part of a tool name*, and a tool
name is protocol-bounded — 1 to 128 characters of letters, digits, ``_``, ``-`` and ``.``. So a
queue naming several envs is checked against that bound before it is served, on the **joined**
name rather than either half, because neither half decides it: a key with a space makes an
invalid name out of a valid tool, and two names each well under the length limit can join to
one that is over it. The check is a refusal at construction because the layer that would
otherwise catch it does not — FastMCP warns about a name outside the set and registers it
anyway, leaving the endpoint to be rejected by a strict client or a downstream provider, where
no harness can see it. A single-env stream is not checked at all, in either half: nothing is
joined there, so there is no name the stream made, and an env's own tool names are that env's
contract with its server rather than this join's business.

Renaming a tool moves the *framing* out of step with it, and that is the agent's problem rather
than the endpoint's: an env's ``instructions`` are the env author's prose and routinely name the
env's own tools ("call ``submit`` with your answer"), while the endpoint now registers
``answers__submit``. Those instructions ship **verbatim** — the stream does not edit an env's
prose, because deciding which words in free text are tool names is a guess, and a wrong guess
corrupts the task itself. Instead the framing says the mapping alongside them, in the stream's
own words (``tool_naming``), naming only tools the endpoint really registers.

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
import contextlib
import copy
import fcntl
import json
import math
import os
import re
import secrets
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    Mapping,
    List,
    Literal,
    NamedTuple,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Set,
    SupportsIndex,
    Tuple,
)

from fastmcp import FastMCP
from fastmcp.tools import ToolResult

from shogym.core import Env
from shogym.feedback.wire import (
    CHANNEL_FEEDBACK_NAME,
    NOTICE_FEEDBACK_NAME,
    REPORT_FEEDBACK_NAME,
)
from shogym.serve.episode import ServedEpisode
from shogym.serve.server import build_tool
from shogym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from shogym.task import TaskSpec, ToolManifest

__all__ = [
    "Closure",
    "CompletedTask",
    "DispensedTask",
    "EvalStream",
    "FeedbackPolicy",
    "Immediate",
    "Information",
    "Never",
    "Placebo",
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

# The envelope a terminating call is answered in, whatever the feedback policy. A policy adds at
# most the one member below to it and may change nothing else, so the shape of a terminal answer
# is decided here rather than by whatever object a caller passed. Read-only, because every
# revealing answer is composed from it at the moment it is sent: a plain dict would leave the one
# invariant part of that answer editable by anything that can reach this module.
_TASK_OVER_FIELDS: Mapping[str, Any] = MappingProxyType(
    {
        "content": "<task ended; the stream recorded the outcome>",
        "terminated": True,
        "hint": f"task over; call `{_GET_TASK_TOOL}` for the next one",
    }
)

# The *entire* response to a terminating call under the default `Never` policy — the same bytes
# for every env, every task and every outcome. Serialized once, at import, so that invariance is
# structural rather than a convention the next edit could quietly break: nothing about the sealed
# episode can be read off a constant, including from which keys are present.
_TASK_OVER = json.dumps({**_TASK_OVER_FIELDS})

# The member a revealing policy's answer rides in, and the *only* one it may add.
_FEEDBACK_MEMBER = "feedback"

# The whole response under a revealing policy that revealed nothing — an env that published no
# episode-level feedback, a policy holding this task's back, a seal that recorded no row, a policy
# that could not answer at all, or a call that reached the episode after it was already over and
# so ended nothing (see `_tombstone_answer`). Serialized at import for the reason `_TASK_OVER` is:
# under a revealing policy this member is always present, so an empty list and a missing key are
# never the same answer, and every reason for an empty one is the same bytes. A stream whose record
# already lost a row may not be the one stream whose terminal answers in a different shape.
_TASK_OVER_SILENT = json.dumps({**_TASK_OVER_FIELDS, _FEEDBACK_MEMBER: []})

# What a row records about the policy that produced it (`ResultRow.feedback_regime`, and the same
# member on a dispense record). A record written before this existed carries no such member and
# was written by a stream that revealed nothing, so it reads back as `never` — the reader idiom is
# `row.get("feedback_regime", "never")`.
_NEVER_REGIME = "never"
_IMMEDIATE_REGIME = "immediate"
_INFORMATION_REGIME = "information"
_PLACEBO_REGIME = "placebo"

# The feedback level a terminal may reveal. `wire.select_inband` already draws this line for a
# single served episode — episode-level items ride out on the terminal result, inference-level
# ones are recorded but not surfaced — and a stream's terminal reveals no more than that one
# would. Applied by the stream, above the policy, so no policy can widen it.
_EPISODE_LEVEL = "episode"

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

# Joins an env key to a native tool name when a stream serves more than one env. Nothing ever
# recovers the halves by splitting the joined string, and nothing needs to: routing goes through
# an explicit map, and it is that map — which refuses to hold one public name twice — that makes
# a public name unambiguous. Banning this in either half is a legibility rule sitting on top of
# that guarantee, not the guarantee itself: two admissible pairs really can join to one string
# (`("a", "_x")` and `("a_", "x")` both give `a___x`), and the map is what catches it.
#
# The ban, and every other rule about the shape of a name here, applies ONLY when a join
# actually happens. With one env in the queue nothing is joined, the env's own names go on the
# wire exactly as they always did, and none of this is any of the stream's business.
_SEPARATOR = "__"

# What a tool name may be, on the wire: one to 128 characters drawn from ASCII letters, digits,
# `_`, `-` and `.`
# (https://modelcontextprotocol.io/specification/2025-11-25/server/tools#tool-names). Restated
# here rather than imported from the MCP package's internals, because it is what *this* module
# promises about the names it manufactures and that promise may not change under it on a
# dependency bump; a test pins the two against each other. See `_unregistrable`.
_TOOL_NAME_CHAR = re.compile(r"[A-Za-z0-9._-]")
_TOOL_NAME_MAX = 128

_RESULTS_FILE = "results.jsonl"
_DISPENSES_FILE = "dispenses.jsonl"
# Which stream owns this provenance directory, in which process, and under which regime. Not a
# record — nothing reads it to score a run — so it is neither appended to nor fsynced: it exists
# for exactly as long as a stream is serving into the directory, and a host crash that loses it
# loses a claim whose owner the crash already killed. See `TaskStream._claim_provenance`, and
# `_locked` for the exclusion that holds it still across an append.
_CLAIM_FILE = "claim.json"

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


def _require_task_ref(ref: Any) -> TaskRef:
    """One queue entry, checked to be the identity it will be *recorded* as.

    A ``NamedTuple`` annotation is documentation, not validation, so anything at all reaches the
    queue — and these two fields are identity, not payload. They name the env whose rows are
    filed under them, they are written into every dispense and every result, and a resumed run
    compares its queue against them. Nothing downstream re-checks them, but several things
    *coerce* them: ``ServedEpisode.open_env`` takes ``int(task)`` to load a task, while the row
    is appended carrying the caller's own value, and :meth:`ResultRow.from_wire` then coerces
    again when the run re-reads what it just committed. A ``1.9`` therefore plays task 1, is
    recorded as ``1.9`` in ``results.jsonl``, and comes back as ``1`` in memory — three readings
    of one task, and a resume that refuses the run's own record as a queue mismatch.

    Coercing here would only move the disagreement earlier: the caller asked for something this
    queue cannot hold, and a canonical identity invented after the fact is what put a number in
    the file that nothing else agrees with. So exact ``str`` and exact ``int`` — subclasses
    included, because a subclass is a value with its own ``__eq__`` sitting in a field every
    later comparison runs on, and the wire form is a plain scalar either way."""
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
            f"({type(task_idx).__name__}); it is written to every dispense and every row, and "
            "what the queue plays has to be what the record says it played"
        )
    return TaskRef(env, task_idx)


def _claim_detail(held: Mapping[str, Any]) -> str:
    """What a refusal can say about the claim it is refusing over, for the human deciding whether
    to pass ``resume=True``.

    Every member is optional and none is trusted: the file may have been written by a version
    that recorded different ones, or may not have parsed at all (see
    :meth:`TaskStream._read_claim`), and a message is the one place where saying less is better
    than guessing. Rendered with ``!r`` so a value that is not what it claims to be looks like
    what it is rather than blending into the sentence."""
    detail = [
        f"{label} {held[member]!r}"
        for label, member in (
            ("under feedback regime", "feedback_regime"),
            ("by pid", "pid"),
            ("at", "claimed_at"),
        )
        if member in held
    ]
    return f" ({', '.join(detail)})" if detail else ""


def _recorded_regime(record: Mapping[str, Any]) -> str:
    """The feedback regime a stored dispense record or result row says it was written under.

    A record carrying no such member predates the policy, and every stream that could have
    written one revealed nothing — so it reads back as :class:`Never`'s regime rather than as
    unknown, which is what makes ``row.get("feedback_regime", "never")`` the whole reader idiom.

    Coerced with ``str`` for the reason every other field :meth:`ResultRow.from_wire` reads is:
    this value is compared against the serving stream's own regime when a run resumes, and a
    stored value that is not text would otherwise decide that comparison by its own ``__eq__``.
    Coercion cannot launder a wrong value into a right one here — no non-string renders as
    ``"never"`` — so a record this module did not write fails the comparison rather than
    passing it."""
    return str(record.get("feedback_regime", _NEVER_REGIME))


@dataclass(frozen=True)
class DispensedTask:
    """What the agent receives: enough to act, and nothing that identifies the task.

    Redaction is structural — there is no field the task index, the target, or the queue
    position could be written into.

    **What this withholds is the queue's identification of the task, and only that.** The index,
    the position and the target are the *stream's* facts about where a task sits in a run, and
    knowing them would let an agent read its own progress and its neighbours' off its own
    envelope. An env's own material is a different thing and is not redacted here: instructions
    are published verbatim above, and an env is free to name the task it is serving inside them
    or inside a value it publishes, because that name is what the agent is working on rather than
    a fact about the run. An env that composes a terminal payload out of the task's identity and
    the agent's own submission is therefore consistent with this class, not in tension with it."""

    env: str
    # The env's own prose, exactly as the env published it. The stream never edits it — see
    # `tool_naming` for what it says instead when the two would disagree.
    instructions: str
    budget: Optional[int]
    tools: Tuple[Dict[str, Any], ...]
    # The capability that names this episode, present only when more than one may be live. It
    # identifies the episode, never the task: it is random, and a new one is minted per dispense.
    lease: Optional[str] = None
    # What this task's tools are called on this endpoint, said in the STREAM's words and only
    # when the stream renamed them (see `_naming_note`). Env-static and derived from the frozen
    # manifest, so it stays inside the redaction: there is nothing task-specific it could hold.
    tool_naming: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        wire: Dict[str, Any] = {"env": self.env, "instructions": self.instructions}
        # Next to the instructions, because that is the prose it is about.
        if self.tool_naming is not None:
            wire["tool_naming"] = self.tool_naming
        wire["budget"] = self.budget
        # Deep, because a tool carries a schema: `dict(tool)` would copy the entry and leave
        # every payload built from this task sharing one schema object with the task itself.
        wire["tools"] = [copy.deepcopy(tool) for tool in self.tools]
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
    ``{dispensed, error}``        no ``finalize`` result reached this row, and ``error`` says
                                  which of the three reasons it was: ``finalize`` failed, the
                                  seal failed while it was running, or the seal failed before
                                  this span was reached and it was never called (see
                                  :meth:`TaskStream._unclosed_spans`)
    ``{dispensed}``               nothing was recorded after the dispense — the row came from
                                  :func:`reconcile`, so ``closure`` is ``broker_abort``
    ============================  ==========================================================

    ``dispensed`` is always present, because a span that would not open refuses the dispense
    outright. So a row written in-process always has exactly one of ``sealed`` or ``error``
    beside it, and only a reconciled row has neither — but the discriminator a consumer should
    read is ``closure``, which is typed and says the same thing about the whole row.

    ``feedback_regime`` names the :class:`FeedbackPolicy` this task was **assigned** to —
    ``"never"`` for a run with no verdict channel open, and so the one thing a reader needs to
    tell an evaluation-grade row from a practice one **without joining against anything**. It is
    the row's own answer to a question the rest of the row cannot settle: a score is the same
    number either way, and only the regime says which arm the task was served under. A row
    written before this member existed came from a stream that revealed nothing, so absent reads
    as ``"never"`` and the idiom is ``row.get("feedback_regime", "never")``.

    **It is the assignment and never the exposure, and the difference is the point.** This row is
    appended and fsynced *before* the policy's answer is composed, which it has to be: the answer
    is composed from the recorded row, and a value handed to an agent before it was durable would
    be a verdict the record might not hold. So the regime here says which channel this task was
    served under — the treatment assigned — and says nothing whatever about whether a value
    reached the caller. A cancelled terminal, a task the stream ended itself (``drained``,
    ``timeout``) and a policy that could not answer all leave a scored row stamped
    ``information`` or ``placebo`` with nobody told.

    That is the field an intention-to-treat analysis wants and the one it should use: every
    assigned task carries one, including the tasks whose delivery failed, and no post-treatment
    filter sits between the assignment and the estimate. What was actually delivered is not this
    module's to say, and a design that needs it records it in the runner."""

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
    # The channel this task was ASSIGNED, never the one it was told through (see the class
    # docstring). Defaulted, and defaulted to the regime that has no channel: every row this
    # module wrote before the policy existed was written by a stream that revealed nothing, and a
    # row built without saying otherwise — `reconcile`'s, a caller's — may not read as one that
    # did.
    feedback_regime: str = _NEVER_REGIME

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
            # Appended, never inserted: every member above is one an existing reader already
            # keys on, and this one is additive so a `Never` row stays a row those readers parse.
            "feedback_regime": self.feedback_regime,
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
            feedback_regime=_recorded_regime(row),
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
    # What the core said failed, when the terminal transaction failed closed: the failure's type,
    # how many errors it reported and which kinds, never its message and never a field location.
    # Read off the episode on its own channel because it is a harness-side fact rather than part
    # of the payload a terminal call answers the agent with, and it is what turns an unscored row
    # from a report that something went wrong into one that says what.
    terminal_failure: Optional[Dict[str, Any]] = None
    # Set when a terminal the *stream* drove failed after ending the task, could not end it at
    # all, or ended it and left a verdict this record could not read off the episode. An
    # exception raised at a boundary the harness drove, never a value the env published — so the
    # classifier may read it without reopening the channel `_classify` closes. The row lands
    # unscored with a diagnostic, and the stop follows the release, like `summary_error` and for
    # the same reason. A `call_error` promoted into this field is the one that does not stop: it
    # is the same finding about the row and a different one about the run (see `_run_seal`).
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
    # :meth:`dispatch`). It becomes this entry's `terminal_error` only if the *stream* ends up
    # forcing the terminal, which is the case where a lost call would otherwise be recorded as a
    # task the agent played out and got wrong. That promotion buys the row and not the stop, and
    # the source recorded beside it is what tells the two apart.
    call_error: Optional[BaseException] = None
    # Set when the env's headline could not be read off this task's feedback. The row still
    # lands (unscored, with a diagnostic); the stop is raised *after* the release, because a
    # row that landed makes the seal final and this entry is the only handle on its episode.
    summary_error: Optional[_MalformedSummary] = None
    # Set when this task's seal failed in its composing half, above the durable append. The row
    # the retry then writes is one this seal composed or stood in rather than one the storage
    # refused, so the stop such a failure owes is not the append's (see
    # `TaskStream._join_seal`). Kept as state for the reason the source beside `terminal_error`
    # is: the exception alone cannot say which half of the seal it came out of.
    compose_error: Optional[BaseException] = None
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
    #
    # **A hand-back always has one.** A seal that failed short of the append cannot leave this
    # empty for the same reason: a retry that found nothing here would compose that second row. So
    # a failure in the composing half leaves the row it had reached: the real one when the outcome
    # was already classified, and an unscored `finalize_error` row saying the seal produced none
    # when it was not (see `TaskStream._retained_row`). The write a retry retries is always a
    # write and never a composition.
    pending_row: Optional[ResultRow] = None
    # namespace -> the open span, and what it observed at dispense.
    spans: Dict[str, ProvenanceSpan] = field(default_factory=dict)
    dispensed_extensions: Dict[str, Any] = field(default_factory=dict)
    # namespace -> what closing that span recorded, filled in one span at a time as they close
    # (see `TaskStream._finalize_spans`). Kept on the entry rather than in that call's own dict
    # because a failure it must let out takes a local one with it: the row the seal still owes
    # would then have nothing to say about the spans that had already closed, and would answer
    # for their namespaces with a failure they never raised (see
    # `TaskStream._unclosed_spans`).
    finalized_extensions: Dict[str, Any] = field(default_factory=dict)

    def failed_to_end(self, exc: BaseException, source: _TerminalErrorSource) -> None:
        """Remember that this task has no verdict behind it, and where that came from.

        The first failure explains the run, so a later one never overwrites it. The source is
        written *here*, beside the exception, because the two must never be able to disagree:
        a caller that set one without the other would leave the classifier and the stop reading
        a source that belongs to some earlier failure."""
        if self.terminal_error is None:
            self.terminal_error = exc
            self.terminal_error_source = source

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


def _frozen_manifest(env_name: str, tools: Sequence[ToolManifest]) -> List[ToolManifest]:
    """An env's own manifest as the wire carries it — the canonical copy everything this stream
    does with that contract is derived from.

    Frozen as soon as the contract has been validated, and before anything is *derived* from it.
    ``model_copy`` is shallow and ``describe`` hands back the env's own objects, so without this
    the signature the drift check compares, the score terminal the stream drives, the map a call
    is routed through and the note a renamed task carries would all be the env's values, each one
    running the env's code at a moment nothing is guarding.

    That is not a leak of decoration: those are the load-bearing paths. A tool name is looked up
    to find the episode a call belongs to and is passed into the episode as the tool to call, so
    a name whose own ``__hash__`` or ``__eq__`` answers differently once the run is under way
    loses the agent's call — on a tool the endpoint advertised in plain text, that the agent
    called exactly as advertised. Nothing downstream can see that a name was ever involved: the
    call is simply lost, and a task the agent then ends itself keeps whatever score the missing
    step left behind.

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


def _described_failure(failure: Optional[Dict[str, Any]]) -> str:
    """The core's structural summary of a failed terminal transaction, as a diagnostic suffix.

    A row that only says a transaction failed closed leaves its reader with nowhere to start:
    every cause reads identically, so the first move is always to go and reproduce the failure
    that the harness had already caught. This appends what the core kept of it (the failure's
    type, how many errors it reported, and which kinds), which is enough to tell an env defect
    from an evaluator timeout without rerunning anything.

    Empty when there is nothing to add, so the sentence it extends stands alone, unchanged, for a
    failure that could not describe itself or for a core that recorded none. Nothing here formats
    an exception or an env value: the summary is built once, at the point the failure was caught,
    out of a fixed vocabulary and a count (see ``failure_summary``), so by the time it reaches
    this module it carries nothing that came from the data the env was holding.
    """
    if not isinstance(failure, dict):
        return ""
    name = failure.get("error")
    if not isinstance(name, str) or not name:
        return ""
    parts: List[str] = []
    count = failure.get("error_count")
    if isinstance(count, int) and count > 0:
        parts.append(f"{count} error{'' if count == 1 else 's'}")
    kinds = failure.get("error_kinds")
    if isinstance(kinds, list) and kinds:
        parts.append(", ".join(str(kind) for kind in kinds))
    if not parts:
        return f" ({name})"
    return f" ({name}: {'; '.join(parts)})"


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


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """Hold every other process out of ``directory`` for the length of the ``with`` body.

    This is the *exclusion* half of provenance ownership, and it is a different mechanism from the
    claim for a different job. The claim file says **who** owns a directory: it is durable, it is
    readable by the human deciding what to do about it, and only ``resume=True`` breaks it. This
    says **nobody else is between my check and my write**: it lives for the milliseconds of one
    critical section and leaves nothing behind. Ownership without it is check-then-write, which is
    not exclusion at all — two writers can each read a claim naming themselves and then append.

    **Why an ``flock`` and not a lock file, an ``O_EXCL`` create or a ``mkdir``.** Those three are
    the same primitive: a lock made out of a file *existing*. A process killed while holding one
    leaves it existing, so every later writer is blocked forever unless something breaks it — and
    the only rule that could break it is "the holder has probably died by now", which is the
    liveness oracle :meth:`TaskStream._claim_provenance` refuses to invent and would be inventing
    here at a *thousandth* of the timescale. An ``flock`` is owned by the kernel rather than by
    the filesystem: it is released when the descriptor closes, and every descriptor closes when
    its process ends, however it ends. So the residue of a crash mid-append is exactly nothing —
    there is no stale lock to clear, no flag to reset, and ``resume=True`` has nothing to do here
    (what it takes over is the *claim*, which is durable precisely because a crash must leave that
    behind).

    **Why the provenance directory itself is what is locked.** A lock on a dedicated file is a
    lock on that file's *inode*, and the file can be removed — by a cleanup script, by a caller
    tidying up, by a stream unmaking a directory it created. Once it is, one process holds the old
    inode while the next creates and locks a new one, and both believe they are exclusive: a lock
    that silently stops locking, which is worse than none. The directory cannot go that way while
    a claim is in it, because ``rmdir`` refuses a directory that holds a file — so the inode this
    exclusion rests on is kept alive by the very claim it protects, rather than by a convention
    about who may delete what.

    **Blocking, not a try-lock.** A try-lock would have to decide what a failure means, and the
    only honest reading of "someone else holds it" is "wait" — the holder is inside a synchronous
    append of a few kilobytes. A caller that finds itself waiting is either taking a directory
    over or serving a directory it should not be serving, and the second is refused a moment later
    by the claim check inside the lock. Waiting is bounded by an ``fsync``, not by anything a
    third party can extend: no env code, no extension callback and no policy runs in here.

    The body may not await, and none does: every caller is synchronous from the ``with`` to its
    close. An ``await`` in here would hold a cross-process lock across a suspension, which is how
    a lock that is only ever held for milliseconds becomes one held for as long as some other
    coroutine likes.

    An ``OSError`` from the lock itself propagates. It means the filesystem cannot provide the
    exclusion at all (some network mounts), and this is taken first at construction, so a run
    refuses before it serves rather than discovering it mid-record."""
    # O_RDONLY, because the lock is on the descriptor and not on what can be done with it: this
    # never reads or writes the directory, and asking for write access to take a lock would refuse
    # the exclusion on a read-only mount that can still perfectly well provide it.
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        # The close *is* the release, and it happens however the body left — an exception on the
        # append, a raise from the ownership check, or a return. An explicit unlock before it
        # would be the same syscall twice, with the second one still deciding the outcome.
        os.close(descriptor)


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
    appending to it — which is a fact about the caller, and the one
    :meth:`TaskStream._append_owned` establishes by holding the directory across the whole
    append. A second appender would make this truncation delete a record it never wrote. A crash
    between the truncation and its own durability simply leaves the record present and readable,
    which is the state the retry would have produced anyway.

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
    that. The provenance directory a harness made a moment earlier, a path some unrelated
    program created last week, a writer that made the chain and died before its own sync: each
    one leaves this call finding everything present and returning success over a store whose
    directory entry is still only in the page cache. A crash then takes the whole record, rows
    and all, with every write having reported success — which is exactly the case this exists to
    prevent, arrived at from the other side.

    Walking to the root costs one directory fsync per level, and a directory with nothing dirty
    behind it costs a syscall rather than a disk flush — small against the record's own file
    sync, which every row pays anyway, and bounded by the depth of the path. What it buys is
    that the publishing is unconditional. The syncs themselves stay best-effort (see
    :func:`_fsync_dir`, where a filesystem that refuses one may not fail the write it was
    protecting), so what is guaranteed is that nothing on the path goes unattempted, not that a
    hostile filesystem was talked into it. Top down, so a crash part-way through leaves a
    durable prefix rather than a durable entry inside an ancestor that is still missing."""
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
            # Taken from the dispense, which is the only thing this row is built from and the
            # only place the regime could have been recorded before the crash. Defaulting it
            # instead would make every abandoned task of a practice run read back as
            # evaluation-grade — a row that is *more* trustworthy-looking than the run that
            # produced it, which is the one direction this record may never round in.
            feedback_regime=_recorded_regime(record),
        )
        for record in read_dispenses(prov_dir)
        if record["lease"] not in sealed
    ]


class FeedbackPolicy(ABC):
    """What a terminating call tells the agent about the task it just ended.

    The two postures a run can be in are the same machinery with one thing swapped, so they are
    one object rather than two servers: a training loop needs the graded record echoed back
    (that dense signal is what an agent improves on), and an evaluation needs the channel shut.
    Naming the choice at the construction site is what lets a review ask "is this surface
    constructed as evaluation or as practice?" — a question a boolean buried in a config cannot
    be asked.

    **A policy decides which feedback items are revealed, and nothing else.** It is handed the
    sealed row's episode-level items and its answer is placed inside an envelope this module
    owns, under one added member (see :meth:`TaskStream._terminal_answer`). It is never given —
    and so can never reveal — the lease, the queue position, the task index, the closure, the
    diagnostic, the queue counts, or the fact that the stream has stopped. That containment is
    what makes the security argument a property of a few lines here rather than of whichever
    object a caller passed, and it is what will let ``Delayed(k)``, ``Batched(n)`` and
    ``Noisy(p)`` arrive as new policies rather than as new surface.

    **Subclassing this is not how a policy is admitted.** A stream serves only the exact policy
    types this module lists in :data:`_POLICIES`, and takes both values below *and*
    :meth:`reveal` itself from that list rather than from the object it was handed. Deriving from
    this class elsewhere produces something a stream refuses, by name, at construction — because
    the pair below is not two independent assertions a policy gets to make about itself, and a
    subclass free to make them separately can reveal a verdict while stamping every row of the run
    with the regime that has no channel. Adding ``Delayed(k)``, ``Batched(n)`` or ``Noisy(p)``
    means writing it here and listing it there; the envelope containment above is what keeps that
    a policy rather than new surface. This class stays public because it is the type of
    :attr:`TaskStream.feedback` and the name for the concept — not because it is an extension
    point.

    **An instance is a marker.** A stream reads no attribute of the object it was passed: the type
    is the whole of what it decides, and the regime, the reveal flag and the behaviour all come
    from the table. The two shipped policies are frozen and fieldless, so every instance of one is
    every other; a parameterised policy admitted later carries its parameter, and that parameter is
    the caller's to choose — what stays this module's is the code that reads it.

    Two class attributes, both taken once when a stream is constructed and kept (they decide a
    wire shape and a value written into every record — see :class:`TaskStream`):

    ``regime``
        the name stamped onto every dispense record and every result row this policy serves.
        It is what makes "these scores were earned with no verdict channel open" checkable from
        the artifact instead of from prose. One regime, one policy: it is what a resumed run
        compares itself against, so two policies sharing a name would make that check pass over
        a record they did not both write.
    ``reveals``
        whether a terminating call carries the feedback member **at all**. It is a property of
        the policy, never of the task: under a revealing policy the member is always present,
        so an empty list is the answer to "this task published nothing", "this policy is
        holding it back" and "no row was recorded" alike, and none of those is readable off the
        shape of the response.
    """

    #: See the class docstring. Declared, not defaulted: a policy that forgets to name its regime
    #: would otherwise file its rows under some inherited name, and a run's record would say a
    #: channel was closed that was open.
    regime: ClassVar[str]
    reveals: ClassVar[bool]

    @abstractmethod
    def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        """What the terminating call reveals, given the sealed row's episode-level feedback.

        ``published`` is a private copy of the items the row records under ``observed`` at
        ``level == "episode"``, in publication order — the env's own values, in the form the
        file holds them. Returning them is :class:`Immediate`; returning nothing is what a
        holding policy does on a task it is not answering yet.

        Called once per terminating call, synchronously, outside every lock, after the row is
        durable. It is not called at all under a policy whose ``reveals`` is false.

        **The stream calls this function, not this method** — the one snapshotted from the admitted
        class in :data:`_POLICIES`, applied to the instance. An entry in an instance's dictionary
        by this name is therefore never found, never called and never refused; it is simply not
        where a stream looks. The signature is what a policy in this module implements, not an
        interface anything outside it can serve over."""
        ...


@dataclass(frozen=True)
class Never(FeedbackPolicy):
    """No verdict channel: a terminating call answers with the fixed payload, the same bytes for
    every env, every task and every outcome. **The default**, and the only posture whose scores
    this package calls defensible — see :class:`EvalStream`, which is this policy made structural.

    :meth:`reveal` exists because the base class declares it and answers with nothing; the stream
    never calls it, because ``reveals`` is what decides whether a response has the member at all
    and a policy that never reveals has no answer to compose."""

    regime: ClassVar[str] = _NEVER_REGIME
    reveals: ClassVar[bool] = False

    def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        return ()


@dataclass(frozen=True)
class Immediate(FeedbackPolicy):
    """The sealed task's own feedback comes back on the call that ended it — the training
    posture, for a run whose point is that the agent improves between tasks.

    Verbatim and nothing more: the env's published episode-level items, exactly as
    ``results.jsonl`` records them. Not the :class:`Score` summary beside them, which is this
    record's reading of those same items rather than a value the env published — reporting it
    would put a number on the wire that no env ever emitted, and would make the response depend
    on whether that reading succeeded.

    What an env chooses to publish is still the env's own business: an env whose feedback names
    its target hands that target over here, because that is the feedback. That is not a defect of
    this policy, it is what "the verdict channel is open" means — and it is why :class:`Never` is
    the default and why an evaluation is a construction rather than an argument."""

    regime: ClassVar[str] = _IMMEDIATE_REGIME
    reveals: ClassVar[bool] = True

    def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        return published


def _channel(published: Sequence[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    """The item the env filed under ``name``, renamed to the one name a revealed item carries.

    Two jobs, and the second is what makes a pair of policies a pair. Selection is by name rather
    than by position, because an env publishes what it publishes: a run whose env emitted the
    summary numbers first and a run whose env emitted them last must open the same channel, and an
    index would make the answer depend on an ordering nothing in the contract fixes.

    **The rename is the point.** An env files its two versions under two names so the record can
    tell them apart, but an item that reached the agent still carrying ``notice`` would announce
    its own arm: the control could be identified from the field name without reading a byte of the
    value. Revealed items are therefore all named :data:`CHANNEL_FEEDBACK_NAME`, so the two arms'
    serialized answers differ in the value and in nothing else. The record is unaffected, because
    it stores what the env published rather than what the policy revealed.

    One item, not a list. An env that files two under one name has published something this
    contract has no reading of, and the first is taken rather than both, so the shape an agent
    sees cannot vary with a mistake upstream."""
    for item in published:
        if item.get("name") == name:
            return [{**dict(item), "name": CHANNEL_FEEDBACK_NAME}]
    return []


@dataclass(frozen=True)
class Information(FeedbackPolicy):
    """One channel open: the item the env published under
    :data:`~shogym.feedback.wire.REPORT_FEEDBACK_NAME`, and nothing beside it.

    The treatment half of a matched pair (see :class:`Placebo`). Where :class:`Immediate` hands
    back everything the row records (the summary numbers included), this hands back the single
    item the env wrote *for the agent to read*, so what a graded ending tells the agent is
    something the env composed rather than a list whose length and contents vary with how many
    metrics that env happens to publish. An env with nothing under that name answers with an
    empty member, exactly as a holding policy would.

    The channel is the env's to fill and the env's to be honest about: an env that names the
    answer in its report hands the answer over here. That is what an open channel is, and it is
    why :class:`Never` remains the default.

    What reaches the agent is named :data:`~shogym.feedback.wire.CHANNEL_FEEDBACK_NAME`, not
    ``report``: see :func:`_channel`."""

    regime: ClassVar[str] = _INFORMATION_REGIME
    reveals: ClassVar[bool] = True

    def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        return _channel(published, REPORT_FEEDBACK_NAME)


@dataclass(frozen=True)
class Placebo(FeedbackPolicy):
    """The same channel, filled with the env's inert stand-in: the item published under
    :data:`~shogym.feedback.wire.NOTICE_FEEDBACK_NAME`, and nothing beside it.

    The control half of the pair. It exists because "told nothing" and "told something that says
    nothing" are different treatments, and :class:`Never` can only serve the first: a run under
    it answers with a member-less envelope, so an agent under :class:`Information` is handed both
    a verdict *and* a channel that a :class:`Never` agent never sees at all. This keeps the
    channel, the member and the shape, and changes only what is in it, which is the comparison a
    paired design is trying to make.

    **The match is the env's to hold up.** Nothing here checks that the notice an env published is
    the length of its report or that it says nothing evaluative; a policy reveals, it does not
    author (see :class:`FeedbackPolicy`). What this side of it guarantees is that the two arms
    differ in the value and in nothing else: one item, the same member, and the same field name,
    because a revealed item is renamed to
    :data:`~shogym.feedback.wire.CHANNEL_FEEDBACK_NAME` before it goes out (see
    :func:`_channel`). An arm that announced itself in the field name would not need its value
    read to be recognised."""

    regime: ClassVar[str] = _PLACEBO_REGIME
    reveals: ClassVar[bool] = True

    def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        return _channel(published, NOTICE_FEEDBACK_NAME)


# The policies a stream may serve under, and — the point of the table — everything each one
# decides: `(policy type, regime, reveals, reveal)`, snapshotted here at import from the classes
# above, so there is exactly one place that says what "immediate" means and it is not an attribute
# of an object a caller owns.
#
# **The fourth member is the *behaviour*, not just a description of it.** A table that named the
# regime and then invoked `policy.reveal(...)` would have moved two of a policy's three decisions
# out of the caller's reach and left the one that produces the bytes the agent reads: an admitted
# `Immediate()` with `object.__setattr__(policy, "reveal", ...)` fabricates the feedback on the
# terminating call while every stamp in the record honestly says `immediate`, so the record shows
# no evidence of the signal the agent was actually shown — and the next run's scores were earned
# under it. `Immediate` promises the sealed row's items verbatim; the promise has to be kept by
# this module's code, so the function snapshotted here is the function called. Snapshotted rather
# than looked up per call for the same reason the regime is: a value read once at import cannot be
# rebound afterwards by anything short of editing this module.
#
# **An allow-list rather than a base class, because `regime` and `reveals` are not independent
# facts.** Trusted from the instance, a policy could name itself `never` and reveal anyway: the
# terminal would carry the verdict while every dispense record and every result row stamped the
# regime with no channel, and a later genuine `Never` resume would be waved through because the
# resume check compares two caller-chosen strings rather than two policies. That is a record that
# is wrong while looking correct, which is the one failure this file exists to make impossible —
# so the regime a run is stamped with is derived from *which* policy it is, and the set of
# policies is closed.
#
# **How a policy is admitted, and why that is the launch gate rather than the design.** `Delayed`,
# `Batched` and `Noisy` are still policies rather than new surface — the envelope containment in
# `_terminal_answer` is what makes that true and none of it changes. What changes is where a
# policy comes from: it is written in this module, given a regime no other entry uses, and added
# to this tuple. It is not subclassed downstream, because a stream cannot check a claim a
# downstream class makes about itself, and the record's claim about the regime is exactly what a
# reader has to be able to trust. One line of this tuple is the whole of what the next one costs.
_Reveal = Callable[[Any, Sequence[Dict[str, Any]]], Sequence[Dict[str, Any]]]

_POLICIES: Tuple[Tuple[type, str, bool, _Reveal], ...] = (
    (Never, Never.regime, Never.reveals, Never.reveal),
    (Immediate, Immediate.regime, Immediate.reveals, Immediate.reveal),
    (Information, Information.regime, Information.reveals, Information.reveal),
    (Placebo, Placebo.regime, Placebo.reveals, Placebo.reveal),
)


def _admitted(policy: Any) -> Optional[Tuple[str, bool, _Reveal]]:
    """The ``(regime, reveals, reveal)`` triple ``policy`` is served under, or ``None`` if it is
    not one this module admits (see :data:`_POLICIES`).

    **Identity, scanned — never ``isinstance``, never a dict keyed by the type.** Each of the
    other two is defeated by an object that merely says the right thing, and both were measured
    rather than assumed. ``isinstance`` passes anything deriving from :class:`FeedbackPolicy`,
    which is the hole itself, and it also passes an object whose ``__class__`` is a property
    answering ``Never``; a ``dict`` lookup runs the type's *metaclass* ``__hash__`` and ``__eq__``,
    so a class with a metaclass answering equal to :class:`Never` is handed :class:`Never`'s
    regime. ``type(x) is C`` consults neither: ``type()`` reports the real type whatever
    ``__class__`` says, and ``is`` cannot be hooked. It is the reason ``_require_task_ref`` and
    the namespace check spell their type tests the same way.

    **All three come from here and not from the object**, which is what makes the exact type
    sufficient. ``Never`` and ``Immediate`` are frozen dataclasses whose ``regime`` and ``reveals``
    are ``ClassVar`` — so neither is a constructor argument and ordinary assignment raises — but
    frozen is not sealed: they inherit a ``__dict__``, and ``object.__setattr__(Never(), "reveals",
    True)`` shadows the class attribute on that instance with no subclass in sight. The same
    mechanism shadows the *method*: ``object.__setattr__(Immediate(), "reveal", ...)`` puts a
    caller's function where the lookup finds it first, and an admitted ``Immediate`` then answers
    the agent with feedback no env published while every stamp says ``immediate``. So the type
    decides, and the type is all the instance is ever consulted for. Nothing on it is read: not the
    regime, not the reveal flag, not the behaviour.

    That leaves the instance as a *marker* — which is what the two shipped policies are: frozen,
    fieldless, and identical to every other instance of their type. It is passed to the function
    below as ``self`` all the same, because the parameterised policies the design anticipates
    (``Delayed(k)``, ``Batched(n)``, ``Noisy(p)``) are markers carrying a number, and the number is
    the caller's to choose. The line this draws is behaviour against configuration: what runs is
    always this module's code for that exact type, and what that code may read from the caller is
    whatever it was written to read. A shadowed ``k`` changes how long a policy holds a verdict
    back — which is what passing ``k`` does anyway; a shadowed ``reveal`` changed what the agent
    was told a task scored, under a stamp saying otherwise."""
    for admitted, regime, reveals, reveal in _POLICIES:
        if type(policy) is admitted:
            return regime, reveals, reveal
    return None


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

            **It is called on the serving loop, and a slow factory is felt by every live
            sibling.** :meth:`get_task` builds the env before it reaches its first await, so for
            as long as a constructor runs no other episode can dispatch and the watchdog cannot
            enforce anyone's deadline: a factory that takes seconds delays a 50 ms heartbeat by
            seconds. That is measurable with the AppWorld port, whose construction walks a corpus
            and an installed interpreter. Keep a factory cheap, or run the run under the
            ``off_loop_factory`` contract added by the stacked lifecycle branch, which moves the
            call onto the episode's own thread and owns what a cancelled caller leaves behind.
            This module does not offer a second mechanism for it.
        tasks: the materialised queue. Non-empty; may repeat a task index. An env key is a
            private label while the queue names one env — anything at all, ``__`` included,
            since the wire carries only the env's own tool names. Name a second env and every
            key is joined into a public tool name, so each one must be a name a tool may be
            called: 1 to 128 characters of letters, digits, ``_``, ``-`` and ``.``, no ``__``,
            and short enough that ``<key>__<tool>`` still fits. Checked at construction, on the
            joined names, and refused rather than encoded — a key that were slugged onto the
            wire would no longer be the key its own rows are filed under.
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
        feedback: what a terminating call tells the agent about the task it just ended (see
            :class:`FeedbackPolicy`). :class:`Never`, the default, answers with the fixed
            payload and opens no verdict channel; :class:`Immediate` answers with the sealed
            row's own episode-level feedback, verbatim; :class:`Information` and :class:`Placebo`
            are a matched pair, each answering with exactly one item under one public name, so
            two arms of a paired design differ in what that item says and in nothing else. Those
            four and nothing else: the policies
            a stream serves under are an allow-list of exact types (see :data:`_POLICIES`), and
            the regime written into every dispense record and every result row is taken from that
            list rather than from the object passed here — so the posture a run served under is a
            property of the artifact rather than a claim the argument made about itself. Use
            :class:`EvalStream` rather than this argument for a run whose scores are meant to be
            evaluation-grade: the argument says what this run did, and the construction says what
            no argument could undo.
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
        # One instance, built at import and shared by every stream that takes the default. Safe
        # because `Never` is frozen and holds nothing per-run; a policy that ever holds state
        # (a `Delayed` queue, a `Noisy` generator) may not be a default for exactly that reason.
        feedback: FeedbackPolicy = Never(),
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
        # An allow-list of exact types, and the pair it hands back is this module's rather than
        # the object's (see `_admitted`). What a policy decides is whether the agent is told its
        # verdict AND what every record of this run says about that, and those two may not be
        # separately assertable: anything that merely quacks like a policy — `True`,
        # `"immediate"`, an object with a `reveal`, a `FeedbackPolicy` subclass naming itself
        # `never` — would decide the first by whatever it answers and the second by whatever it
        # claims, with the answer already on the wire under a stamp denying it.
        admitted = _admitted(feedback)
        if admitted is None:
            raise ValueError(
                "feedback must be one of "
                f"{', '.join(policy.__name__ + '()' for policy, *_ in _POLICIES)}, got "
                f"{feedback!r} ({type(feedback).__name__}); a policy decides whether a "
                "terminating call carries the task's verdict *and* names the regime stamped on "
                "every record this run keeps, so only a policy this module defines can be "
                "trusted to make those two say the same thing. A new policy is admitted by being "
                f"added to this module, beside {Never.__name__} and {Immediate.__name__}, not by "
                f"subclassing {FeedbackPolicy.__name__} elsewhere"
            )
        # Taken once, here, and kept — the rule `provenance` namespaces follow, and for a stricter
        # reason: re-deriving them per task would let a run advertise one regime in its record
        # while answering under another, and these decide both a wire shape and a value written
        # into every record. The third is the module's own `reveal` for this exact type, so the
        # answer the agent is given comes from the same table the stamp does (see `_admitted`).
        regime, reveals, reveal = admitted
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
        # Checked, not merely rebuilt: these two fields are the identity every record of this run
        # is filed under (see :func:`_require_task_ref`).
        queue = [_require_task_ref(ref) for ref in tasks]
        if not queue:
            raise ValueError(
                "a stream needs a non-empty queue: its published tool manifest is derived "
                "from the queued tasks"
            )
        env_names = sorted({ref.env for ref in queue})
        # Whether the env key reaches the wire at all — decided here, because it is what decides
        # whether any of the rules below are this stream's to apply. With one env the stream
        # advertises the env's own tool names untouched, the key stays the internal caller-chosen
        # key it has always been, and a key is free to be anything. With several it is joined into
        # every public tool name, so the constraints on a *tool name* start applying to it. The
        # checks are therefore gated rather than unconditional: a single-env queue is exactly the
        # wire it was before this stream could serve more than one env, down to a key holding
        # `__`.
        prefixed = len(env_names) > 1
        if prefixed:
            for env_name in env_names:
                if _SEPARATOR in env_name:
                    raise ValueError(
                        f"env key {env_name!r} contains {_SEPARATOR!r}, which the stream uses to "
                        "join an env key to a tool name when it serves several envs; the joined "
                        "name would be ambiguous"
                    )
                defect = _unregistrable(env_name)
                if defect is not None:
                    # Refused, not encoded into something legal. A slug would make the name the
                    # agent calls stop matching the key the caller wrote, which is also the key
                    # every row and `results_by_env` is filed under — the endpoint and the record
                    # would then disagree about what this env is called, silently. The caller
                    # chose the key and can choose another.
                    raise ValueError(
                        f"env key {env_name!r} is joined into every tool name this stream "
                        f"advertises, because the queue names several envs, and {defect}"
                    )

        self._env_for = env_for
        self._queue: List[TaskRef] = queue
        self._max_in_flight = max_in_flight
        self._deadline = deadline
        # The validated policy, bound to the three values it was validated for — what the rest of
        # this object reads, so nothing it does can disagree with what was checked. The instance is
        # kept only to hand back from `feedback` and to pass as `self` to the function beside it;
        # no attribute of it is ever read.
        self._feedback = feedback
        self._regime = regime
        self._reveals = reveals
        self._reveal = reveal
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
        self.claim_path = self.prov_dir / _CLAIM_FILE
        # This stream's identity in the directory it owns, minted before the claim it goes into.
        # Unguessable rather than merely unique because it is half of what "still mine" means at
        # every later write: a counter or a pid would be reproduced by the next process to hold
        # that number, and a directory taken over is exactly when that matters. The other half is
        # the pid in the claim, which this cannot supply — a token is a value in memory and `fork`
        # copies memory, so an inherited stream's token is its parent's (see `_holds_claim`).
        self._owner = secrets.token_hex(16)
        # The third fact of the same identity, and the only one that is not a value: which
        # *object* the token was minted for. Both of the others are copied by any duplication —
        # the token is a string and the pid is an int, so a second object made out of this one's
        # state carries both and answers `_holds_claim` exactly as this one does. A weak
        # reference cannot be corrected that way: it is taken here, against this object, and a
        # duplicate holds the *same reference*, which still names the original (`weakref.ref` is
        # atomic to `deepcopy` and unpicklable, so no copier fixes it up), so `self._identity()
        # is self` is true here and false in anything made out of this. Weak so that the witness
        # costs the object no reference cycle; if it were ever dead the answer is `False`, which
        # is a refusal, never a grant. The duplication protocols are refused outright before any
        # of this is reached (see `_refuse_duplication`) — this is what makes the *append* sound
        # against a duplicate built some way no protocol of ours is asked about.
        self._identity: "weakref.ReferenceType[TaskStream]" = weakref.ref(self)
        self._claimed = False
        # The claim a `resume=True` took over, kept only until this constructor is past every
        # refusal: one that raises puts it back (see `_release_claim`).
        self._displaced: Optional[Dict[str, Any]] = None
        self._made_prov_dir = False
        # Taken here, ahead of the catalog: it is the cheapest refusal this constructor has, and
        # refusing before a factory is called costs no env the caller would then have to see
        # closed. Ownership is one call and not two — what the directory already holds is checked
        # inside the same exclusion that installs the claim, because an answer read outside it
        # authorises nothing (see `_claim_provenance` and `_require_fresh_provenance`).
        self._claim_provenance(resume)

        # One long-lived env per env name, used only to read the published contract (it never
        # begins a session, so closing it releases nothing an episode owns). Constructing it is
        # also what provisions an env whose data is fetched lazily.
        #
        # Everything from here to the end of construction runs under a cleanup guard: a factory
        # may provision real resources, and a constructor that raises hands back no object, so
        # nothing else could ever close what it built. That covers a partly-built catalog as
        # well as any later check that refuses the queue — and the claim just taken, which no
        # `aclose` will ever come to release because there is no stream to call it on.
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
            # from it. Everything downstream is one of two things: a copy of these values (the
            # signature the drift check compares, the score terminal the stream drives, the route
            # a call resolves to, the note a renamed task is told) or a name built out of them.
            # Freezing only what the endpoint advertises would leave every one of those carrying
            # the env's own objects while the two ends of the call — the endpoint and the episode
            # — were canonical: the agent makes exactly the call it was shown, the *bridge*
            # between them runs the env's code on the way past, and the call is lost with nothing
            # in the record saying a name was ever involved.
            self._manifest: Dict[str, List[ToolManifest]] = {
                name: _frozen_manifest(name, self._validate_manifest(name)) for name in env_names
            }
            # The frozen contract, in comparable form. Every episode this stream starts runs on
            # a *different* instance the factory built, so this is what each one's own manifest
            # is checked against before its task is dispensed. Compared on the NATIVE names the
            # env publishes; the `<env>__<tool>` prefixing below is derived from it afterwards,
            # so the check never has to know a prefix exists.
            self._signature: Dict[str, Tuple[Any, ...]] = {
                name: _manifest_signature(tools) for name, tools in self._manifest.items()
            }
            self._score_terminal: Dict[str, Optional[str]] = {
                name: next((m.name for m in tools if m.terminal_kind == "score"), None)
                for name, tools in self._manifest.items()
            }
            # What the agent actually sees, per env: the native manifest at capacity 1 with a single
            # env, wrapped with the required `lease` above capacity 1, and prefixed `<env>__<tool>`
            # when more than one env is in play (`prefixed`, decided above the catalog).
            self._advertised: Dict[str, List[ToolManifest]] = {}
            # The ONLY way a public tool name is resolved: an explicit map built here, never a split
            # of the published string. It is the collision check too — one public name, one owner,
            # whichever pair of halves produced it (see `_SEPARATOR`).
            self._routes: Dict[str, Tuple[str, str]] = {}
            # What each env's tasks are told their tools are called, when that is not what the env
            # itself calls them. `None` when nothing was renamed, and then the framing is silent.
            self._naming: Dict[str, Optional[str]] = {}
            for env_name, tools in self._manifest.items():
                advertised: List[ToolManifest] = []
                renamed: List[Tuple[str, str]] = []
                for manifest in tools:
                    if prefixed:
                        if _SEPARATOR in manifest.name:
                            raise ValueError(
                                f"env {env_name!r} advertises a tool named {manifest.name!r}, "
                                f"which contains the {_SEPARATOR!r} the stream uses to join an "
                                "env key to a tool name when it serves several envs; the joined "
                                "name would be ambiguous"
                            )
                        public = f"{env_name}{_SEPARATOR}{manifest.name}"
                        # Checked on the joined string, which is the one that goes on the wire.
                        # Neither half decides this on its own: the key is already known to be a
                        # legal name and each of these tools' names may be too, while the join of
                        # two legal names can still be longer than a tool name may be.
                        defect = _unregistrable(public)
                        if defect is not None:
                            raise ValueError(
                                f"env {env_name!r} advertises a tool named {manifest.name!r}, "
                                f"which this stream would register as {public!r} because the "
                                f"queue names several envs — and {defect}"
                            )
                    else:
                        # Nothing is joined, so there is no name here for the stream to have made:
                        # `public` is the env's own, on the wire exactly as it was before this
                        # stream could serve more than one env. Whether an env's own tool names
                        # are legal is that env's contract with its server, unchanged by this and
                        # not a join's to police — `sub__mit` is a perfectly good tool name.
                        public = manifest.name
                    # The tool itself is already this stream's own plain data (frozen native, at
                    # the top of this block). The *name* may not be: above one env it is a join,
                    # and the env key half of it has only ever been checked to be exact non-empty
                    # text — which an unpaired surrogate is. So the name the endpoint registers is
                    # proved here, whether it was made by this stream or published by the env.
                    try:
                        public = json.loads(_wire_json(public))
                    except (ValueError, TypeError, UnicodeError) as exc:
                        raise ValueError(
                            f"env {env_name!r} advertises a tool this endpoint could not put on "
                            f"the wire ({_rendered_failure(exc)}); the name a server registers "
                            "is what every task's framing carries and what a call arrives under, "
                            "so it has to be a value the endpoint can send"
                        ) from exc
                    frozen = {
                        "name": public,
                        "description": manifest.description,
                        "input_schema": manifest.input_schema,
                    }
                    clash = self._routes.get(public)
                    if clash is not None:
                        owner, taken = clash
                        if owner == env_name:
                            raise ValueError(
                                f"env {env_name!r} publishes two tools named {manifest.name!r}; a "
                                "server registers one schema per tool name, so the second would "
                                "stand in for the first"
                            )
                        raise ValueError(
                            f"tool {taken!r} of env {owner!r} and tool {manifest.name!r} of env "
                            f"{env_name!r} would both be advertised as {public!r}"
                        )
                    self._routes[public] = (env_name, manifest.name)
                    renamed.append((manifest.name, public))
                    shown = manifest.model_copy(update=frozen)
                    advertised.append(_leased_manifest(shown) if max_in_flight > 1 else shown)
                self._advertised[env_name] = advertised
                self._naming[env_name] = _naming_note(renamed) if prefixed else None
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
                    self._require_regime_matches(
                        _recorded_regime(record), source="a dispense record"
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
                    self._require_regime_matches(row.feedback_regime, source="a result row")
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
            self._watchdog: Optional[asyncio.Task[None]] = None
            self._releasing: Optional[asyncio.Task[None]] = None
        except BaseException as error:
            # Let the directory go before the envs, because this one cannot fail and cannot
            # block: a constructor that raises hands back no object, so a claim left behind here
            # is a directory nothing will ever serve and that only `resume=True` could reopen —
            # a refusal the caller earns by *fixing* what this call complained about. And the
            # directory itself, if this call is what made it: a refused construction has served
            # nothing and recorded nothing, and it may not be the reason a later `resume=True`
            # looks reasonable. Restoring, because a resume that refused took a claim over on its
            # way in: what it displaced goes back, so a refusal costs the stream that was already
            # serving nothing at all.
            self._release_claim(restoring=True)
            self._unmake_provenance()
            for note in self._close_catalog_now():
                error.add_note(note)
            raise

    # ----- an ownership identity is not duplicable -----

    def _refuse_duplication(self, attempted: str) -> NoReturn:
        """Refuse to hand back a second object made out of this one — **the one refusal behind
        every duplication surface below.**

        A stream is not a value. It holds an ownership claim on a provenance directory, the queue
        position it is up to, an unrepeatable ``seq``, the leases of the episodes that are live
        right now, and a catalog of envs it will close. Every one of those means "this object";
        none of them means anything when there are two. And the ownership machinery cannot tell
        the two apart on its own: the token is a string and the pid is an int, so a duplicate
        made inside this process carries both and passes :meth:`_holds_claim`, which is exactly
        what makes this worse than a benign copy. Both objects dispense the same queue position,
        both seal, and :meth:`_append_owned` — the check that makes "one record, one stream" true
        of a *record* — authorises both appends, because each one honestly is the stream that
        holds the claim. The record ends with two contradictory scored rows under one position and
        one ``seq``, both stamped with the same regime, neither stream stopped and both closes
        returning cleanly. That was reproduced through the public API with ``copy.copy`` and again
        with ``pickle``, on an exact :class:`EvalStream`, with no private mutation anywhere.

        So this is refused where a second usable object would be *created*, rather than caught
        later at a write: a copy that survives its constructor is a copy something already holds,
        and the loud refusal a caller can act on is the one that arrives before any task is
        dispensed. It is a :class:`TypeError` because it is a statement about the type — this
        operation does not exist for a stream, in the way :func:`pickle.dumps` refuses a lock.

        There is no supported second object. Two streams serving one queue is not a thing to
        arrange more carefully; the arrangements are: build the stream in the place that will
        serve it (a process, a thread, an object graph — all the same statement), serve two
        queues into two directories, or continue a record a stopped stream left behind with
        ``resume=True``, which takes the directory over deliberately and mints an identity of its
        own rather than borrowing one.

        **What is refused is every duplication a stream would otherwise perform on request**, not
        the one that was reported: ``copy``, ``deepcopy``, both halves of the pickle protocol, and
        the state a duplicate would be rebuilt out of. The rest of the surface was swept and is
        closed by absence rather than by a refusal, which is worth writing down so the next reader
        does not have to re-derive it. ``__getnewargs__``/``__getnewargs_ex__`` are consulted only
        by pickling, which stops above them. ``copy.replace`` (3.13, and this project pins below
        it) dispatches to a ``__replace__`` a class must opt into, and refuses a class that has
        none. There is no ``__setstate__`` either: nothing may be revived into a stream, so the
        half of the protocol that would accept a state does not exist. And nothing in this module
        builds a stream out of another stream — ``resume=True`` reads the *record* and mints a new
        identity, and :func:`build_stream_server` closes over the one object it was handed.

        This is the object-level twin of the ``fork`` refusal :meth:`_require_claim` explains, and
        the two together are the whole of "one stream, one identity": ``fork`` duplicates the
        object with the process, these duplicate it inside one. Neither is the last line of
        defence: :meth:`_holds_claim` asks which *object* holds the claim, so a duplicate built by
        some means neither of them is consulted about still cannot write."""
        raise TypeError(
            f"{attempted} of this {type(self).__name__} is refused: a stream is an ownership "
            f"identity, not a value. It claims {self.prov_dir} for this object, and a duplicate "
            "would carry its token, its queue position, its unrepeatable seq and its live leases "
            "— so both objects pass every ownership check, hand out the same queue positions and "
            "file contradictory rows under one seq, with nothing in the record saying which is "
            "illegitimate. Build the stream where it will serve, or continue a record a stopped "
            "stream left behind by constructing one with resume=True"
        )

    def __copy__(self) -> NoReturn:
        """``copy.copy`` is refused: see :meth:`_refuse_duplication`.

        Reached before ``copy`` falls back to the pickle protocol, so the message names the
        operation the caller actually performed. Shallow is the dangerous one, not the safe one:
        it shares the env catalog and the open episodes as well as duplicating the token, and it
        keeps the *scalar* queue position and ``seq``, which is precisely how two objects come to
        dispense position 0 twice."""
        self._refuse_duplication("copy.copy")

    def __deepcopy__(self, memo: Dict[int, Any]) -> NoReturn:
        """``copy.deepcopy`` is refused: see :meth:`_refuse_duplication`.

        A deep copy of the *stream* is not a way round the shallow refusal — it duplicates the
        same token, and the envs and provenance extensions it would copy on the way are objects
        with a live session and a directory behind them, which a copy does not get a second of.
        Refused here rather than left to fail somewhere inside the copy, since a partial deep copy
        of a serving stream is a mess with a claim in it. This also covers the case with no
        ``copy`` call in sight: deep-copying any structure that merely *holds* a stream (a config
        dict, a run record) arrives here, where the shallow copy of that same structure only
        aliases the stream and is fine."""
        self._refuse_duplication("copy.deepcopy")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        """Pickling is refused: see :meth:`_refuse_duplication`.

        This is the surface with the longest reach, and the one nobody types: it is what
        :mod:`pickle` calls at every protocol, and with it every library that moves an object
        somewhere — a ``spawn``/``forkserver`` :mod:`multiprocessing` argument, a process pool, a
        task queue, a cache. Unrefused it does not merely duplicate the identity, it *stores* it:
        the token is in the bytes, and a stream revived from them holds a claim on a directory
        whoever revived it may be serving right now, in this process, where the pid check matches
        by construction — and in another one, one day, where a recycled pid could make it match by
        accident. Reproduced before the refusal existed: the token was in the bytes, and the
        :class:`EvalStream` revived from them served into the original's directory alongside it,
        both closing cleanly over one queue position."""
        self._refuse_duplication("pickle")

    def __reduce__(self) -> NoReturn:
        """Pickling is refused: see :meth:`_refuse_duplication`.

        Redundant with :meth:`__reduce_ex__` for :mod:`pickle` itself, which never reaches here
        once that one raises, and not redundant for a caller that asks this object for its
        reconstructor directly: this is an ordinary method, the tuple it used to return rebuilds
        the stream token and all, and a serialiser that calls it rather than going through
        :mod:`pickle` never meets the method above. Both are defined because the surface being
        closed is "hand out a recipe for a second one", not "run ``pickle``" — measured rather
        than assumed, by deleting each in turn: without this one, ``stream.__reduce__()`` hands
        back a working reconstructor again while ``pickle.dumps`` still refuses."""
        self._refuse_duplication("pickle")

    def __getstate__(self) -> NoReturn:
        """Handing out this object's state is refused: see :meth:`_refuse_duplication`.

        The last step of the same family, and the one that is a plain method call rather than an
        operation: ``object.__getstate__`` returns the instance ``__dict__``, which is the token,
        the queue position and the ``seq``, and ``object.__new__(cls).__dict__.update(state)``
        turns that back into a serving stream. Unreachable through :mod:`copy` and :mod:`pickle`
        now that both are refused above, and defined anyway, because a refusal that depends on
        which door the caller came through is not a property of the object. There is deliberately
        no ``__setstate__`` to go with it: nothing may be revived into a stream, so there is
        nothing for one to do."""
        self._refuse_duplication("reading the state")

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
        refused at the one moment where nothing has been spent yet.

        **This is a check about what the directory already holds, and that is only half of it.**
        Two streams constructed against the same *empty* directory both pass it, because at the
        moment each one looks there is nothing to find — and then both serve, and the record ends
        up holding two runs' rows under one set of positions, in both regimes, with no stop
        anywhere. What closes that is :meth:`_claim_provenance`, which turns "nobody has recorded
        here" into "nobody else is serving here".

        **Which is why this runs inside that claim's critical section, and nowhere else. Callers
        hold :func:`_locked`.** An answer read outside the exclusion is a statement about a moment
        the claim does not cover, and the gap is wide enough for a whole other run: one that
        begins, seals and *releases* its claim inside it leaves a directory that holds records and
        is owned by nobody, so the paused constructor finds no claim to lose to and installs its
        own over a complete record. That was reproducible from the public API — an ``immediate``
        row and a ``never`` row for queue position 0, both ``seq`` 1, both streams closing
        normally. A check that does not hold its exclusion across the write it authorises
        authorises nothing, which is the rule :meth:`_append_owned` keeps for every append and the
        reason this is not also called before the lock: a second, cheaper answer to this question
        is one a later edit could mistake for the one that decides."""
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

    def _claim_provenance(self, resume: bool) -> None:
        """Take ownership of the provenance directory for as long as this stream serves into it,
        and refuse to serve at all if another stream holds it.

        **Why existence checks are not enough.** Everything :meth:`_require_fresh_provenance`
        refuses, it refuses by reading the directory — so it is a statement about the past, and
        two constructors reach it before either has a past. Both pass, both serve, and the record
        ends with two rows for one queue position, one stamped ``never`` and one ``immediate``,
        no stop raised and nothing on either row saying the other exists. Every guarantee this
        module makes about a record is a guarantee about *one* record written by *one* stream, and
        nothing so far said which stream that was.

        **The claim is the creation.** ``O_CREAT | O_EXCL`` is one atomic operation on every POSIX
        filesystem this serves from: of any number of processes attempting it, exactly one
        returns a descriptor and the rest get ``EEXIST``. So the loser is not racing a check it
        might win — there is no window in which both are owners, because ownership *is* the
        create. It is taken before the catalog is built and long before any append, so a refusal
        costs nothing and no directory is ever written to by a stream that did not own it first.

        What goes inside is this stream's ``owner`` token, the regime it serves under, the pid
        that took it, and the wall clock at the claim.

        **The pid says which process, never whether that process is alive.** Reading it as
        liveness would be inventing an oracle out of a number that means nothing across a
        container boundary, a host, or a pid wraparound, and the one thing worse than refusing a
        stale claim is silently breaking a live one. Comparing it with :func:`os.getpid` is a
        different question and always answerable: *am I the process that took this claim?* That is
        what makes ownership process-bound, and it is what a fork runs into — a child inherits
        this object and its unguessable token, so the token alone says yes to a process that never
        claimed anything (see :meth:`_require_claim`).

        **Staleness is the human's call, spelled ``resume=True``.** A stream killed mid-run leaves
        its claim behind, which is exactly the directory a crashed evaluation has to be resumed
        into. This cannot tell that claim from a live one and does not try: ``resume`` already
        means "I am continuing a run that stopped", which is the same assertion, made by the only
        party who can check it. So resuming takes the claim over — and takes it over through the
        same ``O_EXCL`` create, after unlinking the one it was shown, so that two callers who both
        assert it are not both believed.

        **The takeover is one step, not two, because the record is read between them.** Unlink and
        create are separate syscalls, and a stream displaced between them would find the directory
        unclaimed rather than claimed by someone else; worse, a displaced stream's append could
        land after the taker had already read the record it is about to continue, so the taker
        numbers a ``seq`` the record already holds. Both are closed by taking the whole of it
        under :func:`_locked`, the same exclusion every append takes — so a takeover and an append
        are ordered against each other rather than interleaved, whichever wins. What the loser of
        that ordering gets is a refusal at its next append and never a second row.

        **A fresh claim is refused by the *record* inside this same section.** A claim file and a
        record are the two halves of "nobody else is serving here", and only one of them survives
        a run that ends: a stream that closed in an orderly way removed its claim and left its
        rows. So the freshness refusal happens here, immediately before the create, rather than
        as a separate statement before the lock — see :meth:`_require_fresh_provenance` for the
        run that fits in the gap when it does not.

        **A claim carries the regime because the records may not have one yet.** A run killed
        between its claim and its first dispense leaves a directory with nothing in it to compare
        against, so :meth:`_require_regime_matches` — which reads records — has nothing to say,
        and a resume under the other posture would be waved through onto an empty file. The claim
        is the record of the regime that exists before any record does.

        **The replay this constructor seeds afterwards needs no lock of its own**, and it is the
        fresh path's ordering the right way round: a resume reads the record *after* taking the
        claim, never before. By then no other stream can add to it, because every append re-reads
        this claim inside the same exclusion and refuses if it is not the appender's. So the
        record a resume reads is the complete record of everything written before it — the one
        property seeding a ``seq``, a lease set and a done-position set depends on, and the one
        :meth:`_require_regime_matches` needs to be reading a whole posture rather than half of
        one. A *later* ``resume=True`` can still displace this stream while it reads, and rows in
        the other regime can appear under it as it does; that costs nothing, because the same
        claim it no longer holds refuses its own next append, so nothing it misread can become
        anything it writes."""
        # The directory has to exist before a create in it can be exclusive. Made in two steps so
        # that whether *this* call made it is knowable rather than inferred: a constructor that
        # goes on to raise has to leave the filesystem as it found it, and `exist_ok=True` cannot
        # tell "I made this" from "it was already here" — which is the difference between tidying
        # up after a refusal and deleting a directory the caller had prepared. Not made durable:
        # see `_CLAIM_FILE`.
        self.prov_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(self.prov_dir)
        except FileExistsError:
            pass
        else:
            self._made_prov_dir = True
        with _locked(self.prov_dir):
            if resume:
                held = self._read_claim()
                if held is not None:
                    # Checked before the takeover, not after: this is the only check that can see
                    # the regime of a run that was killed before it recorded anything, and once the
                    # claim is gone that evidence is gone with it.
                    self._require_regime_matches(
                        str(held.get("feedback_regime", _NEVER_REGIME)),
                        source="an ownership claim",
                    )
                    # The assertion `resume=True` makes, carried out. Nothing can come between this
                    # and the create below, so a claim that is gone here was gone before this
                    # stream arrived.
                    try:
                        self.claim_path.unlink()
                    except FileNotFoundError:
                        pass
                    else:
                        # Kept, because this constructor may still refuse: everything that can
                        # reject the queue, the record or the envs runs after this point, and a
                        # call that refuses has to leave the directory as it found it — which for
                        # a takeover means the claim it displaced, not merely no claim of its own
                        # (see :meth:`_release_claim`).
                        self._displaced = held
            # The freshness refusal, here rather than before the lock, because this is the only
            # place its answer can authorise anything (see :meth:`_require_fresh_provenance`).
            # Immediately before the create, so nothing — not even the takeover above — sits
            # between "this directory holds no run" and the claim that makes it stay that way.
            self._require_fresh_provenance(resume)
            try:
                self._write_claim(
                    {
                        "owner": self._owner,
                        "feedback_regime": self._regime,
                        # Read back by every append and every release, as identity: this is the
                        # process the claim was taken in, and no other may write under it (see
                        # :meth:`_require_claim`). The wall clock is for the human alone.
                        "pid": os.getpid(),
                        "claimed_at": time.time(),
                    }
                )
            except FileExistsError:
                held = self._read_claim() or {}
                raise ValueError(
                    f"{self.prov_dir} is claimed by another stream"
                    + _claim_detail(held)
                    + "; two streams serving one provenance directory write two runs' rows under "
                    "one set of queue positions, and nothing on a row says which run wrote it. "
                    "Serve this queue into a fresh provenance directory — or, if that stream is "
                    "gone and this run is continuing its record, pass resume=True to take the "
                    "directory over."
                ) from None
            self._claimed = True

    def _write_claim(self, payload: Mapping[str, Any]) -> None:
        """Create the claim file holding ``payload``, or raise ``FileExistsError`` if one is
        already there. **Callers hold :func:`_locked`.**

        The exclusive create is the whole mechanism (see :meth:`_claim_provenance`), so it stays in
        one place: the claim a stream takes and the claim a refused construction puts back are
        written by the same three lines, and neither can become an ordinary open under a later
        edit."""
        descriptor = os.open(self.claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, allow_nan=False) + "\n")

    def _read_claim(self) -> Optional[Dict[str, Any]]:
        """Whatever is in the claim file, or ``None`` if there is nothing readable there.

        Deliberately forgiving, because every caller is already deciding something safe on the
        strength of it: a claim that will not parse still *exists*, so the fresh path refuses on
        it (with a poorer message), and the ownership check treats it as not this stream's and
        stops. A missing owner reads as no owner, which no stream's token equals. So the failure
        mode of an unreadable claim is a refusal, never a silent grant."""
        try:
            held = json.loads(self.claim_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return held if isinstance(held, dict) else None

    def _holds_claim(self, held: Mapping[str, Any]) -> bool:
        """Whether ``held`` — a claim as it is on disk right now — is *this object, this stream,
        in this process*.

        Three facts, and no two of them are sufficient. The **token** says the claim was taken by
        this stream: unguessable, so no other stream can produce it and a directory taken over
        cannot be mistaken for one still held. The **pid** says the process asking is the process
        that took it, which the token cannot say, because a token is a value in memory and
        :func:`os.fork` copies memory. After a fork the child holds a stream whose token matches a
        claim it never took, whose queue position, ``seq`` and lease counter are its parent's, and
        whose appends land in its parent's files.

        The **object** says the same thing about a duplicate made *inside* one process, which
        neither of the others can say, because both of them are values and a duplicate is made out
        of this object's values: a ``copy.copy`` carries the identical token and runs under the
        identical pid, so it answered this question "yes" and appended a second scored row for a
        queue position the original was also serving. :meth:`_refuse_duplication` refuses the
        duplication protocols outright, which is where a caller wants to hear about it; this is
        the same statement made where the record is actually protected, at the append, so that a
        duplicate built by some means no protocol of ours is consulted about — ``__new__`` plus a
        copied ``__dict__`` — is caught by the ownership check rather than by an enumeration of
        the ways it might have been built. The witness is a weak reference taken at construction
        against the object it lives on, and a duplicate holds the original's reference rather than
        one to itself (see ``_identity``). A dead one reads as "not mine", which refuses.

        This is identity, not liveness. It never asks whether the pid in the claim is running —
        that question has no honest answer across a container boundary or a pid wraparound, and
        :meth:`_claim_provenance` refuses to invent it. It asks whether that pid is *this* one,
        which is a comparison of two numbers this process knows for certain.

        Read from what is on disk rather than from anything cached, and always inside
        :func:`_locked` by the two callers that act on it: a claim is a fact about the directory
        now, and a stream that consulted its own memory would answer this question with the
        moment it was constructed."""
        return (
            self._identity() is self
            and held.get("owner") == self._owner
            and held.get("pid") == os.getpid()
        )

    def _require_claim(self) -> None:
        """This stream still owns the directory it is about to write to, and this process is the
        one that owns it. **Callers hold :func:`_locked`** — see :meth:`_append_owned`, which is
        the only caller that appends.

        The ``O_EXCL`` create at construction is what makes ownership exclusive; this is what
        makes it *cover the run*. A claim taken once and never looked at again is a statement
        about the moment of construction, and the thing it has to rule out — two streams
        appending to one record — happens at every write after it. Four cases reach here, and
        all four were reproduced through the public API:

        * a ``resume=True`` that took the directory over from a stream still serving into it —
          including one with a task in flight, whose seal would otherwise append a second scored
          row for a position the taking-over stream replays;
        * two resumes racing, where one unlinks after the other has already created;
        * a process that forked a live stream, where the token matches and nothing else does;
        * an object duplicated out of a live stream, where the token *and* the pid match and only
          the object differs.

        All four end the same way: the stream that may not write finds out *before* its append
        rather than after it, and says so loudly. There is no quiet path — no row is written, no
        answer to the agent changes shape, and the stream stops, so the drain reports it.

        **Forking a live stream is refused, not supported.** A stream owns a provenance directory,
        a queue position, an unrepeatable ``seq`` and a registry of open episodes, none of which
        has a meaning that survives being duplicated; a child that served would file rows under
        numbers its parent is also using. If two processes are to serve, they serve two queues
        into two directories, or one crashes and the other resumes it with ``resume=True``. A
        ``fork`` for anything that does not touch the stream (a subprocess helper, say) is
        untouched by this: nothing is checked until the child tries to write.

        **Duplicating a live stream is refused the same way, and refused earlier.** The last case
        is the fork's twin inside one process, and :meth:`_refuse_duplication` closes it at every
        protocol a duplicate could be made through, so it is not a case a caller should ever reach
        by accident. It is answered here as well because this is where the record is defended: an
        ownership check that could be satisfied by an object made out of another object's state
        would be checking a value, not an owner."""
        held = self._read_claim() or {}
        if self._holds_claim(held):
            return
        if held.get("owner") == self._owner and held.get("pid") == os.getpid():
            raise RuntimeError(
                f"{self.prov_dir} is claimed by the stream this object was duplicated from"
                + _claim_detail(held)
                + f"; this is pid {os.getpid()}, the same process, so the claim was copied rather "
                "than taken. Two objects holding one claim would hand out the same queue "
                "positions and file contradictory rows under the same seq. A stream cannot be "
                "duplicated: build the stream where it will serve"
            )
        if held.get("owner") == self._owner:
            raise RuntimeError(
                f"{self.prov_dir} is claimed by this stream in another process"
                + _claim_detail(held)
                + f"; this is pid {os.getpid()}, so this stream was inherited across a fork. A "
                "stream cannot be shared that way: both processes would hand out the same queue "
                "positions and file rows under the same seq. Build a stream in the process that "
                "will serve it"
            )
        raise RuntimeError(
            f"{self.prov_dir} is no longer claimed by this stream"
            + _claim_detail(held)
            + "; another stream took the directory over — with resume=True, which asserts this "
            "one had stopped — so anything appended from here on would file two runs' records "
            "under one set of queue positions"
        )

    def _append_owned(self, path: Path, record: Dict[str, Any]) -> None:
        """Add one line to this stream's record — **the only way this module ever does.**

        Both durable logs go through here, because ownership that is checked before a write and
        not *held across* it is not ownership. Read the claim, find your own name, and append, and
        a takeover landing between the second and third steps leaves two streams that each passed
        an honest check and each wrote a row. That was reproducible from the public API on both
        logs: a resumed stream taking over while a dispense was in flight produced two ``seq=1``
        dispenses for one position, and a taking-over stream produced a second scored row for a
        position the displaced stream still had live.

        So the check and the append are one critical section, excluded across processes
        (:func:`_locked`) and synchronous within one, which is the whole of what makes "one
        record, one stream" true of a *record* rather than of a constructor. Whoever loses the
        ordering finds a claim that is not theirs and raises without writing.

        Everything in here is synchronous. Nothing awaits, so no cancellation point can split the
        check from the write it authorises, and the lock is never held across a suspension.

        The lock also settles a second thing the check never could: two processes appending at
        once **interleaved their bytes**. A record and its terminating newline are two writes (see
        :func:`_append_jsonl`), so a concurrent appender could land a record in the middle of
        another's line, leaving a file that will not parse at all — the record destroyed rather
        than merely doubled. One writer at a time is what rules that out.

        Durable always: both logs are read back after a crash, and a record that reached only the
        page cache is exactly the record a hard kill loses."""
        with _locked(self.prov_dir):
            self._require_claim()
            _append_jsonl(path, record, durable=True)

    def _release_claim(self, *, restoring: bool = False) -> None:
        """Let the provenance directory go, if this stream still holds it.

        Removed rather than left behind, so that a claim on disk means a stream that never got to
        finish. That is the whole of what makes ``resume=True`` an assertion about a *crash*: a
        run that closed in an orderly way leaves nothing for a later one to take over, so the
        only claim anyone is ever asked to break is one whose owner really did stop without
        releasing it.

        **Released only once nothing can write here again**, which is later than it looks: a seal
        whose append failed keeps its registry entry so a later drain can retry the *write* (see
        :meth:`_run_seal`), and a drain that was cancelled mid-seal leaves the same. Letting the
        claim go while a row was still owed would mean the retry that lands it is a stream writing
        into a directory it no longer owns — refused, correctly, and the row lost with it. So the
        drain releases this when its registry is empty, and a stream still owing a row holds the
        directory until the drain that finishes it (see :meth:`aclose`).

        Only ever removes its own — the same ownership every append is held to
        (:meth:`_holds_claim`), taken under the same exclusion (:func:`_locked`) for the same
        reason. A stream that lost the directory to a ``resume=True`` takeover has already stopped
        and must not delete the live claim of whoever took it; a forked child closing the stream
        it inherited must not release its parent's. Held across the read and the unlink because a
        release is a write like any other: the claim it removes has to be the claim it read, or a
        takeover arriving between them is undone by a stream that no longer owns anything.

        ``restoring`` is for the one caller that is undoing a takeover rather than ending a run:
        a constructor that took a directory over and then refused. Everything that can reject a
        queue, a record or an env runs after the claim is taken, and a call that refuses may not
        leave the filesystem changed — which for a resume means putting back the claim it
        displaced, not merely removing its own. Otherwise a mistyped queue would dispossess a
        stream that was serving, and the run it stopped would be a run nobody asked to stop.

        Never raises, and never masks. It runs in a failed constructor beside an error already on
        its way out, and in the shutdown release, which may not raise at all (see
        :meth:`_release_stream`). A claim that cannot be removed leaves a directory that
        ``resume=True`` reopens, which is a worse morning than a clean one and a much better one
        than a lost error."""
        if not self._claimed:
            return
        self._claimed = False
        displaced, self._displaced = self._displaced, None
        try:
            with _locked(self.prov_dir):
                if not self._holds_claim(self._read_claim() or {}):
                    return
                self.claim_path.unlink()
                if restoring and displaced is not None:
                    self._write_claim(displaced)
        except (OSError, ValueError):
            # `ValueError` for the restore alone: what goes back is a claim this run read off the
            # disk, and a file holding `NaN` parses and will not re-encode. A directory left
            # unclaimed is the same outcome as a stream that finished, which is the safe half of
            # this — the unlink has already happened, so what is lost is the *restore*.
            pass

    def _unmake_provenance(self) -> None:
        """Remove the provenance directory again, if this constructor is what created it and
        nothing was ever put in it. Only a failed construction calls this.

        ``rmdir`` and not anything recursive, so this cannot destroy a record: a directory holding
        so much as one file refuses to go, and the one file this call could have put there — the
        claim — was just removed. A directory that existed before this constructor ran is left
        alone whatever is in it, which is why whether this call made it is tracked rather than
        guessed (see :meth:`_claim_provenance`).

        Never raises, for :meth:`_release_claim`'s reason: it runs beside an error already on its
        way out, and a directory left behind is untidy where a masked error is wrong."""
        if not self._made_prov_dir:
            return
        self._made_prov_dir = False
        try:
            os.rmdir(self.prov_dir)
        except OSError:
            pass

    def _close_catalog_now(self) -> List[str]:
        """Release the catalog envs from *sync* code, which only a failed constructor needs:
        there is no stream to call :meth:`aclose` on, and nothing else holds these envs.

        Returns what could not be finished, for the caller to attach to the error it is already
        raising. Cleanup must not mask that error — but it must not be silent either: a swallowed
        close is an env left open with nothing left holding it, which is the whole failure this
        path exists to prevent.

        **Every env is let go independently**, the same guarantee :meth:`_release_stream` gives
        the orderly path and for the same reason: ``Env.close()`` is third-party code that may
        block for as long as it likes, and closed one at a time the first such env decides
        whether any env after it is closed at all. So every close is *started* before any of them
        is waited for. It still does not bound one — a hung env leaves this frame pending, since
        there is no wall clock over teardown here — it stops being everyone else's wait.

        Whether that is a complete close is decided once rather than per env, because it is a
        property of this frame and not of any env: with no loop running these are closed together
        on one temporary loop, and inside one they are handed to the loop that built them and
        finish just after this error propagates (see :func:`_close_on_owning_loop`).

        This frame never awaits, so no cancellation can be *delivered* into it: one observed here
        was raised by the env's own close, and it masks the error being raised exactly as any
        other failure would. Hence ``None`` for the cancellation (see
        :func:`_must_propagate`)."""
        notes: List[str] = []
        # Handed over before any of them is closed, so a close that raises cannot leave half the
        # catalog still on the registry with nobody left to answer for it.
        catalog = list(self._catalog.items())
        self._catalog.clear()
        scheduled, outcomes = _close_on_owning_loop(catalog)
        for name, failure in outcomes:
            if failure is not None:
                if _must_propagate(failure, None):
                    raise failure
                notes.append(
                    f"the catalog env for {name!r} could not be closed while this error was "
                    f"being raised ({_rendered_failure(failure)}); it may still hold resources"
                )
            elif scheduled:
                notes.append(
                    f"the catalog env for {name!r} is being closed on the loop that built "
                    "it; a synchronous constructor cannot await that, so the close finishes "
                    "just after this error propagates"
                )
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

    def _require_regime_matches(self, recorded: str, *, source: str) -> None:
        """A stored record must have been written under the regime this stream serves.

        A record is read as one run. Its rows are averaged together, and the claim a mean of them
        supports depends entirely on which channel the run served its tasks under — that is the
        difference between a benchmark number and a learning curve. So a directory holding rows
        from both postures is a record whose parts are individually honest and whose whole is
        not, which is precisely the failure a per-row stamp exists to make impossible: written on
        every row, then contradicted by the row beside it.

        What is compared is the *assignment*, and it is the right thing to compare: two runs that
        assigned different channels are two experiments whatever each delivery did, and a stamp
        that meant "delivered" could not be written when the row is (see :class:`ResultRow`).

        Only a resumed run can reach this, and that is the whole of the exposure — a stream that
        is not resuming refuses a directory holding *any* record (see
        :meth:`_require_fresh_provenance`). The record is not the only thing that can name a
        regime, though: a run killed between its claim and its first dispense recorded nothing at
        all, so the ownership claim carries the regime too and is compared here through the same
        check (see :meth:`_claim_provenance`). Refused at construction, before anything is spent,
        like the queue check beside it."""
        if recorded == self._regime:
            return
        raise ValueError(
            f"{self.prov_dir} holds {source} written under feedback regime {recorded!r}, but "
            f"this stream serves under {self._regime!r}; the rows of one record are read "
            "together, and which channel each task was served under is what says which claim "
            "their mean supports — resume under the regime the record was written with, or "
            "serve into a fresh provenance directory"
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

        A framing is the frozen contract plus two values off the episode's own spec, and the
        contract half is already plain data by the time it is advertised (see the constructor).
        These two are not. A model validates a field when it is built and not when it is
        assigned, so an env that edits its spec afterwards can publish anything at all as
        ``instructions`` or ``horizon``, and nothing between here and the wire looks at them:
        they are carried verbatim into :class:`DispensedTask` and serialised by whoever answers
        ``get_task``.

        **Where it is found decides what it costs.** The dispense is durable before the task is
        handed out, so a framing that fails after that point is a committed dispense the agent
        was never answered for — the episode is live, the drain ends it, and the row it lands
        says the agent played the task out and got it wrong. That is a wrong number where a
        missing one was the truth. Found here the bad state is unreachable rather than
        recoverable: nothing is written, the position is still owed, and the episode is let go by
        the same handler that answers a drifted manifest.

        Confirming is the whole of it, because there is nothing left to detach: ``str`` and
        ``int`` are immutable, so a confirmed one aliases nothing an env can reach through, and
        every other field of the framing is this stream's own or the frozen contract's. Reading
        them is still the env's code — the spec is the env's object — so the read is contained
        like every other read of an env's values here, and a value that cannot be read is not a
        different finding from one that cannot be carried.

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
                # the way out — and that failure lands where it costs most: the dispense is
                # already durable, so the answer the agent was owed never arrives, the drain ends
                # the task, and the row says it played the task out and got it wrong. Proved here
                # instead, on exactly the two values that go out (see :func:`_wire_json`).
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

        The framing is assembled from several sources, and only one of them was ever proved: the
        episode's ``instructions`` and ``budget`` (see :meth:`_require_framable`). The env key,
        the frozen contract, the naming note and the lease were added afterwards, past the point
        where the dispense is durable — so a value that cannot be encoded there costs a task: the
        endpoint answers ``get_task`` with a serialization error, the agent is handed nothing, and
        the drain then ends a task nobody ever received and records it as one the agent played and
        lost. An env key is an ordinary non-empty ``str`` as far as every check upstream of here
        is concerned, and one holding an unpaired surrogate is exactly that.

        So what is proved is the object itself, through the encoder the endpoint uses (see
        :func:`_wire_json`) and on the wire form it will actually answer with. Field by field it
        would be the same check with a list to keep in step; whole, a field added later is covered
        the day it is added.

        Nothing of this run's record is durable yet when this runs, so a refusal costs no task:
        the dispense is unwritten, the position is still owed and no row is due. The stop is the
        run's, on the line this module already draws — an env key, a tool naming note and a frozen
        contract are properties of the stream, not of this task, so the next dispense would fail
        the same way."""
        framing = DispensedTask(
            # The stream's key for this env: it is what `<env>__<tool>` is built from, so it has
            # to be what the agent is told the task belongs to.
            env=ref.env,
            # Verbatim. The env wrote this and it names the env's own tools; when those are not
            # the names this endpoint registers, `tool_naming` below says so beside it rather
            # than the stream editing an env author's prose (see `_naming_note`). The two values
            # `_require_framable` confirmed, not a fresh read of the spec.
            instructions=instructions,
            budget=budget,
            # Derived from the *published* manifest, never from the live episode's: this is the
            # contract the server registered, and the check above is what makes the two the same
            # thing. Detached from it, and per dispense, for the reason :func:`_detached_manifest`
            # gives: what a task is framed with is a reading of the frozen contract, not a handle
            # on it, and one task's framing is not the next one's either. Its name and description
            # are already this stream's own plain copies, taken when the contract was frozen.
            tools=tuple(
                {
                    "name": m.name,
                    "description": m.description,
                    "input_schema": copy.deepcopy(m.input_schema),
                }
                for m in self._advertised[ref.env]
            ),
            lease=lease if self._max_in_flight > 1 else None,
            # From the same map the tools above came from, so the framing cannot name a tool this
            # endpoint does not serve. `None` unless the stream renamed something.
            tool_naming=self._naming[ref.env],
        )
        try:
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
    def feedback(self) -> FeedbackPolicy:
        """The policy this stream serves under (see :class:`FeedbackPolicy`).

        Read-only, and read *once* at construction: what this hands back is the object the caller
        passed, but nothing about the run is read from it. Its exact type was matched against
        :data:`_POLICIES` then, and the regime stamped on the record, the decision to reveal at
        all and the function that composes what is revealed all come from that table. There is no
        setter, so the posture a run started in is the posture its whole record was written in —
        and no attribute of this object can change what a run does or what its record says."""
        return self._feedback

    @property
    def tools(self) -> Sequence[ToolManifest]:
        """Everything this stream advertises, across every env in its queue: the envs' own
        schemas at capacity 1 with a single env, lease-carrying wrappers above capacity 1, and
        ``<env>__<tool>`` names when more than one env is in play.

        A detached view of the frozen contract, rebuilt per read (see
        :func:`_detached_manifest`) — reading what this endpoint serves may not be a way to
        change it."""
        return tuple(
            _detached_manifest(manifest)
            for tools in self._advertised.values()
            for manifest in tools
        )

    @property
    def results_by_env(self) -> Dict[str, Tuple[ResultRow, ...]]:
        """The rows so far, grouped by env.

        Grouping is the default because 0.7 on one env and 0.7 on another are not the same
        quantity, and one mean over both silently mixes scales. Aggregating anyway is the
        caller's call to make — :attr:`results` is right there.

        Detached per read, like :attr:`results` and from the same recorded rows. Two accessors
        onto one record are two chances to hand it out, and a row reached through this one is no
        more the record's than a row reached through that one."""
        grouped: Dict[str, List[ResultRow]] = {name: [] for name in sorted(self._advertised)}
        for row in self._results:
            grouped.setdefault(row.env, []).append(_detached_row(row))
        return {name: tuple(rows) for name, rows in grouped.items()}

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

        **The provenance directory is let go last, and only when this stream can no longer write
        to it.** That is a stronger condition than "the drain returned": a seal whose append
        failed, and one a cancelled shutdown left mid-flight, both keep their registry entry so
        that a later ``aclose`` can retry the write — and a stream that had already released its
        claim would find the directory no longer its own and lose the row. So the claim is
        released here, after the watchdog is stopped and the envs are let go, and only with the
        registry empty (see :meth:`_release_claim`).

        Raises if anything stopped the stream, here or earlier in the run — a dispensed task
        that went unrecorded, a task that could not be recorded as dispensed at all, a summary
        the record cannot headline, an env that raised while a task ended, or an episode that
        would have been framed with a contract the endpoint does not serve. A call the agent
        made and the harness lost is **not** one of them: it costs its own task its score and
        says so on the row, and no other task's (see :meth:`dispatch`). Together with
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
            try:
                await self._released()
            finally:
                # The provenance directory goes last of all, and only when this stream can never
                # append to it again — which is not the same as "the drain returned". An entry is
                # retired when its row is durable, so a registry that still holds one is a stream
                # that still owes a write: a seal whose append failed on the storage, or one a
                # cancelled shutdown left mid-flight. Both are finished by a *later* `aclose`, and
                # a stream that had let its claim go by then would find the directory no longer
                # its own and lose the row it was retrying. So the claim outlives every such
                # retry, and a run killed before one arrives leaves the claim behind — which is
                # what it is for.
                #
                # Below the release above rather than inside it, because the deadline watchdog can
                # seal a task too: it is stopped in there, so by here the drain is the only thing
                # that could have written and it is over. In a `finally` of its own so that an env
                # whose `close` fails still leaves the directory unclaimed — that failure is the
                # run's to report, not a reason for the next run to need `resume=True`.
                if not self._live:
                    self._release_claim()
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
        moment finishes the seal it started instead of being cancelled halfway through it.

        **Every catalog env is let go independently.** ``Env.close()`` is third-party code that
        may block for as long as it likes — an MCP session that never answers, a subprocess that
        will not reap — and closed one at a time, the first such env decides whether any env
        after it is closed at all: they stay open, holding sessions and subprocesses, for exactly
        as long as it hangs. So each close is handed to its own task, and the handing over is one
        uninterruptible statement, so no env can be left waiting on another's turn. What that
        does *not* do is bound a close: a hung env still leaves this pending, since there is no
        wall clock over teardown here and inventing one would be a decision about how long an
        env's own cleanup may take. It stops being everyone else's wait.

        Nothing here may raise. This runs in :meth:`aclose`'s ``finally`` as the stream's own
        claimed task, so an exception escaping it would replace the run-level report the drain
        was about to make — here it would also leave every catalog env open, since they are let
        go after the watchdog — and, because the claim is the task and a failed one stays
        claimed, it would do the same on every later attempt. The watchdog already records what
        it could not seal (see :meth:`_watch_deadlines`); this is the backstop for a failure
        that did not, and it is recorded rather than swallowed so the drain still reports it.

        "Nothing may raise" includes ``CancelledError``, which an env's ``close`` can raise like
        any other exception — contained per env in :meth:`_close_catalog_env`. This task is joined
        through a shield and nothing in this module ever cancels it, so one arriving from an env
        is the env's, not a caller's, and letting it out would leave the release task *cancelled*
        while it is the claim: the envs it already popped are unreachable, and every later
        ``aclose`` re-awaits that same cancelled task and raises again. A shutdown with no
        orderly exit, for a teardown failure that is not the run's outcome."""
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
        # Popped and handed over in one statement with no await in it, so there is no state in
        # which some envs are the release's and the rest are still the registry's.
        closing = [
            asyncio.ensure_future(self._close_catalog_env(self._catalog.pop(name)))
            for name in list(self._catalog)
        ]
        # Every one of them is awaited before any is acted on: an env whose close fails must not
        # decide whether another env is closed at all, and an unread failure is asyncio's
        # unretrieved-exception warning. Containment already happened inside each task, so
        # anything still here is something no boundary in this module may swallow.
        for outcome in await asyncio.gather(*closing, return_exceptions=True):
            if isinstance(outcome, BaseException):
                raise outcome

    async def _close_catalog_env(self, env: Env) -> None:
        """Close one catalog env, containing what it raises. Teardown is best-effort and one
        env's failure is not another's, nor the run's outcome.

        The cancellation baseline is this task's own, which is what makes the containment
        correct here: nothing in this module cancels these, so a ``CancelledError`` observed
        inside one was raised by the env and is contained like any other failure. One delivered
        *to* this task could only come from the release being cancelled — and that is the caller's
        cancellation, which passes through (see :func:`_must_propagate`)."""
        cancellation = _Cancellation()
        try:
            await env.close()
        except BaseException as exc:  # noqa: BLE001 — teardown is best-effort
            if _must_propagate(exc, cancellation):
                raise

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
        no task. That is why a renamed tool is answered by ``tool_naming`` rather than by
        leaving the env's instructions to name something uncallable (see :func:`_naming_note`).

        The framing is also confirmed to be something the endpoint can *carry*, and confirmed
        before the dispense is committed (see :meth:`_require_framable`): a task that cannot be
        handed over has to be no task at all rather than a durable dispense with no answer.

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
                # The episode's own snapshot, taken when it was opened and the same for every
                # reader (see :meth:`ServedEpisode.describe`). That is what makes the check below
                # worth making: the manifest confirmed here, the framing built from it, and the
                # schema the episode's seal validates a terminal call against are one
                # description, so an env cannot pass the check with one contract and enforce
                # another on the agent that was framed with it.
                spec = episode.describe()
                self._require_published_manifest(ref.env, spec)
                # Both checks sit above `_write_dispense`, and this one for the second reason
                # that check gives: a task nobody could be handed is not a task, and finding that
                # out after the record is durable turns a task that was never served into a row
                # saying the agent served it badly (see :meth:`_require_framable`). What it
                # confirmed is what the framing below is built from — a second read of the spec
                # would be a second value, and an unchecked one.
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

            undispensed: Optional[ServedEpisode] = None
            unrecorded: Optional[BaseException] = None
            framing: Optional[DispensedTask] = None
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
                    # Built before anything is durable, and proved whole: what the agent is
                    # handed is this object, and every field of it has to be one the endpoint can
                    # answer with (see :meth:`_deliverable_framing`). A refusal here costs the
                    # same as a manifest this stream cannot serve — the position is still owed,
                    # no row is due, and the episode is let go with the refused dispense below.
                    #
                    # Minting sits inside the same guard, and for the same reason rather than for
                    # tidiness: it can refuse (see :meth:`_mint_lease`), and it runs where nothing
                    # else would catch it — a raise from the `_Live(...)` arguments would leave
                    # this method with `undispensed` unset, so the env and MCP sessions this
                    # episode holds would be let go by nobody at all.
                    try:
                        live = _Live(
                            lease=self._mint_lease(),
                            seq=self._seq + 1,
                            position=position,
                            ref=ref,
                            episode=episode,
                            spans=spans,
                            dispensed_extensions=dispensed_extensions,
                        )
                        framing = self._deliverable_framing(ref, instructions, budget, live.lease)
                    except BaseException as exc:
                        undispensed = episode
                        unrecorded = exc
                    else:
                        # Durable BEFORE the task is exposed: after this point a crash is
                        # reconcilable, because the record says a task was handed out and no
                        # result answers it. And nothing is committed before it: a position
                        # stepped over a write that failed is a task quietly dropped from the
                        # queue, its episode absent from the registry that is the only handle on
                        # it, and a drain that reports a clean run over the hole.
                        try:
                            self._write_dispense(live)
                        except BaseException as exc:
                            # Nothing was handed out, so this position is still owed and no row
                            # is due — the same shape as the manifest refusal above. But a
                            # provenance directory that cannot be appended to is not a per-task
                            # problem: the next dispense record and every result row after it go
                            # to that same directory, so the run can no longer be a record of the
                            # queue. Serving on would spend the rest of the queue against a file
                            # that already lost a task, so the stream stops and says so at both
                            # boundaries. (Cancellation is excluded, as everywhere else here:
                            # nothing failed.)
                            if not isinstance(exc, asyncio.CancelledError):
                                self._stop(
                                    exc,
                                    dispensing=(
                                        "this stream stopped: a task could not be recorded as "
                                        f"dispensed to {self.dispenses_path}, so a crash from "
                                        "here on could not be told apart from a task that was "
                                        "never handed out"
                                    ),
                                    closing=(
                                        "this stream could not record a dispense to "
                                        f"{self.dispenses_path}; the queue was not served to "
                                        "the end"
                                    ),
                                )
                            # Closed outside the lock, with the dispense that was refused before
                            # it: same window, same teardown, one path.
                            undispensed = episode
                            unrecorded = exc
                            framing = None
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
            assert framing is not None
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
        is over — a fixed payload identical for every task and every outcome — plus, under a
        revealing feedback policy, that task's own published episode-level feedback and nothing
        else (see :meth:`_terminal_answer` and :class:`FeedbackPolicy`). A call that reaches an
        episode already over is answered as terminating too — that is the episode's state and not
        this call's doing — and gets the same payload with the feedback member empty, because the
        reveal belongs to the call that sealed the task (see :meth:`_tombstone_answer`).
        Everything below describes the default, :class:`Never`.

        Everything a terminal produces stays with the harness, not just the row the seal records
        (lease, position, task index, raw feedback). The env's terminal response is redacted too:
        for a ``score`` terminal it is the verdict this stream just recorded, and a queue that
        repeats an index would make it the signal that identifies the repeat. The feedback sidecar
        a served episode rides its terminal feedback out on is dropped for the same reason and
        must stay dropped — relaying it, as the single-episode server does, would reopen the
        channel this closes. A fixed payload also says nothing about how the stream classified the
        ending: a caller able to tell an unscored infrastructure failure from an earned zero has a
        reason to cause one.

        A call that ends the task answers with that payload *whatever happened while it ended*,
        and under a revealing policy with the same members in the same shape. An exception is a
        different response shape, and the shape is the channel: an env that
        published a clean summary on one verdict and a malformed one on the other would tell the
        agent its verdict through whether the call succeeded. So a failed seal — and an env that
        raises once it has already ended the episode — are recorded on the stream and answered
        with the same bytes as a clean one. Only a call that leaves the task *live* still raises:
        there the exception is the env's own answer to a call the agent can make again, no
        different in kind from the env text an ordinary call returns, and no task has ended for
        it to be a verdict about.

        Answering that one is not the same as forgetting it. A call that raised reached no result,
        so nothing it was for is on the episode's record, and if the agent never does end the task
        then the outcome the stream composes is one nothing the agent did produced. The failure is
        therefore kept on the entry and consulted by the seal: an agent that goes on to end the
        task itself keeps whatever that terminal says, and a task the *stream* has to end instead
        lands unscored rather than in a scored closure it did not earn.

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
        live, native = resolved
        # Outside the registry lock: the episode has its own lock, holding a stream-wide one
        # across an awaited env call would serialise the whole stream behind one tool, and the
        # deadline needs that lock to arbitrate — an env slow in a tool or in its finalizer must
        # not be able to spend the wall clock and still be recorded as an ordinary seal.
        cancellation = _Cancellation()
        try:
            call = await live.episode.call(native, args)
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
                # agent can make again — but not on its own. It reached no result, so nothing
                # the agent asked for is on this episode's record, and the outcome the stream
                # would compose from that record is one the agent never got to play for. Left at
                # a bare re-raise this is the whole failure: the drain later drives the terminal
                # itself and files the task in a *scored* closure, so an agent whose submission
                # the harness dropped is recorded as one that answered wrong — a number a run's
                # mean would then average in. So the loss is kept on the entry and the seal
                # decides — kept and not acted on, because an agent that recovers and ends the
                # task itself has earned whatever that terminal says (see `_compose_row`). What
                # it costs is this task's score and nothing else: the stream serves on, since the
                # next task need not meet what this call met.
                if live.call_error is None:  # the first loss explains the run
                    live.call_error = exc
                raise
            # The episode ended and *then* the call failed: the terminal is committed and the
            # feedback the episode was about to hand over is what raised. So a row is still owed
            # — it lands carrying whatever feedback survived, which is what makes the loss
            # legible — and the stream stops. A row with no readable outcome is the same
            # eval-integrity failure whether the value is unusable or missing, and an env that
            # raises here raises for every task *of that env* in the queue; without the stop a
            # solved task is recorded unscored and `aclose()` reports a clean run. The stop is
            # the whole stream's even when the queue holds other envs: one record is missing an
            # outcome, and that is the run's, not this env's.
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
            return ToolResult(content=self._terminal_answer(await self._seal_redacted(live)))
        if not call.terminated:
            return ToolResult(content=json.dumps({"content": call.content, "terminated": False}))
        if call.tombstoned:
            # A call that ended nothing. The seal is still joined — this answer may not say the
            # task is over before the row that says so is durable, and a seal that failed needs
            # every joiner to hand the claim back and report the stop — but the row that join
            # hands back is not this call's to be told about (see `_tombstone_answer`).
            await self._seal_redacted(live)
            return ToolResult(content=self._tombstone_answer())
        # Only the call that actually ended the task may say how it ended. Every call that
        # arrives after the episode has ended is answered with a tombstone, and a tombstone
        # is `terminated` too — so a `terminate` racing an accepted `submit` returns first
        # (the submission is still in its finalizer), and taking the ending from whichever
        # response reaches the seal first would file the agent's earned, scored submission
        # as a task it aborted. Stamped synchronously, the moment the call returns, so
        # whoever claims the seal already sees it. The payload is not taken from here at
        # all: it is the core's to stamp, and `_record` reads it off the episode.
        live.terminal_tool = native
        return ToolResult(content=self._terminal_answer(await self._seal_redacted(live)))

    # ----- routing -----

    async def _resolve(self, tool: str, lease: Optional[Any]) -> Any:
        """Find the episode a call belongs to, or refuse it.

        A lease alone is not identity. At capacity above 1 every env tool is exposed at once, so
        a worker holding a valid lease for one task could name a tool belonging to another env or
        to no task at all. The registry therefore binds ``(lease, env, native tool)`` and checks
        all three **before** the call can reach the episode, where an unknown tool would consume
        a step of the budget and land in the trajectory. Returns the episode and the *native*
        tool name to call it with, or the refusal to return instead.

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
            route = self._routes.get(tool)
            if route is None:
                return _stream_error(
                    "tool_not_in_task", f"tool {tool!r} is not advertised by this task"
                )
            env_name, native = route
            if env_name != live.ref.env:
                # A lease that is valid but denotes the wrong thing — the same failure shape
                # that disqualified lease-scoped tool names. Two envs can advertise the same
                # native name with different meanings, so routing on the lease alone could seal
                # and score the wrong task.
                return _stream_error(
                    "wrong_env",
                    f"tool {tool!r} belongs to env {env_name!r}; this lease names a "
                    f"{live.ref.env!r} task",
                )
            return live, native

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

    async def _seal_redacted(self, live: _Live) -> Optional[ResultRow]:
        """Seal a task whose terminating call is being answered, and hand back the row that seal
        recorded — ``None`` when it recorded none.

        The row :meth:`_seal` records is for the harness, not the caller — and so is the
        exception it raises instead. Every failure it can raise is already recorded on the stream
        before it leaves: whatever row did land is in ``results.jsonl``, :attr:`stopped` is set,
        the next dispense refuses, and :meth:`aclose` raises. Nothing is therefore lost by
        answering the agent with the constant, and raising instead would tell it precisely what
        this call is not allowed to tell it. Cancellation is a ``BaseException`` and still passes
        through: the caller it would have answered is already gone, and the claim hand-back needs
        it to.

        **What is handed back is the row, not the outcome of the seal.** A seal can record a row
        and still raise — an unheadlinable summary lands the row first and stops the stream after
        the release (see :meth:`_run_seal`) — and on that path the task's feedback is durable and
        is the same feedback the file holds. So the failure path answers with
        :attr:`_Live.row`, which is that recorded row or ``None`` if the seal never got one
        written. It is read off the entry rather than reconstructed: one seal, one row, and the
        entry is where that row was published.

        The fallback ``_stop`` is belt and braces for a future edit that raises without recording
        one — the first cause wins, so it is a no-op on every path that exists today."""
        try:
            return await self._seal(live)
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
            return live.row

    def _terminal_answer(self, row: Optional[ResultRow]) -> str:
        """The whole response to a call that ended a task: the fixed envelope, plus whatever the
        feedback policy reveals about the row that seal recorded.

        Under :class:`Never` — the default — this is the constant and nothing is computed at all,
        so a run with no verdict channel is byte-for-byte the run this module served before
        policies existed.

        Under a revealing policy the envelope gains exactly one member, ``feedback``, and it is
        **always present**: an env that published nothing at episode level, a policy holding this
        task back, and a seal that recorded no row all answer with an empty list. That is the
        whole of what keeps the channel to the one thing it is for. A member that appeared only
        when there was something to say would let an agent read "the stream failed to record my
        task" off the shape of a response — the stop channel this module closes everywhere else,
        reopened at the one call that always knows.

        What the policy is handed is a private copy of the sealed row's episode-level items, and
        what it hands back goes inside the envelope rather than beside it. It cannot add a
        member, cannot reach the lease, the position, the index, the closure or the queue, and is
        never asked about a task other than the one this call ended — the row comes from *this*
        seal, never from the results list, which under concurrency would be a sibling's.

        **What runs is the module's function for the admitted type, never a method looked up on
        the caller's object** (see :data:`_POLICIES`). ``self._feedback.reveal(...)`` would find an
        instance-dictionary entry first, so an admitted ``Immediate`` could answer the agent with
        items no env ever published while the row beside it, the dispense before it and the resume
        after it all stamped ``immediate`` — a durable record with no evidence of the signal the
        agent was actually shown, and a later run whose scores were earned under it. The regime
        was already taken from the table; this takes the behaviour the regime is a *name for* from
        the same place, so that ``immediate`` means the sealed row's items verbatim rather than
        "some member called ``feedback`` was open".

        **A policy that cannot answer is a run-level failure, not a task-level one.** No policy
        this module admits today can fail here — :class:`Immediate` hands back items the record
        already encoded with this same encoder and setting — so what follows is the containment a
        policy added later inherits rather than a path a run reaches now. A policy that raises
        would raise on every task, so the row already stamped with its regime is a row claiming a
        channel the agent was never told through: the record would say a training run happened
        where none did. So the failure is contained (this answer may not become a traceback at the
        agent, and the shape may not change), the agent is told the task ended and nothing more,
        and the stream stops — loud to the harness, silent to the agent, like every other
        integrity failure here.

        Serialised here rather than by the caller, with the encoder and the ``allow_nan`` the
        record itself is committed with: a value the endpoint could not send has to be found
        inside the containment, because outside it the failure is a traceback at the agent in
        place of the answer every other ending returns."""
        if not self._reveals:
            return _TASK_OVER
        try:
            revealed = self._reveal(self._feedback, _revealable(row))
            return json.dumps(
                {**_TASK_OVER_FIELDS, _FEEDBACK_MEMBER: [dict(item) for item in revealed]},
                allow_nan=False,
            )
        except BaseException as exc:  # noqa: BLE001 — the policy's failure, never this answer's
            # Nothing here awaits, so no cancellation can be delivered into it and one observed
            # was raised where it was observed (see :func:`_must_propagate`).
            if _must_propagate(exc, None):
                raise
            rendered = _rendered_failure(exc)
            self._stop(
                exc,
                dispensing=(
                    f"this stream stopped: the {self._regime!r} feedback policy could not answer "
                    f"a terminating call ({rendered}), so a task recorded under that regime was "
                    "served with the channel its row claims closed"
                ),
                closing=(
                    f"this stream stopped before its queue was served: the {self._regime!r} "
                    f"feedback policy could not answer a terminating call ({rendered})"
                ),
            )
            return _TASK_OVER_SILENT

    def _tombstone_answer(self) -> str:
        """The whole response to a call that reached an episode already over: the task ended, and
        nothing whatever about how.

        **The reveal belongs to the call that sealed the task.** A tombstone is what an episode
        answers every ``tools/call`` with once it has left ``OPEN``: nothing is dispatched,
        nothing is ended, and ``terminated`` reports the episode's state rather than anything this
        call did. Composed like a terminal, an ordinary ``noop`` racing an accepted ``submit`` is
        handed the verdict its sibling earned — a second recipient of a task's feedback, on a call
        that did not ask for it and did not end anything. Under :class:`Never` the answer is the
        same constant either way, which is why this survived until a revealing policy existed.

        **The member is still present, and empty — not absent.** Under a revealing policy the
        member is a property of the policy and never of the task (see :class:`FeedbackPolicy`), so
        an answer missing it would be a shape no policy chose, and a shape a revealing run cannot
        otherwise produce: its absence would then mean exactly one thing, "you were not the call
        that ended this", readable off the envelope by a caller that shares a stream with the one
        that was. :data:`_TASK_OVER_SILENT` already means four things — an env that published
        nothing, a policy holding this task back, a seal that recorded no row, a policy that could
        not answer — and this is the fifth, so the response space gains no value and nothing new
        is readable off the one it reuses. It also keeps the promise a parser was written against:
        under a revealing run ``feedback`` is always there to read.

        **What the two policies show a tombstoned caller.** Under :class:`Never` this is
        :data:`_TASK_OVER`; under a revealing policy it is that same envelope with an empty
        member. The difference between them is the member, which is what the policy decides for
        every terminal answer of the run — so the pair says "this run reveals" and nothing else,
        which the same caller reads off any task it ends itself and which every row of the record
        is stamped with. What it may not say, and no longer does, is what the task scored.

        **The policy is not asked.** It answers one question — what does the call that ended this
        task reveal about it — and this call ended no task, so there is nothing to put to it. That
        matters for the policies this design anticipates rather than for the two that ship: a
        ``Delayed(k)`` or ``Batched(n)`` holds verdicts back and lets them go on a *later* ending,
        so a tombstone put through it would be an ending that never happened, spending a hold and
        releasing a batch to whoever happened to race a seal."""
        return _TASK_OVER_SILENT if self._reveals else _TASK_OVER

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
        and the retry starts at the write (see :meth:`_run_seal`). **Every hand-back carries a
        row**, including one from a seal that failed short of composing its own: that one leaves
        the row it had reached, or an unscored stand-in if it had reached none, rather than
        leaving the retry to compose a second (see :meth:`_retained_row`).

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

        **Two failures reach here and the stop says which.** One is the durable append refusing a
        row the seal had ready, which is a failure of the storage and leaves the record short an
        outcome until some retry lands it. The other is the seal failing above that append, which
        touched no storage at all and leaves its row on the entry for the retry to write (see
        :meth:`_retained_row`). They send an operator to different places, so they are not
        reported in the same words.

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
                # The seal *failed* rather than being deferred, so the queue stops: served on, the
                # rest of it is served over a record that may be missing an outcome the agent
                # earned. A merely cancelled seal never reaches here: that one is finished by
                # whoever drains next.
                composing = live.compose_error
                if composing is None:
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
                else:
                    # The other failure a hand-back can carry, and neither half of the message
                    # above is true of it: nothing was written, so nothing about the storage
                    # failed, and the row this seal owes is on the entry for the retry, which may
                    # well land it and leave a complete record behind a stopped run. Pointing an
                    # operator at `results.jsonl` names the one part of this that worked, and
                    # calling the record incomplete is a claim about a write that has not been
                    # attempted yet. What is true whichever way that write goes is that this
                    # task's seal did not finish and the queue stopped there.
                    rendered = _rendered_failure(composing)
                    self._stop(
                        composing,  # the failure the row's own diagnostic names
                        dispensing=(
                            "this stream stopped: a dispensed task's seal failed before its row "
                            f"was recorded ({rendered}), so no further task could be sealed "
                            "either"
                        ),
                        closing=(
                            "this stream stopped before its queue was served: a dispensed task's "
                            f"seal failed before its row was recorded ({rendered})"
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

        **A seal that failed short of a row is retried the same way, and for the same reasons.**
        The objections above are about re-composing, not about how the first attempt ended, so an
        entry may never hand its claim back with nothing on it: a failure in the composing half
        leaves a row and re-raises, and the retry writes *that* (see :meth:`_retained_row`). Which
        row depends on how far the seal got: its own, when the outcome was already read and only
        the extensions were left, and an unscored stand-in when nothing was.

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
                row = await self._retained_row(live, forced)
            async with self._lock:
                # Durable before the row counts anywhere else. `reconcile` reads a missing result
                # as a crash, so a row that only reached the page cache would turn a sealed,
                # scored task into a `broker_abort` after a host crash — an outcome the agent
                # earned, reported as an infrastructure failure. Everything in here is
                # synchronous, so no cancellation point can split the write from the claim it
                # makes.
                #
                # Owned, exactly as the dispense is: a task that was in flight when a `resume=True`
                # took the directory over is a task whose *position* the taking-over stream
                # replays, so its row would be a second scored outcome for one queue position —
                # written by a stream that had already lost the directory, with both rows honestly
                # stamped and nothing saying which run each belongs to. The seal fails here
                # instead, and a failed seal stops the stream (see :meth:`_join_seal`).
                self._append_owned(self.results_path, row.to_wire())
                # What the run keeps is the row *the file now holds*, re-read from its own wire
                # form. That is the canonical snapshot every reader is shown a copy of (see
                # :attr:`results`), and taking it here is what makes those copies cheap and
                # certain: they copy plain data, run no env code, and cannot disagree with the
                # record. Held any other way this list would carry the env's own objects — the
                # feedback values it published, and the one list that `observed` and
                # `score.feedback` both are — and every view of it would be a handle on them.
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
            # :meth:`_compose_row`), and the source recorded beside the failure is what says
            # which this is — never the exception object, which an env may raise on both
            # boundaries. What a lost call owed the record is the row it already produced:
            # unscored, saying the call was lost, which is the whole property — an outcome
            # nothing the agent did produced cannot be averaged into a benchmark. Stopping on top
            # of that spends the rest of the queue on one lost call, and the failure it names is
            # one the *next* task need not have: a mid-episode call is where a transient fault
            # lands, and a session that hiccups once would end a 480-task run. A terminal that
            # failed is the opposite case and still stops here — there the env is on its way out
            # of a task it had already ended, and every task of that env leaves the same way.
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

    async def _retained_row(self, live: _Live, forced: Optional[Closure]) -> ResultRow:
        """Compose this task's row, and leave a row on the entry **either way**.

        The hand-back in :meth:`_join_seal` exists so a later drain can retry the append, and what
        makes that retry safe is the retained row: composing is the half that may not run twice
        (see :meth:`_run_seal`). A seal that failed *before* it had a row therefore cannot simply
        hand its claim back — a retry finding no row composes a second one, and both halves of
        that are the very things the retention exists to prevent. Measured: every span's
        ``finalize`` runs again, and the classification is re-read from an episode the first
        attempt already force-terminated, so a task the stream **drained** is recorded under a
        *scored* closure the agent never earned.

        So a failure here composes nothing further and stands a row in instead. It is unscored and
        says why (see :meth:`_unsealed_row`), which is the honest reading: the seal did not finish,
        so no verdict stands behind this task — the same answer a terminal the stream drove and the
        env failed on already gets, reached one step earlier. The failure is then re-raised
        unchanged, so the claim still goes back, the stream still stops, and a later drain still
        retries the *append*, of this row and never of the composition.

        **A failure that had reached an outcome first keeps it.** The composition decides the
        closure and the score before it runs the extensions, so a failure out of *those* leaves a
        seal that knows exactly how the task ended; that one composes its real row and leaves it
        here (see :meth:`_compose_row`), and the stand-in below is for a failure that reached no
        outcome at all. Standing an unscored row in over both would answer a task the agent
        solved with an infrastructure failure."""
        try:
            live.pending_row = await self._compose_row(live, forced)
        except BaseException as exc:  # noqa: BLE001 — a row stands in below, and this re-raises
            # Recorded so the stop this owes is the one it is: what failed is the composing half,
            # and storage is the append's business (see :meth:`_join_seal`).
            live.compose_error = exc
            if live.pending_row is None:
                live.pending_row = self._unsealed_row(live, forced, exc)
            raise
        return live.pending_row

    def _unsealed_row(
        self, live: _Live, forced: Optional[Closure], cause: BaseException
    ) -> ResultRow:
        """The row a seal that failed before composing one leaves for the retry to write.

        Built from the entry alone — the position, the lease and the queue reference the dispense
        already recorded — because everything else a row carries is read off the episode, and
        reading the episode is what just failed. So this cannot fail in turn: the one value it
        takes from outside is the failure's own description, and that is rendered through the
        guard every other message about a caught failure uses (see :func:`_rendered_failure`).

        Unscored, and ``finalize_error`` rather than whatever the seal was forcing: a closure is
        a claim about how the task ended, and a seal that produced no row is a task whose ending
        this record cannot vouch for. The reason the stream reached for the seal — a drain, a
        deadline — is not lost; it is simply not a verdict, and the diagnostic carries it.

        ``observed`` is empty, which is the true statement about *this row*: nothing the env
        published reached it. The spans are recorded the way any failed seal records them (see
        :meth:`_unclosed_spans`), and a failure this early has closed none of them, so what that
        says here is that none was ever finalized."""
        rendered = _rendered_failure(cause)
        # What the seal was reaching for when it failed. Not a closure — this row's closure says
        # the seal produced none — but the one fact about the ending that is still knowable here,
        # and the reader of a `finalize_error` row wants to know which of them it was.
        sealing_for = "the agent's own terminal" if forced is None else repr(forced)
        return ResultRow(
            seq=live.seq,
            lease=live.lease,
            position=live.position,
            env=live.ref.env,
            task_idx=live.ref.task_idx,
            closure="finalize_error",
            score=None,
            observed=[],
            diagnostic=(
                "the seal failed before it composed a row, so this task has no verdict behind "
                f"it ({rendered}); it was sealing for {sealing_for}"
            ),
            extensions=self._unclosed_spans(live, cause),
            # The regime this stream serves, exactly as a composed row records it.
            feedback_regime=self._regime,
        )

    def _unclosed_spans(self, live: _Live, cause: BaseException) -> Dict[str, Any]:
        """The span entries for a row a seal that could not finish leaves behind: what each span
        that closed actually recorded, and, for the rest, that nothing of theirs is in this row.

        A span that closed keeps its own entry. It ran exactly once, its side effects are out in
        the world and what it returned is provenance the seal already collected, so rewriting it
        would drop that payload and file the namespace under a failure the extension never raised.
        A namespace this seal never reached at all would be recorded, by that same rewrite, as one
        whose ``finalize`` failed. Neither is true, and the row is the only account these
        extensions get.

        The member is still ``error``, for both. An orderly row's entry is a ``dispensed`` with
        exactly one of ``sealed`` or ``error`` beside it, and that shape is what tells it from a
        reconciled one (see :class:`ResultRow`); a third member would widen a wire contract for
        every consumer, on the rarest path this module has, to say what the entry's own text says
        and what ``closure`` says about the whole row. So ``error`` carries the truth that a span
        was never asked, in its own words."""
        unreached = (
            "the seal failed before this span was finalized, so its finalize was never called: "
            f"{_rendered_failure(cause)}"
        )
        entries: Dict[str, Any] = {}
        for namespace in live.spans:
            entry = live.finalized_extensions.get(namespace)
            if entry is None:
                entry = {
                    "dispensed": live.dispensed_extensions.get(namespace),
                    "error": unreached,
                }
            entries[namespace] = entry
        return entries

    async def _compose_row(self, live: _Live, forced: Optional[Closure]) -> ResultRow:
        """Everything a seal does exactly once: end the episode, read its verdict, classify it,
        close the spans, and build the row from all of it. Nothing here is retried, so a failure
        in the last of those steps still builds the row from the ones before it, and leaves it on
        the entry for the retry that may not run them again."""
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
                # and letting the reassignment drop the refusal records the abort's fail-closed
                # `correct=False` as an outcome — for a queue in which every task of that env
                # will refuse the same way.
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
            # the source recorded beside it is what tells the two apart (see `_run_seal`). An
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
        # **Reading the payload is the env's data too**, and it is contained for the same reason
        # every other read of an env's own values here is. The core stamps its flag onto a copy
        # of the verdict *the env returned*, so both the stamping and the lookup below compare
        # this module's key against keys the env owns — and an env's `__eq__` is env code, which
        # may raise. Uncontained it takes the whole row with it: the closure is classified from
        # this payload, so nothing is composed, the seal fails, and a task the agent earned and
        # a sealed episode graded is answered by `reconcile` as a broker crash — the harness
        # blamed for a failure that is the env's. A payload this record cannot read is a task
        # with no verdict behind it, which is exactly what a failed terminal already means here,
        # so it is recorded as one: the row lands unscored with a diagnostic, and the stream
        # stops, because an env that does this does it for every task of its own in the queue.
        #
        # Nothing here awaits, so no cancellation can be delivered into it and one observed was
        # raised where it was observed (see :func:`_must_propagate`).
        try:
            payload = episode.terminal_payload
            # Re-keyed through `str` once, here, so the classification below compares strings
            # this module made rather than objects the env did — the read is contained, and the
            # value it yields carries nothing that could raise from a later lookup.
            live.terminal_payload = (
                None if payload is None else {str(key): value for key, value in payload.items()}
            )
            # Read in the same contained block, though its contents are the core's own strings
            # rather than the env's values: it comes off the same object, and a read of that
            # object that cannot be trusted is not one this row should trust for either field.
            live.terminal_failure = episode.terminal_failure
        except BaseException as exc:  # noqa: BLE001 — the env's failure, never this row's
            if _must_propagate(exc, None):
                raise
            live.terminal_payload = None
            live.terminal_failure = None
            live.failed_to_end(exc, "terminal")

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
        #
        # The *whole* boundary is contained, not only the malformed summary it is named for.
        # Picking a headline runs the env's own code: the funnel matches published names, and a
        # name is an object the env supplied, so the comparison that finds an item is that
        # object's `__eq__`. Anything else out of here escapes the seal entirely — no row at all,
        # and `reconcile` answers the dispense with a broker crash, so a task the agent solved is
        # filed as an evaluator that fell over. A summary this record cannot read is not a
        # different finding from one it cannot honour, and it gets the same answer.
        score: Optional[Score] = None
        if closure in _SCORED_CLOSURES:
            unheadlinable: Optional[_MalformedSummary] = None
            try:
                score = Score(
                    reward=_pick_float(observed, _REWARD_NAMES),
                    success=_pick_bool(observed, _SUCCESS_NAMES),
                    feedback=observed,
                )
            except _MalformedSummary as exc:
                unheadlinable = exc
            except BaseException as exc:  # noqa: BLE001 — the env's failure, never this row's
                # Nothing in the funnel awaits, so no cancellation can be delivered into it and
                # one observed here was raised where it was observed (see `_must_propagate`).
                if _must_propagate(exc, None):
                    raise
                unheadlinable = _MalformedSummary(
                    f"reading it raised {_rendered_failure(exc)}"
                )
            if unheadlinable is not None:
                live.summary_error = unheadlinable
                diagnostic = (
                    "the env published a summary value this record cannot headline: "
                    f"{unheadlinable}"
                )
        # Extensions run here — after the outcome is decided, before the row is written, and
        # outside every lock. Whatever they return is namespaced; whatever they do wrong is
        # recorded in their own namespace and cannot stop the row. What they are *handed* is
        # built per extension and detached from `score`, which is the row's own summary object.
        #
        # **A failure they let out does not take the outcome with it either.** The one shape that
        # reaches here is a failure no containment holds (a non-`Exception` `BaseException`, or
        # this seal's own cancellation), and it arrives with the episode sealed, its verdict read
        # and its closure classified. An extension may not change a task's outcome, which is the
        # whole reason they run after the classification, and a failure the stream cannot contain
        # is not the extension's licence to: standing an unscored row in over the top of it files
        # a task the agent solved as an infrastructure failure, for a verdict this seal was
        # holding as it failed. So the row is composed from what is already in hand and from the
        # spans that did close, and it is the *failure* that goes on out, to stop the stream and
        # hand the claim back as before.
        unfinished: Optional[BaseException] = None
        try:
            extensions = await self._finalize_spans(live, closure, score)
        except BaseException as exc:  # noqa: BLE001 (the row is composed below, then re-raised)
            unfinished = exc
            extensions = self._unclosed_spans(live, exc)
        row = ResultRow(
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
            # The regime this stream serves, validated at construction and never re-read from the
            # policy object — the row says what the run was, not what the policy happens to
            # answer at the moment the row is built.
            feedback_regime=self._regime,
        )
        if unfinished is not None:
            # The seal's answer is already reached, so what is left is the failure. The row goes
            # on the entry for the retry to write, exactly as one whose append failed is retained
            # (see :meth:`_retained_row`), and the failure is re-raised unchanged: the claim goes
            # back and the stream stops, as it does for any seal that could not finish.
            live.pending_row = row
            raise unfinished
        return row

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

        Only a raise that left the episode still OPEN is handed back, so the caller can fall back
        to the reserved abort rather than leaving a task nothing ended. What the fallback does
        *not* do is settle it: an env whose score terminal raises has published no verdict for
        this task and will publish none for the next, so the refusal outlives the abort that
        answered it and the row is classified from it either way. The refusal this fallback is
        for — a score terminal whose arguments the stream cannot invent — never arrives here at
        all: missing arguments are refused against the advertised schema and come back as an
        ordinary validation result, not as a raise.

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
        durable dispense is never answered, and an orderly shutdown reconciles as a crash.

        **The spans close one at a time, and what each one recorded is kept as it goes.** A
        failure this may not contain still leaves a row owed: the seal fails, its claim goes back,
        and a later drain writes what the seal had reached (see :meth:`_compose_row`). So the
        entries are accumulated on the entry rather than in a local map the raise would take with
        it. Recomposing them from the open spans instead answers for every namespace with the one
        failure that escaped: a span that closed loses the payload it returned and is filed under
        a failure another extension raised, and a span this never reached is recorded as one whose
        ``finalize`` failed."""
        extensions = live.finalized_extensions
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
                    # Out of here the failure is the seal's rather than this span's, and it is
                    # recorded as one: this span did not return, which is a different statement
                    # from a `finalize` that failed on its own and is worded as one, because the
                    # failure may be this seal's cancellation arriving inside the callback. The
                    # entry is still kept, because dropping it would leave the row owing an
                    # account of a span it opened, and the row is where an extension's half of
                    # the task is answered for.
                    entry["error"] = (
                        "the seal failed while this span was finalizing, so nothing it returned "
                        f"was recorded: {_rendered_failure(exc)}"
                    )
                    extensions[namespace] = entry
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
                "the terminal transaction failed closed; the env published no verdict"
                + _described_failure(live.terminal_failure),
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
        it — and consulting it means reading an :data:`~shogym.types.EpisodeFeedbackValue`, which
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
        suspension point between the record landing and the bookkeeping that answers for it.

        The ownership check is not above the append, it is *around* it: this is the moment a task
        becomes part of this directory's record, and a stream that has lost the directory must add
        nothing to it — which a check that merely ran first would not deliver, since a takeover
        can land between the check and the write (see :meth:`_append_owned`). It raises where a
        failed append raises, so it is answered the same way: the dispense is refused with nothing
        handed out, and the stream stops rather than serving the rest of its queue into a record
        that is no longer its own (see :meth:`_require_claim`)."""
        self._append_owned(
            self.dispenses_path,
            {
                "lease": live.lease,
                "seq": live.seq,
                "position": live.position,
                "env": live.ref.env,
                "task_idx": live.ref.task_idx,
                "dispensed_at": time.time(),
                # Stamped HERE, before the task is handed out, so the regime is durable before
                # anything could have been revealed under it — and so the one row a crash leaves
                # this run to write, the `broker_abort` :func:`reconcile` builds, can say which
                # posture it was dispensed under instead of defaulting to the safe-looking one.
                "feedback_regime": self._regime,
                # namespace -> what that extension observed at dispense. The `{"dispensed": ...}`
                # wrapper the row carries is deliberately *not* here: it exists to sit beside
                # `sealed`/`error`, which are outcomes, and a dispense record has no outcome.
                # `reconcile` puts it on, so exactly one place knows the row's shape.
                "extensions": live.dispensed_extensions,
            },
        )

    def _require_open(self) -> None:
        if self._stopped is not None:
            raise RuntimeError(self._stopped.dispensing) from self._stopped.cause
        if self._closed:
            raise RuntimeError("this stream is closed")


# What :class:`EvalStream` was passed for an argument it refuses. A sentinel rather than a
# default of `None`, because the refusal is about the argument being *supplied at all*: `None` is
# a value a caller can pass, and one that would then be waved through as "not really a policy".
_REFUSED: Any = object()


class EvalStream(TaskStream):
    """A :class:`TaskStream` whose evaluation posture is a construction rather than a
    configuration: it pins ``feedback=Never()`` and refuses the argument outright.

    The difference from ``TaskStream(..., feedback=Never())`` is not what the run does — the two
    serve identically, and their rows are identical evidence — it is what a *reader of the code*
    can conclude. One says this run had no verdict channel; the other says no argument at this
    construction site could have opened one. A project whose self-improvement claims rest on its
    evaluation credibility cannot have that credibility be a value someone can edit, so the name
    is the guarantee and the guarantee has to be enumerated.

    **Guaranteed, and by what mechanism.** Each of these is enforced here or inherited from a
    mechanism named below it; nothing on this list is a convention.

    1. *No verdict channel, and no way to ask for one.* ``feedback`` is pinned to
       :class:`Never`, so a terminating call answers with the fixed payload — the same bytes for
       every env, task and outcome. Mechanism: this constructor passes ``Never()`` to
       :class:`TaskStream` and raises on any ``feedback`` argument, :class:`Never` included (see
       below). There is no setter for :attr:`TaskStream.feedback`, and the regime and the decision
       to reveal are read once at construction and kept, so nothing is re-read mid-run.

    2. *Every record this stream writes says so.* ``feedback_regime="never"`` is stamped on each
       dispense record, before the task is handed out, and on each result row — and on the
       ``broker_abort`` row :func:`reconcile` builds from an abandoned dispense. Mechanism: the
       stamp is taken from the validated regime of the pinned policy, at
       :meth:`TaskStream._write_dispense` and :meth:`TaskStream._compose_row`. So
       ``row.get("feedback_regime", "never") == "never"`` is the whole of the reader's check, on
       the artifact, with no join against anything.

    3. *One record never mixes postures.* A stream that is not resuming refuses a provenance
       directory that already holds any record; a resuming one refuses a directory whose
       dispenses, rows, or ownership claim name a different regime; and every stream takes
       exclusive ownership of its directory before it builds anything, so two streams pointed at
       the same empty one cannot both serve into it. Mechanism:
       :meth:`TaskStream._require_fresh_provenance`, :meth:`TaskStream._require_regime_matches`
       and :meth:`TaskStream._claim_provenance`, all at construction, before anything is spent —
       the first of them *inside* the exclusion the last one claims in, since a check the claim
       does not hold its exclusion across says nothing about the record that claim is installed
       over — and with :meth:`TaskStream._append_owned` re-verifying the ownership inside an
       exclusive lock on the directory at every dispense *and* every row, since a claim taken once
       says nothing about the writes that follow it and a claim merely re-read says nothing about
       the write it precedes.

       *Two streams* is also the count that a second **object** would break, and ownership alone
       cannot tell a duplicate apart: an evaluation stream copied with ``copy.copy`` or revived
       from a pickle carries the same token under the same pid, so it passes every check above and
       files a second scored row for a queue position the original is serving. Mechanism:
       :meth:`TaskStream._refuse_duplication`, which refuses ``copy``, ``deepcopy``, pickling and
       state extraction on this class as on its base, so no second object exists to check; and
       :meth:`TaskStream._holds_claim`, which asks *which object* alongside which stream and which
       process, so the append is defended even against a duplicate built without them.

    4. *Everything a stream already guarantees*, unchanged and named here because an evaluation
       rests on them: the framing has no field a task index or target could be written into; the
       stream seals and scores every episode itself, never the agent; a dispense is durable before
       the task is exposed, so a crash is reconcilable rather than silent; and an outcome the
       agent did not earn lands unscored rather than as a zero. Mechanism: :class:`DispensedTask`,
       :meth:`TaskStream._seal`, :meth:`TaskStream._write_dispense`, :data:`_SCORED_CLOSURES`.

    **Not guaranteed, and deliberately not claimed.** These matter to an evaluation and none of
    them is decidable here, so this class does not pretend to decide them:

    - *That the queue is held out.* Which task indices belong to an evaluation pool is the
      caller's split; this object is handed a materialised queue and cannot know what a training
      run was shown.
    - *That the agent cannot read the record.* ``prov_dir`` carries task indices and raw feedback
      and belongs to the harness, but nothing here can prove it is off a filesystem the agent
      under test can read.
    - *That provenance extensions keep what they are shown.* An extension's ``finalize`` is handed
      the task's score (see :class:`CompletedTask`), which is the verdict; what it then does with
      it is outside this process's reach. Extensions are still permitted, because the snapshots a
      real held-out run needs are provenance extensions.
    - *That the env carries no state between tasks.* ``env_for`` is called once per dispensed task
      and the episode closes what it was given, but a factory returning fresh handles onto one
      shared backend satisfies that and shares everything. Identity-checking the returned envs
      would prove non-identity, which is not the property, so nothing here claims it.

    **Refused knobs.** Of :class:`TaskStream`'s arguments, exactly one decides posture, and it is
    refused. The others are not posture questions and are passed through: ``max_in_flight`` and
    ``deadline`` change how a queue is served, not what the agent is told; ``resume`` is the
    only correct way to continue a crashed evaluation, and guarantee 3 is what makes it safe;
    ``provenance`` records more about a run rather than revealing more of it.

    Args:
        Every :class:`TaskStream` argument except ``feedback``, with the same meanings.
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
        # Accepted only in order to be refused. Left out of the signature entirely, the refusal
        # would be Python's `unexpected keyword argument`, which says what happened and not one
        # word about why — and a caller reading that has every reason to reach for `TaskStream`
        # and pass the policy there, which is exactly the move this class exists to make visible.
        feedback: Any = _REFUSED,
    ) -> None:
        if feedback is not _REFUSED:
            # `Never()` is refused too, and that is the point rather than an oversight. A value
            # that has to be passed is a value the next edit can change, and a construction site
            # that reads `EvalStream(..., feedback=Never())` invites precisely that edit while
            # looking like it was reviewed. The class is the statement; there is nothing to pass.
            raise ValueError(
                f"{type(self).__name__} takes no `feedback` policy, and refuses "
                f"{feedback!r} for that reason rather than for what it is: an evaluation's "
                "verdict channel is closed by construction, so there is no argument that could "
                "open it and none that could confirm it is shut. Use "
                f"`{TaskStream.__name__}(..., feedback=...)` for a run whose scores are not "
                "meant to be evaluation-grade"
            )
        super().__init__(
            env_for,
            tasks,
            prov_dir=prov_dir,
            max_in_flight=max_in_flight,
            deadline=deadline,
            resume=resume,
            provenance=provenance,
            provenance_timeout=provenance_timeout,
            feedback=Never(),
        )


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
    it, the seal fails before it has classified anything, and what the run records for the task is
    the unscored row a seal that reached no outcome stands in (see
    :meth:`TaskStream._retained_row`). One that blocks would wedge the seal where no extension
    bound applies.
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

    Nor can it refuse a summary the row can carry. ``shogym.feedback.wire`` validates every item on
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

    Called **after** the append, which is what makes it safe to run at all: those exact values
    have just been serialised, so nothing here can fail that the write did not already fail. The
    same call a step earlier would be a normalization that could suppress a row the run had
    otherwise earned — the failure mode :func:`_detached_summary` refuses ``deepcopy`` for."""
    return ResultRow.from_wire(json.loads(json.dumps(row.to_wire(), allow_nan=False)))


def _detached_row(row: ResultRow) -> ResultRow:
    """One recorded row, copied whole, for a reader that must not be able to reach the run's.

    :class:`ResultRow` is frozen and shallow: ``observed`` is a list of dicts and ``extensions``
    is a dict, so a reader handed the row itself can edit what the run reports without touching
    what the file says — and both public accessors would show the edit, since they are two views
    of one list. A run's record is not a thing reading it may rewrite.

    ``deepcopy`` here, for the reason :func:`_detached_manifest` gives and not despite the one
    :func:`_detached_summary` gives: what is behind this is already plain data (see
    :func:`_recorded_row`), so copying it copies data and runs no code an env wrote. That is what
    makes the read total — a reader gets the row whatever the env published in it."""
    return copy.deepcopy(row)


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


def _unregistrable(name: str) -> Optional[str]:
    """Why ``name`` cannot honestly be published as a tool name, or ``None`` if nothing is.

    The protocol bounds what a tool name may be — one to 128 characters of letters, digits,
    ``_``, ``-`` and ``.`` — and the bound is not enforced where a violation is created. FastMCP
    logs a warning and registers the tool anyway, so an endpoint built from a name outside the
    set looks healthy from inside this process and is refused somewhere the harness cannot see:
    a strict client, or a provider that re-publishes the tool list to a model. That is why this
    is a refusal at construction rather than a warning, and why the string it is asked about is
    the *joined* one that actually goes on the wire rather than either half of it.

    It says why rather than yes/no because the caller who has to fix it needs to be told what
    is wrong with which name; the messages at the two call sites differ only in whose name it
    was."""
    if not name:
        return "it is empty"
    illegal = sorted({char for char in name if not _TOOL_NAME_CHAR.fullmatch(char)})
    if illegal:
        return (
            f"it contains {', '.join(repr(char) for char in illegal)}, and a tool name may hold "
            "only letters, digits, '_', '-' and '.'"
        )
    if len(name) > _TOOL_NAME_MAX:
        return (
            f"it is {len(name)} characters long, over the {_TOOL_NAME_MAX} a tool name may be"
        )
    return None


def _naming_note(renamed: Sequence[Tuple[str, str]]) -> str:
    """The stream's own sentence about what a task's tools are called here.

    An env's ``instructions`` are its author's prose, and they name the env's tools by the env's
    own names — "call ``submit`` with your final ``answer``" — because that is what an env author
    was ever promised. Prefixing makes those names uncallable, and a framing that says ``submit``
    beside a tool list that says ``answers__submit`` gives a literal instruction-follower two
    incompatible commands, one of which the endpoint will refuse before this module is even
    reached.

    So the mapping is said *beside* the instructions rather than spliced into them. Rewriting the
    env's prose is the alternative and it is worse in both directions: it makes the stream edit
    text it did not write, on a guess about which words in free text are tool names, and a wrong
    guess corrupts the task itself. Here the env's prose is untouched and the correction is
    plainly the stream's.

    Every name on the right-hand side comes from the same map :meth:`TaskStream._resolve` routes
    on, so this can never point the agent at a name the endpoint does not register."""
    mapping = ", ".join(f"`{native}` is called as `{public}`" for native, public in renamed)
    return (
        "This endpoint serves several envs and registers each tool name once, so this task's "
        f"tools are registered under its env key: {mapping}. The instructions use the env's own "
        "names; the names to call are the ones listed here, which are the names in `tools`."
    )


def _detached_manifest(manifest: ToolManifest) -> ToolManifest:
    """One advertised tool on a copy of its own schema, for a reader that must not be able to
    reach the stream's.

    The frozen contract is only frozen if reading it cannot rewrite it, and a ``ToolManifest`` is
    a mutable model whose ``input_schema`` is a mutable dict — ``model_copy`` copies the model
    and keeps the dict. Shared, the same object is what a task's framing advertises, what a
    server registered, and what a new episode's manifest is confirmed against: a reader that
    edited it would change the contract the agent is shown *and* the one the drift check
    compares against, so the two would still agree and nothing would notice. The snapshot behind
    this is plain JSON (see the constructor), so copying it copies data and runs no env code."""
    return manifest.model_copy(
        update={"input_schema": copy.deepcopy(manifest.input_schema)}
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


def _close_on_owning_loop(
    catalog: Sequence[Tuple[str, Env]],
) -> Tuple[bool, List[Tuple[str, Optional[BaseException]]]]:
    """Close every env in ``catalog`` from sync code **without moving any of them to a loop that
    does not own them**. Reports whether the closes were scheduled rather than completed, and
    what each one raised.

    ``Env.close`` is a coroutine and the contract says nothing about loop affinity, while the
    factory that built these envs is explicitly allowed to provision resources — so an env built
    inside a running loop may hold objects belonging to *that* loop. Running its close on a
    private worker loop is therefore not a safe generalisation: at best it raises (a future
    attached to a different loop) and at worst it deadlocks, because the sync constructor
    waiting on the worker's result is blocking the very loop that close is waiting on.

    So the closes run where the envs were built. With no loop running there is nothing to
    conflict with and these are complete, synchronous closes, run **together** on one temporary
    loop: an env's close may block for as long as it likes, and started one at a time the first
    one to do so would leave every env behind it open with nothing else holding it. Inside a
    running loop a *synchronous* constructor cannot await one, so each close is scheduled on that
    loop — already independent, since scheduling waits for nothing — and completes as soon as the
    caller yields, after the error has propagated. That is the cost of validating in ``__init__``:
    the alternative is an async construction boundary, which is a different API. A caller that
    needs the closes to be finished before it sees the error can construct outside a loop; a
    caller inside one is told, on the error itself, that the cleanup is still in flight.

    Which of the two applies is a property of the calling frame, not of any env, so it is decided
    once here rather than per env."""
    try:
        loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        try:
            return False, asyncio.run(_close_together(catalog))
        except BaseException as exc:  # noqa: BLE001 — never mask the failure being raised
            # The batch shares one temporary loop, so a failure to run it at all is every env's
            # and is reported against each of them.
            return False, [(name, exc) for name, _ in catalog]
    outcomes: List[Tuple[str, Optional[BaseException]]] = []
    for name, env in catalog:
        try:
            task = loop.create_task(env.close())
        except BaseException as exc:  # noqa: BLE001 — reported to the caller, never raised here
            outcomes.append((name, exc))
            continue
        _pending_closes.add(task)
        task.add_done_callback(_pending_closes.discard)
        outcomes.append((name, None))
    return True, outcomes


async def _close_together(
    catalog: Sequence[Tuple[str, Env]],
) -> List[Tuple[str, Optional[BaseException]]]:
    """Start every catalog env's close, then wait for all of them, pairing each outcome with the
    env's name.

    Each close is contained in its own task, so what ``gather`` waits for cannot fail: one env's
    failure may not end the wait while the others are still running on a loop this call is about
    to close, which is the whole point of starting them together. Classifying what came back is
    the caller's — including a ``CancelledError``, which nothing here requests, so one that
    arrives was raised by an env's own close."""
    failures = await asyncio.gather(*(_close_contained(env) for _, env in catalog))
    return [(name, failure) for (name, _), failure in zip(catalog, failures)]


async def _close_contained(env: Env) -> Optional[BaseException]:
    """Close one env, handing back what it raised instead of raising it."""
    try:
        await env.close()
    except BaseException as exc:  # noqa: BLE001 — reported to the caller, never raised here
        return exc
    return None


def _revealable(row: Optional[ResultRow]) -> List[Dict[str, Any]]:
    """The most a terminating call could ever tell the agent about the task it ended: that row's
    **episode-level** feedback, in publication order, as its own copy.

    Episode level and no further, because that is the line
    :func:`~shogym.feedback.wire.select_inband` already draws for a single served episode — the
    terminal carries the values that scored the whole task, while inference-level items stay
    recorded-but-not-surfaced. A stream that revealed those would be surfacing per-step shaping
    a served episode withholds, at a boundary that is stricter rather than looser.

    Taken from the row, so what can be revealed is exactly what was recorded: the values in the
    file, read out of the run's own canonical copy of it (see :func:`_recorded_row`), which is
    plain data by the time it arrives here. Copied per item because it is handed to
    caller-supplied code — a policy may not be able to edit the record by editing what it was
    shown — and the copy is cheap for the same reason the read is safe.

    ``None`` is a seal that recorded no row at all, and answers with nothing rather than with
    anything reconstructed: there is no verdict to reveal for a task the record does not hold."""
    if row is None:
        return []
    return [dict(item) for item in row.observed if item.get("level") == _EPISODE_LEVEL]


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
    (:data:`shogym.types.EpisodeFeedbackValue`, and ``shogym.feedback.wire`` validates exactly that
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


def _get_task_description(max_in_flight: int) -> str:
    """The ``get_task`` description an agent reads, written for the capacity the stream was
    actually constructed with.

    The displacement rule is the agent's to know, and this text is the only place it can learn
    it: a pull made while every slot is full seals the oldest live task and scores it as an
    ordinary loss (see :meth:`TaskStream.get_task`), and that row is indistinguishable from a
    task the agent played and lost. A description written capacity-agnostically therefore
    charges the agent for a protocol it was never told, and leaves an earned-looking closure on
    the row: the exact failure the redactions elsewhere in this module exist to prevent. So the
    number of slots is named, and what a pull at it costs is stated outright.

    **Mechanics and consequences only, and in the indicative rather than the imperative.** What
    the tool does, what a call costs and what the numbers mean are facts about the endpoint;
    "end the task you hold first" and "work the task with the tools it lists" are advice about
    how to behave, and advice given here is a treatment applied to every agent this module serves
    and carried into every row. A run measuring how an agent chooses to spend a queue would then
    be measuring prose no harness chose, and one whose own instructions were written to leave
    that choice open would have the nudge reinstated underneath it, by the tool it has to call to
    get a task at all. So a consequence is stated ("calling ``get_task`` while a task is live
    forfeits that task") and the choice is left where it belongs.

    Built at registration, from the concrete stream, because the capacity cannot change
    afterwards: a description is advertised once and a caller may cache it, so one that named a
    number the run could move would be worse than one that named none.

    The one-slot wording is separate rather than a plural of the general case. At one slot there
    is no "oldest" to speak of and no free pull to distinguish from a costly one: every pull made
    while a task is live forfeits that task, which is a simpler and much sharper rule than the
    general one, and it is the capacity a stream is built with unless a caller asks otherwise."""
    if max_in_flight == 1:
        capacity = (
            "At most one task may be held at a time (``max_in_flight`` is 1). Calling "
            "``get_task`` while a task is live forfeits that task: the stream seals it and "
            "scores it as a loss. Ending a task frees its slot."
        )
    else:
        capacity = (
            f"Up to {max_in_flight} tasks may be held at once (``max_in_flight`` is "
            f"{max_in_flight}). Below that limit a pull is free and displaces nothing. Calling "
            f"``get_task`` while {max_in_flight} tasks are live forfeits the oldest of them: the "
            "stream seals that task and scores it as a loss. Ending a task frees its slot."
        )
    return (
        "Takes the next task off the queue and starts it.\n\n"
        "Returns the task framing (``{env, instructions, budget, tools}``, plus ``tool_naming`` "
        "when this endpoint serves several envs and renamed them) and never the task index or "
        "the target. A task can be completed with the tools it lists, by the names it lists them "
        "under; calls to those tools route to it automatically.\n\n"
        f"{capacity}\n\n"
        'Returns ``{"done": true}`` once the queue is empty. That answer is about the queue and '
        "nothing else: a pull the queue cannot answer displaces nothing, so a live task stays "
        "live, and ``in_flight`` says how many are live."
    )


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
    server: FastMCP = FastMCP(name=name or "shogym:tasks", mask_error_details=True)

    # The advertised text is passed rather than left to this function's docstring, because what
    # it has to say depends on the capacity this particular stream was built with (see
    # :func:`_get_task_description`) and a docstring is fixed at import.
    @server.tool(name=_GET_TASK_TOOL, description=_get_task_description(stream.max_in_flight))
    async def get_task() -> Dict[str, Any]:
        """Dispense the next task over MCP: the stream's own :meth:`TaskStream.get_task`, with an
        exhausted queue and a stopped stream answered alike."""
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

    @server.tool(
        name=_QUEUE_INFO_TOOL,
        # `in_flight` is the count the displacement rule is written in terms of, so the capacity
        # it is measured against is named here too: a caller polling this one is exactly the
        # caller deciding whether its next pull is free.
        description=(
            "Reports ``{remaining, consumed, in_flight}`` for the task queue. ``in_flight`` "
            f"counts the tasks currently live, against a limit of {stream.max_in_flight}."
        ),
    )
    async def queue_info() -> Dict[str, Any]:
        """Reports the queue's counts over MCP: the stream's own :meth:`TaskStream.queue_info`."""
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
