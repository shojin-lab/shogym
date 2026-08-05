"""Serve a *queue* of env tasks over one MCP endpoint.

:class:`~hgym.serve.episode.ServedEpisode` serves one episode. :class:`TaskStream` is the
next abstraction up: it holds a materialised queue of ``(env, task_idx)`` refs, hands out one
task at a time, routes the env's native tool calls to the episode that task is running in,
**seals and scores that episode itself** (never the agent), and appends one provenance row per
dispensed task.

The stream owns scoring, so the framing it returns is deliberately *redacted*: the caller sees
``{env, instructions, budget, tools}`` and never the task index or the target. That property is
structural — the value returned carries no field that could hold them.

The same rule holds for the *whole* agent-visible surface, not just the framing. A queue may
repeat a task index, so anything that identifies a task is a correlation channel: learn an index
once and a later occurrence can be recognised and replayed against a scorer. So the provenance
row a seal produces — its lease, queue position, index and the env's raw terminal feedback —
stays in ``results.jsonl`` and :attr:`TaskStream.results`, and a terminating call tells the
caller only that the task ended. For the same reason the server masks tool exceptions: an env
that raises while loading a task can name it in the exception text, and MCP would otherwise
relay that verbatim.

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

Recording: every dispensed task lands exactly one row, and the row is durable before it is
anything else — appended to ``results.jsonl`` and fsync'd *before* it is published on
:attr:`TaskStream.results`. A stream that cannot record a row has lost an outcome the agent
earned, so it stops rather than carrying on and handing back a file that looks complete.
Releasing the episode is separate: a seal that ran releases whether or not the row landed,
while a seal whose *caller* was cancelled before its row landed releases nothing and hands its
claim back — nothing was lost there, the episode still holds the outcome, and the next drain or
dispense records it. Once the row lands there is nothing left to hand back, so the rest of that
seal — the release, and the stop an unheadlinable summary owes — finishes whatever became of
the caller, rather than leaving an env nothing can reach or a stop nothing will report.

One provenance directory holds one run. A stream numbers from the start of its own queue and
appends, so pointed at a directory another run wrote it would file its rows under positions that
file already uses, with nothing on a row to say which run wrote it — a record that
double-counts a task the queue holds once, while :attr:`TaskStream.results` shows only the newer
run. Nothing here can tell that apart from a continuation meant on purpose, so a ``prov_dir``
that already holds rows is refused at construction, before anything is spent.

A row's ``feedback`` is the env's own output, verbatim and authoritative: the wire items
exactly as the episode published them — ``{name, value, level[, step]}``, ordered, every
occurrence — which is the form the JSONL trace and :class:`~hgym.evaluate.EvalResult` already
record feedback in. Keyed by name it would be a projection instead: the wire is a list and
permits a name more than once, so a mapping would decide by list order which occurrence
survives and drop the rest, and it would erase the level that says whether a value scored the
task or one step of it.

``reward`` and ``success`` are a *summary* of that record, read from the **episode-level**
items, and only ever repeat a value the env already published as a number and a bool
respectively. Nothing is coerced into them — ``bool("false")`` is ``True``, so coercion is how
a record comes to overstate a benchmark — and nothing wrong-typed is quietly dropped into
``None`` either, since that only trades the overstatement for an equally silent undercount. A
summary value that is of the wrong type, or published more than once, stops the stream after
the row carrying it is recorded. So in a run that finished, ``None`` means exactly one thing:
the env published no such field at episode level.

Drive it directly (``get_task`` / ``dispatch`` / ``queue_info``) or wrap it for an agent with
:func:`build_stream_server`. The direct API needs no MCP at all, which is what makes the
lifecycle testable.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from fastmcp import FastMCP
from fastmcp.tools import ToolResult

from hgym.core import Env
from hgym.serve.episode import ServedEpisode
from hgym.serve.server import build_tool
from hgym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from hgym.task import TaskSpec, ToolManifest

__all__ = ["TaskRef", "TaskStream", "build_stream_server"]

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
# kept verbatim under `feedback`; these two are the summary fields, and a missing one stays
# absent rather than becoming a fabricated zero. Read strictly — see `_pick_summary`.
_REWARD_NAMES = ("reward", "partial_credit")
_SUCCESS_NAMES = ("success", "correct")


class _MalformedSummary(ValueError):
    """An env published a summary-named feedback value this record cannot honestly headline."""


class TaskRef(NamedTuple):
    """One queue entry: which env, which task index. Repeats are legal — the queue is a
    sequence, and a task's identity within a run is its *position*, not its index."""

    env: str
    task_idx: int


@dataclass
class _Live:
    """A dispensed task: its episode and the bookkeeping needed to seal and record it.

    ``sealed`` is a *claim* on the seal, not proof one happened: it is taken before the first
    await and handed back if that attempt is abandoned. ``row`` is what makes a seal final —
    it is set only once the row is durable, so it is the flag every other reader tests.

    ``settling`` is the tail that a final row owes: letting go of the episode, and stopping the
    stream for a summary this record cannot headline. It is held as a task so it belongs to the
    stream rather than to whichever caller happened to start it — see :meth:`TaskStream._settled`
    for why that is not the caller's to abandon. It is set only where ``row`` already is, so an
    entry whose claim was handed back never has one running over it."""

    lease: str
    seq: int
    position: int
    ref: TaskRef
    episode: ServedEpisode
    sealed: bool = False
    row: Optional[Dict[str, Any]] = None
    summary_error: Optional[_MalformedSummary] = None
    settling: Optional["asyncio.Task[None]"] = None


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
    """Describe a failure an env supplied, without running its code unguarded.

    Every message this module builds about a failure it has *caught* formats that failure — and
    formatting an exception runs code belonging to whoever raised it, a second time and outside
    the ``except`` that just contained it. ``__str__`` is theirs, and an accident is enough: a
    message built lazily from state that is gone by the time it is asked for raises here rather
    than at the raise site. The second exception is not the one the handler caught, so it does
    not stay caught — it walks out of the handler carrying the handler's job with it. Measured,
    at the agent's own terminal: the redacted constant a terminating call answers with becomes a
    traceback, the stop is never published because its message is an *argument* to it, so the
    queue is served on against an env that fails every task, and ``aclose()`` reports a clean
    run having lost the row. A failure this module has already decided to contain may not be
    un-contained by the act of writing it down.

    So: the message is attempted, and on failure the type alone, and on failure a constant. What
    is never attempted twice is the caller's code — a fallback that formats the same object again
    would be the same bug one line down.

    ``CancelledError`` is caught here rather than let through. Nothing in this function awaits,
    so no cancellation can be *delivered* during it; one raised here was raised by the object
    being rendered, and letting that through would strand the seal exactly as a ``finalize`` that
    raises one does. ``SystemExit`` and ``KeyboardInterrupt`` still propagate, which is the line
    this module already holds for the env callbacks themselves: an interpreter-level signal costs
    the row loudly rather than being swallowed inside a diagnostic."""
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


class TaskStream:
    """Serve a queue of env tasks: one episode per dispensed task, sealed and scored by the
    stream, one provenance row each.

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
        prov_dir: where ``results.jsonl`` is appended. Its rows carry the task index and the
            env's raw feedback, so it belongs to the harness — keep it off any filesystem the
            agent under test can read. One directory per run: one that already holds rows is
            refused, because this stream has no way to continue another run's record.
        max_in_flight: how many episodes may be live at once.
    """

    def __init__(
        self,
        env_for: Callable[[str], Env],
        tasks: Sequence[TaskRef],
        *,
        prov_dir: Path,
        max_in_flight: int = 1,
    ) -> None:
        if max_in_flight != 1:
            raise ValueError(
                f"max_in_flight={max_in_flight} is not supported yet; this stream serves one "
                "episode at a time"
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
        self.prov_dir = Path(prov_dir)
        self.results_path = self.prov_dir / "results.jsonl"
        # Checked here, ahead of the catalog: it is a statement about the arguments, and refusing
        # before a factory is called costs no env the caller would then have to see closed.
        self._require_fresh_provenance()

        # One long-lived env per env name, used only to read the published contract (it never
        # begins a session, so closing it releases nothing an episode owns). Constructing it is
        # also what provisions an env whose data is fetched lazily.
        #
        # Everything from here to the end of construction runs under a cleanup guard: a factory
        # may provision real resources, and a constructor that raises hands back no object, so
        # nothing else could ever close what it built. That covers a partly-built catalog as
        # well as a failing validation.
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
            # is checked against before its task is dispensed.
            self._signature: Dict[str, Tuple[Any, ...]] = {
                name: _manifest_signature(tools) for name, tools in self._manifest.items()
            }
            self._score_terminal: Dict[str, Optional[str]] = {
                name: next(
                    (m.name for m in tools if m.terminal_kind == "score"),
                    None,
                )
                for name, tools in self._manifest.items()
            }
            # What the agent actually sees. This stream serves one env and joins nothing, so the
            # names on the wire are the env's own — and they are already this stream's plain data
            # (frozen native, above), which is what every registration and every task's framing
            # is built from.
            self._advertised: Dict[str, List[ToolManifest]] = {
                name: list(tools) for name, tools in self._manifest.items()
            }

            self._position = 0
            self._seq = 0
            self._consumed = 0
            self._live: Dict[str, _Live] = {}
            self._results: List[Dict[str, Any]] = []
            self._lock = asyncio.Lock()
            self._closed = False
            self._catalog_closed = False
            self._stopped: Optional[_Stopped] = None
        except BaseException as error:
            for note in self._close_catalog_now():
                error.add_note(note)
            raise

    # ----- construction-time validation -----

    def _require_fresh_provenance(self) -> None:
        """Refuse a provenance directory that another run already recorded into.

        A stream numbers from the start of its own queue — ``position`` from 0, ``seq`` from 1 —
        and :meth:`_append_row` appends. Pointed at a directory that already holds rows it would
        therefore file its own under keys that file already uses, with nothing on a row naming
        the run that wrote it, while :attr:`results` holds only this run's. Scoring the file then
        counts a task the queue holds once twice over, and scoring the object does not: two
        faithful readings of the same run that disagree, which is what makes this worth failing
        on rather than warning about.

        Continuing a directory deliberately is a coherent thing to want and a different feature —
        it needs the recorded rows reconciled against the queue in hand before any of them is
        trusted. Nothing here can do that, and without it an intended continuation and a rerun
        that forgot to move its output are the same call. So the ambiguity is refused at the one
        moment where nothing has been spent yet."""
        if self.results_path.is_file() and self.results_path.stat().st_size > 0:
            raise ValueError(
                f"{self.results_path} already holds recorded rows; this stream numbers from the "
                "start of its own queue and appends, so it would file a second run's rows under "
                "the first run's positions with nothing to tell them apart. Serve this queue "
                "into a fresh provenance directory."
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
        self, ref: TaskRef, instructions: str, budget: Optional[int]
    ) -> Dict[str, Any]:
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
        step; whole, a field added later is covered the day it is added.

        Nothing of this run's record is durable yet when this runs, and no position has been
        consumed, so a refusal costs no task. The stop is the run's, on the line this module
        already draws — an env key and a frozen contract are properties of the stream, not of this
        task, so the next dispense would fail the same way."""
        framing = {
            # This stream's key for the env, which is what its rows are recorded under — not the
            # name the instance calls itself, which nothing here is keyed by.
            "env": ref.env,
            # The two values `_require_framable` confirmed, not a fresh read of the spec.
            "instructions": instructions,
            "budget": budget,
            # The *published* manifest, never the live episode's: this is the contract the server
            # registered, and the check above is what makes the two the same thing. Detached from
            # it, and per dispense, for the reason :func:`_detached_manifest` gives: what a task
            # is framed with is a reading of the frozen contract, not a handle on it, and one
            # task's framing is not the next one's either. Its name and description are already
            # this stream's own plain copies, taken when the contract was frozen.
            "tools": [
                {
                    "name": m.name,
                    "description": m.description,
                    "input_schema": copy.deepcopy(m.input_schema),
                }
                for m in self._advertised[ref.env]
            ],
        }
        try:
            _wire_json(framing)
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
    def results(self) -> Sequence[Dict[str, Any]]:
        """The rows recorded so far, in seal order.

        A detached view of the recorded rows, rebuilt per read (see :func:`_detached_row`) — for
        the reason :attr:`tools` is one, and with the same consequence: what a reader is handed
        is a reading of the record, never a handle on it. A row is a dict of dicts, so a reader
        given the run's own could edit what the stream reports without touching what
        ``results.jsonl`` says, and every later read would show the edit."""
        return tuple(_detached_row(row) for row in self._results)

    @property
    def stopped(self) -> bool:
        """True once an integrity failure stopped the stream: a row that could not be recorded,
        a summary the record cannot headline, an env that raised while ending a task, or an
        episode framed with a contract the endpoint does not serve.

        Nothing the agent sees reports any of them. This flag, the rows already in
        ``results.jsonl``, and the exception :meth:`aclose` raises are where a harness learns of
        it — and a long run can poll this rather than wait for the drain."""
        return self._stopped is not None

    @property
    def tools(self) -> Sequence[ToolManifest]:
        """The tool manifest this stream serves (validated identical across the queue), as the
        frozen contract the constructor made of it.

        A detached view of that contract, rebuilt per read (see :func:`_detached_manifest`) —
        reading what this endpoint serves may not be a way to change it."""
        return tuple(
            _detached_manifest(manifest)
            for manifest in self._advertised[next(iter(self._advertised))]
        )

    def queue_info(self) -> Dict[str, Any]:
        return {
            "remaining": len(self._queue) - self._position,
            "consumed": self._consumed,
            "in_flight": sum(1 for live in self._live.values() if not live.sealed),
        }

    async def __aenter__(self) -> "TaskStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Orderly drain: seal, score and record every live episode, then release the catalog
        envs. Idempotent. This covers an orderly exit only — a SIGKILL or a ``docker rm -f``
        runs none of it.

        The drain is *total*: one episode that cannot be sealed does not cost the others their
        seal, and the catalog is released whatever happens above it — including a drain that is
        itself cancelled, since a failure that skipped the release would skip it for good.

        ``_closed`` refuses every later dispense, but on its own it must not *end* the drain: a
        shutdown cancelled mid-seal leaves a dispensed task still unrecorded and its episode
        still open, and returning early would make the retry that could still record it a no-op
        — a task counted as consumed that nothing can ever land a row for. So the drain runs
        again and finishes whatever the cancelled attempt left behind, while the catalog is
        released once. A second call over a settled stream therefore does nothing, which is the
        idempotence that was ever wanted. A task whose row already landed is joined here too, in
        case its seal's tail is still in flight (see :meth:`_settled`) — so a stop that tail owes
        is reported *by* this call rather than a moment after it returned.

        Raises if anything stopped the stream, here or earlier in the run — a dispensed task
        that went unrecorded, a summary the record cannot headline, an env that raised while
        ending a task, or an episode that would have been framed with a contract the endpoint
        does not serve. Together with :attr:`stopped` and the rows themselves, this is where a
        stream driven entirely over MCP reports any of them: the harness never calls
        ``get_task`` itself, and nothing the agent sees says a run went wrong. A retried drain
        reports it too: the report belongs to the run, not to whichever call reached the end
        of it first."""
        async with self._lock:
            self._closed = True
            try:
                for live in list(self._live.values()):
                    try:
                        await self._seal(live)
                    except Exception:  # noqa: BLE001 — recorded on the stream; drain the rest
                        pass
            finally:
                if not self._catalog_closed:
                    self._catalog_closed = True
                    # An env's ``close`` can raise ``CancelledError`` like any other exception,
                    # and one raised there is the env's rather than this caller's (see
                    # :func:`_must_propagate`). Letting it out would end the drain here, with
                    # ``_catalog_closed`` already set: the envs after this one are never closed
                    # and no later ``aclose`` will try again, so an orderly shutdown leaves them
                    # holding their sessions and subprocesses for the life of the process — and
                    # reports itself cancelled for a teardown that is not the run's outcome.
                    cancellation = _Cancellation()
                    for env in self._catalog.values():
                        try:
                            await env.close()
                        except BaseException as exc:  # noqa: BLE001 — teardown is best-effort
                            if _must_propagate(exc, cancellation):
                                raise
        if self._stopped is not None:
            raise RuntimeError(self._stopped.closing) from self._stopped.cause

    async def get_task(self) -> Optional[Dict[str, Any]]:
        """Dispense the next queued task, starting its episode. ``None`` once exhausted.

        Returns the framing and nothing else — ``{env, instructions, budget, tools}``, never the
        task index or the target. How much queue is left is a property of the *queue*, not of the
        task it hands out, and :meth:`queue_info` is where a caller reads it; a task that also
        reported its own position in the run would make the framing two contracts at once and
        the redaction claim a longer sentence than it needs to be.

        Pulling a new task while one is still live abandons it, so the abandoned episode is
        sealed and scored first: every dispensed task lands exactly one row.

        The tools it lists are the ones the endpoint actually serves, and the episode is checked
        against them before it is dispensed (see :meth:`_require_published_manifest`) — the
        framing an agent acts on and the surface it can call are the same contract or there is
        no task.

        The framing is also confirmed to be something the endpoint can *carry*, and confirmed
        before the dispense is committed (see :meth:`_require_framable`): a task that cannot be
        handed over has to be no task at all rather than a consumed position with no answer."""
        async with self._lock:
            self._require_open()
            # Every entry, not only the unsealed ones: a task whose row landed can still owe the
            # tail of its seal, if the caller that started it went away mid-release. `_seal`
            # joins that tail and returns the row. Stepping over it would open the next episode
            # while the last one was still being torn down — and, for a row this record cannot
            # headline, would dispense a task over a stop that was one await from being set.
            for live in list(self._live.values()):
                await self._seal(live)
            self._require_open()
            if self._position >= len(self._queue):
                return None

            ref = self._queue[self._position]
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
                # Nothing was dispensed: the position is still owed, no row is due, and no
                # registry entry names this episode — so it is released here rather than left
                # for a drain that would seal it and record a task nobody was ever handed.
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
            # Built and proved before the position is consumed and the episode registered: what
            # the agent is handed is this object, and every field of it has to be one the
            # endpoint can answer with (see :meth:`_deliverable_framing`). A refusal here is the
            # same shape as a drifted manifest — the position is still owed, no row is due, and
            # the episode is released by the handler above.
            try:
                framing = self._deliverable_framing(ref, instructions, budget)
            except BaseException:
                cancellation = _Cancellation()
                try:
                    await episode.close()
                except BaseException as exc:  # noqa: BLE001 — teardown must not mask the failure
                    if _must_propagate(exc, cancellation):
                        raise
                raise
            self._position += 1
            self._seq += 1
            self._consumed += 1
            live = _Live(
                lease=secrets.token_hex(16),
                seq=self._seq,
                position=self._position - 1,
                ref=ref,
                episode=episode,
            )
            self._live[live.lease] = live
            return framing

    async def dispatch(self, tool: str, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Route one native tool call to the live episode, sealing it when it terminates.

        An ordinary call returns the env's own response: that *is* the agent's observation, and
        nothing but the env can produce it. A terminating call returns only the fact that the task
        is over — a fixed payload identical for every task and every outcome.

        Everything a terminal produces stays with the harness, not just the seal's provenance row
        (lease, position, task index, raw feedback). The env's terminal response is redacted too:
        for a ``score`` terminal it is the verdict this stream just recorded, and a queue that
        repeats an index would make it the signal that identifies the repeat. The feedback sidecar
        a served episode rides its terminal feedback out on is dropped for the same reason and
        must stay dropped — relaying it, as the single-episode server does, would reopen the
        channel this closes.

        A call that ends the task answers with that payload *whatever happened while it ended*.
        An exception is a different response shape, and the shape is the channel: an env that
        published a clean summary on one verdict and a malformed one on the other would tell the
        agent its verdict through whether the call succeeded. So a failed seal — and an env that
        raises once it has already ended the episode — are recorded on the stream and answered
        with the same bytes as a clean one. Only a call that leaves the task *live* still raises:
        there the exception is the env's own answer to a call the agent can make again, no
        different in kind from the env text an ordinary call returns, and no task has ended for
        it to be a verdict about."""
        async with self._lock:
            live = next((it for it in self._live.values() if not it.sealed), None)
            if live is None:
                return ToolResult(
                    content=json.dumps(
                        {"error": "no active task", "hint": f"call `{_GET_TASK_TOOL}` first"}
                    )
                )
            cancellation = _Cancellation()
            try:
                call = await live.episode.call(tool, dict(arguments or {}))
            except BaseException as exc:  # noqa: BLE001 — see below; never re-raised at the agent
                # An env can raise `CancelledError` like anything else, and one raised *by the
                # env* is not this caller's cancellation (see `_must_propagate`). Told apart here
                # rather than by type, because letting it through skips everything below: the
                # stop is never recorded, so the queue serves on against an env that already lost
                # an outcome; the seal never runs; and the terminating call answers the agent
                # with a traceback in place of the constant every other ending returns.
                if _must_propagate(exc, cancellation):
                    raise
                if not live.episode.terminated:
                    raise
                # The episode ended and *then* the call failed: the terminal is committed and
                # the feedback the episode was about to hand over is what raised. So a row is
                # still owed — it lands carrying whatever feedback survived, which is what makes
                # the loss legible — and the stream stops. A row with no readable outcome is the
                # same eval-integrity failure whether the value is unusable or missing, and an
                # env that raises here raises for every task in the queue; without the stop a
                # solved task is recorded unscored and `aclose()` reports a clean run.
                #
                # The failure is described through the guarded funnel: this message is an
                # *argument* to the stop, so a failure that cannot be formatted would take the
                # stop, the row and the redacted answer with it — see :func:`_rendered_failure`.
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
                return ToolResult(
                    content=json.dumps({"content": call.content, "terminated": False})
                )
            await self._seal_redacted(live)
            return ToolResult(content=_TASK_OVER)

    # ----- sealing -----

    async def _seal_redacted(self, live: _Live) -> None:
        """Seal a task whose terminating call is being answered with the redacted payload.

        The row :meth:`_seal` returns is for the harness, not the caller — and so is the
        exception it raises instead. Every failure it can raise is already recorded on the stream
        before it leaves: whatever row did land is in ``results.jsonl``, :attr:`stopped` is set,
        the next dispense refuses, and :meth:`aclose` raises. Nothing is therefore lost by
        answering the agent with the constant, and raising instead would tell it precisely what
        this call is not allowed to tell it.

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

    async def _seal(self, live: _Live) -> Dict[str, Any]:
        """End the episode authoritatively, read its score off the sealed evidence, record the
        row, and release the episode (and its env). Runs at most once per dispensed task.

        A row that cannot be persisted stops the stream (see :meth:`_require_open`) — a sync
        that fails is a row that was not recorded, exactly like a write that fails. So does a
        row that cannot be *summarized*: a ``success``/``reward`` that is wrong-typed, or
        published twice, is a property of the env rather than of the task, so it would recur
        for the whole queue.

        **A seal that was abandoned is not a seal that failed.** The drain, the next dispense
        and a retried call all reach here, and a caller that goes away mid-seal — an MCP client
        that cancelled its request, a shutdown that was itself cancelled — has not lost
        anything: the episode is still alive and still holds the authoritative outcome. So
        cancellation hands the claim back rather than consuming it. The entry stays in the
        registry, unsealed and unreleased, and the next arrival finishes the seal this one
        started. Marking it sealed would make a dispensed task invisible to every later
        attempt: no row, no release, ``consumed`` counting it and nothing able to record it —
        the one-row-per-dispense invariant broken by an ordinary cancellation. Stopping the
        stream for it would be wrong for the same reason: nothing was lost, so there is nothing
        to stop for.

        **Once the row lands, though, there is nothing left to hand back.** ``row`` is the
        finality marker, so everything downstream of it — the release, and the stop a summary
        this record cannot headline owes — is the tail of a seal that already happened and can
        only be *finished*, never retried. It therefore runs as the stream's own work rather
        than the caller's; see :meth:`_settled`."""
        if live.row is not None:
            # A final row whose tail a lost caller left in flight: join it rather than stepping
            # over it, so nobody reads this entry as settled ahead of the stop it may still owe.
            await self._settled(live)
            return live.row
        # Claim the seal before the first await. The *row* is what makes the claim final, which
        # is why every other reader tests `row` and not this flag: a claim that is abandoned is
        # handed straight back below.
        live.sealed = True
        try:
            row = await self._record(live)
        except BaseException as exc:
            if live.row is None:
                live.sealed = False
                if isinstance(exc, asyncio.CancelledError):
                    # Deferred, not lost. Nothing is released and nothing is recorded on the
                    # stream: the episode is left exactly as this attempt found it, so the next
                    # drain or dispense can still record the outcome it is holding.
                    raise
            # The invariant this object sells is that every dispensed task lands exactly one
            # row. Breaking it is not a resource problem, it is an eval-integrity one: the
            # scored episode is gone from the record and the file that remains looks complete,
            # so a benchmark number computed over it is quietly wrong. Remember the failure and
            # stop dispensing, so the run ends at the first lost row rather than finishing and
            # being believed. Recovering is the operator's call, not this object's.
            self._stop(
                exc,  # the first loss is the one that explains the run
                dispensing=(
                    "this stream stopped: a dispensed task could not be recorded to "
                    f"{self.results_path}, so the run's record is missing an outcome the agent "
                    "actually earned"
                ),
                closing=(
                    f"this stream could not record every dispensed task to {self.results_path}; "
                    "the run's record is incomplete"
                ),
            )
            # A seal that *failed* is not retried, so this is the last chance to let go of what
            # the entry holds — the release has nothing to do with whether the row landed. Not
            # the shared tail below: no row landed, so this entry's claim has just been handed
            # back, and a tail left running over a handed-back claim is one a drain arriving
            # inside it could re-record on top of. The stop this path owes is already set, above.
            await self._release(live)
            raise
        await self._settled(live)
        if live.summary_error is not None:
            # Recorded on the stream by the tail above, which also released the episode; all
            # that is left here is telling this caller. Raised after the release because a row
            # that landed makes the seal final: a later `_seal` returns that row without
            # releasing anything, so raising ahead of it would strand this episode's env for the
            # life of the process.
            raise live.summary_error
        return row

    async def _settled(self, live: _Live) -> None:
        """Wait for everything a landed row owes — **without being able to abandon it**.

        The release and the summary stop are consequences of a row that is already durable, so
        they belong to the stream rather than to whichever of the drain, the next dispense or a
        retried call reached them. A caller can be cancelled at any await it owns, and both of
        these are awaits: an MCP client that gave up on its request would otherwise leave the
        episode's env open with its registry entry already dropped — unreachable by any later
        drain, so its MCP sessions and subprocesses are held for the life of the process — and,
        for a summary this record cannot headline, would skip the stop that is the whole reason
        the offending row was written first. That row's ``success`` is ``null``, which in a run
        that *finished* means exactly one thing: the env published no such field. Losing the stop
        turns it into the other thing, and the run reports itself complete carrying it.

        So the tail is run as a task and awaited through a shield: this caller's cancellation
        reaches the caller and stops there, while the work goes on to completion. It is claimed
        once and every later arrival joins that same task, so it never runs twice."""
        settling = live.settling
        if settling is None:
            settling = live.settling = asyncio.ensure_future(self._settle(live))
        await asyncio.shield(settling)

    async def _settle(self, live: _Live) -> None:
        """The claimed tail: release the episode, then stop the stream if the row it recorded
        carries a summary this record cannot headline.

        The two are ordered and the order is not cosmetic. The row landed first, deliberately:
        ``feedback`` is authoritative and is written verbatim, so every offending item is in
        ``results.jsonl`` as durable evidence of *which* env published *what* — an exception
        string dies with the process. That is also why the record survives a duplicate: both
        occurrences are in the row, so the ambiguity the stream refused to resolve is legible in
        the file. The release comes next, and only then the stop, so that no reader can find this
        entry gone from the registry while the stream still looks like it is serving."""
        await self._release(live)
        if live.summary_error is not None:
            self._stop(
                live.summary_error,
                dispensing=(
                    f"this stream stopped: env {live.ref.env!r} published a summary value "
                    f"this record cannot headline ({live.summary_error}), so no further task "
                    "can be scored against it"
                ),
                closing=(
                    f"this stream stopped before its queue was served: env "
                    f"{live.ref.env!r} published a summary value this record cannot "
                    f"headline ({live.summary_error})"
                ),
            )

    async def _release(self, live: _Live) -> None:
        """Let go of a task's episode (and its env) and drop it from the registry.

        Separate from recording it, and reached whether or not the row landed: the episode owns
        MCP sessions and an env, this entry is the only handle on either, and a seal that failed
        is not retried.

        Nothing here may raise, ``CancelledError`` included, and that one costs most. An env's
        ``close`` can raise it, and on the settled path this runs inside the tail task that
        :meth:`_settled` joins through a shield — so one observed here is the env's, not a
        caller's. Out it would go into the middle of :meth:`_seal`, PAST the durable append and
        BEFORE the stop an unheadlinable summary still owes: the terminating call would answer
        the agent with a traceback instead of the constant, the stop would never be published,
        and ``aclose`` would then report a clean run over a row this record has already refused
        to headline."""
        cancellation = _Cancellation()
        try:
            await live.episode.close()
        except BaseException as exc:  # noqa: BLE001 — the row is settled; teardown is best-effort
            if _must_propagate(exc, cancellation):
                raise
        finally:
            self._live.pop(live.lease, None)

    async def _record(self, live: _Live) -> Dict[str, Any]:
        """Bring the episode to an end, read its score off the sealed evidence, and append its
        one durable row.

        A task that already ended is *adopted*, never re-ended. A terminating call whose caller
        went away leaves the episode sealed with its verdict still in flight, and the score this
        row reports has to be the one that transaction commits — the episode is the authority on
        how it ended, not whichever of the drain, the next dispense or a retried call happens to
        arrive first.

        Recording is *durable first*: a row reaches :attr:`results` only once it is in
        ``results.jsonl`` **and fsync'd**, so the in-memory view can never claim an outcome the
        file does not have. The alternative ordering reads as harmless and is not — whoever
        scores the run would see a complete set of rows while the durable record was silently
        one short. Closing the file is not enough on its own: it survives this process dying,
        not the host, and the record's whole job is to outlive both."""
        episode = live.episode
        # A terminating call whose caller was cancelled leaves the episode sealed with its
        # finalization still running — the episode shields it so a disconnect can never abandon
        # the evaluator. Wait for that verdict to land before reading anything off the episode.
        # Forcing a terminal over the top of it would only read the post-seal tombstone and then
        # snapshot the feedback the finalizer has not published yet, so a task the agent solved
        # would be recorded unscored, with empty feedback — indistinguishable from an env that
        # publishes no summary at all. The wait costs nothing that was not already spent: the
        # release awaits the same finalization through `episode.close()`, so this only changes
        # whether the row is read before or after it.
        await episode.wait_finalized()
        if not episode.terminated:
            # The agent stopped short. Drive the env's own terminal on its behalf so the row
            # carries an authoritative outcome rather than a guess: the score terminal first
            # (a sealed partial state can still earn partial credit), falling back to the
            # reserved abort when the env's score terminal needs arguments we cannot invent.
            # A `CancelledError` the env raises is that env failing and is contained with the
            # rest (see :func:`_must_propagate`). Letting it through instead cancels whoever is
            # sealing: no row is composed for a task that was dispensed, the entry is handed
            # back unsealed, and the drain that meets it reports the run as cancelled rather
            # than recording the outcome the queue is still owed.
            score_terminal = self._score_terminal.get(live.ref.env)
            cancellation = _Cancellation()
            if score_terminal is not None:
                try:
                    await episode.call(score_terminal, {})
                except BaseException as exc:  # noqa: BLE001 — a refused seal falls through to abort
                    if _must_propagate(exc, cancellation):
                        raise
            if not episode.terminated:
                try:
                    await episode.call(TERMINATE_TOOL_NAME, {})
                except BaseException as exc:  # noqa: BLE001
                    if _must_propagate(exc, cancellation):
                        raise

        # The env's items, in the order and at the levels it published them. Keyed by name this
        # would be a *projection*, not a record: an env may publish a name twice, and a mapping
        # silently keeps one of them — losing the evidence and, for a summary name, deciding
        # the headline by list order (see `_pick_summary`).
        feedback = [dict(item) for item in episode.terminal_feedback]
        # The summary is read as a unit and strictly (see `_pick_summary`): either this row
        # carries the env's own headline or it carries none, never a half-filled one beside a
        # value the stream is refusing to trust. A malformed one is carried on the entry and
        # raised by the caller once the row is durable and the episode released.
        reward: Optional[float] = None
        success: Optional[bool] = None
        try:
            reward = _pick_float(feedback, _REWARD_NAMES)
            success = _pick_bool(feedback, _SUCCESS_NAMES)
        except _MalformedSummary as exc:
            reward = None
            success = None
            live.summary_error = exc
        row = {
            "seq": live.seq,
            "lease": live.lease,
            "position": live.position,
            "env": live.ref.env,
            "task_idx": live.ref.task_idx,  # provenance only — never returned to the agent
            "reward": reward,
            "success": success,
            "feedback": feedback,
        }
        # Everything from here is synchronous, so no cancellation point can split the durable
        # row from the claim it makes: an entry either has its row or is still sealable.
        self._append_row(row)
        # What the run keeps is the row *the file now holds*, re-read from its own wire form.
        # That is the canonical snapshot every reader is shown a copy of (see :attr:`results`),
        # and taking it here is what makes those copies cheap and certain: they copy plain data,
        # run no env code, and cannot disagree with the record. Held any other way this list
        # would carry the env's own values — the feedback items it published — and every view of
        # it would be a handle on them.
        recorded = _recorded_row(row)
        live.row = recorded
        self._results.append(recorded)
        return recorded

    def _append_row(self, row: Dict[str, Any]) -> None:
        """Append one row and *sync it*, for a record whose whole point is to survive the
        process that wrote it. Closing the file only hands the bytes to the kernel — enough to
        outlive this process, not the host — and the caller publishes the row and answers the
        agent the moment this returns. The first row also syncs the directory: an unsynced
        directory entry can lose the whole file, rows and all — and, when the directory had to
        be created here, so can an unsynced entry naming *it* (see :func:`_mkdir_durable`).

        **The terminating newline is written after the row it terminates is already on disk**,
        so a row that survives cannot take the rest of the file with it. Written together they
        carry no ordering: a row is one ``write`` — 46 KB of published feedback is still one —
        and a crash is free to persist the block holding the newline while losing one in the
        middle, leaving a torn write that reads back as a whole record. Split, a terminated line
        is one that was durable before its terminator was, and the crash costs a reader only the
        record nobody was ever told about: an unterminated last line, which is a write that
        never returned. Line-delimited JSON is read a line at a time by everything that consumes
        it, so one torn record at the end otherwise makes the whole run unreadable — every intact
        row before it included."""
        _mkdir_durable(self.prov_dir)
        fresh = not self.results_path.exists()
        with self.results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False))
            handle.flush()
            os.fsync(handle.fileno())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if fresh:
            _fsync_dir(self.prov_dir)

    def _require_open(self) -> None:
        if self._stopped is not None:
            raise RuntimeError(self._stopped.dispensing) from self._stopped.cause
        if self._closed:
            raise RuntimeError("this stream is closed")


def _recorded_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """A composed row, re-read from the wire form the results file just committed — the run's
    own canonical copy of what it recorded.

    Composing a row leaves the env's values on it: ``feedback`` holds the items the episode
    published, so the row a seal builds is a handle on env objects, and this is where the run
    stops holding one. Taken through the same strict JSON the file holds — the same encoder and
    the same ``allow_nan`` :meth:`TaskStream._append_row` committed it with — so what stays in
    memory and what a later reader parses out of ``results.jsonl`` are the same row, and every
    view taken of it afterwards copies plain data (see :func:`_detached_row`).

    Called **after** the append, which is what makes it safe to run at all: those exact values
    have just been serialised, so nothing here can fail that the write did not already fail. The
    same call a step earlier would be a normalization that could suppress a row the run had
    otherwise earned."""
    return json.loads(json.dumps(row, allow_nan=False))


def _detached_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """One recorded row, copied whole, for a reader that must not be able to reach the run's.

    ``deepcopy`` rather than a serialization round trip, and here that is total rather than a
    risk: what is behind this is already plain data (see :func:`_recorded_row`), so copying it
    copies data and runs no code an env wrote — a reader gets the row whatever the env
    published in it."""
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
        index or the target. Returns ``{"done": true}`` once the queue is empty. Work the task
        with the native tools it lists; they route to it automatically."""
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
            info = stream.queue_info()
            return {"done": True, "remaining": 0, "consumed": info["consumed"]}
        return dispensed

    @server.tool(name=_QUEUE_INFO_TOOL)
    async def queue_info() -> Dict[str, Any]:
        """Report ``{remaining, consumed, in_flight}`` for the task queue."""
        return stream.queue_info()

    reserved = {_GET_TASK_TOOL, _QUEUE_INFO_TOOL}
    for manifest in stream.tools:
        if manifest.name in reserved:
            raise ValueError(
                f"env tool name {manifest.name!r} collides with the stream's reserved control "
                f"tool; an env served by a stream may not expose a tool named {manifest.name!r}"
            )
        server.add_tool(build_tool(manifest, stream.dispatch))
    return server
