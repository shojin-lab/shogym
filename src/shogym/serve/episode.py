"""The episode-serving engine (RFC 008): drive a served env one tool call at a time.

A tool call *is* the step. :class:`ServedEpisode` opens the env's essential MCP sessions
directly (the terminate server + the env-mandatory servers), dispatches each incoming call
to the right session, records a flat :class:`~shogym.trajectory.Trajectory`, gates the
horizon env-side, and runs the env's pure ``verify`` on each call. Feedback rides back on
the ``_meta`` sidecar (episode-level hidden until the terminal result) and every step is
appended to the JSONL trace. No gym ``step``/``Observation`` anywhere.

**Seal-before-verdict (durable + middleware-gated).** An env opts in by declaring a
``score``-terminal tool (``terminal_kind="score"`` in its manifest) *and* a ``finalize`` hook.
For such an env, a score-terminal call runs a *terminal transaction* instead of an ordinary
step: :meth:`call` (1) validates the args against the full advertised JSON schema while the
episode is still OPEN — an invalid request is a normal validation error, never a sealed
finalizer with unvalidated evidence; (2) atomically seals the episode (``OPEN -> SEALED``);
(3) runs ``finalize`` as a single, cancellation-safe finalization producing core-owned
:class:`~shogym.serve.lifecycle.TerminalEvidence`; (4) records the evidence in the **durable
finalization store** and appends a versioned ``terminal`` trace event; (5) tears down; (6)
returns a public-safe, sanitized payload. After the seal every ``tools/call`` is tombstoned
with no inward dispatch. The state machine is
``OPEN -> SEALED -> FINALIZING -> FINALIZED -> TEARING_DOWN -> CLOSED``.

An env that marks nothing ``score`` never enters this path: ``_seal_enabled`` is False, so
:meth:`call` runs *only* the legacy single-step body — its behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

import jsonschema
from jsonschema.validators import validator_for

from shogym.envs import make
from shogym.feedback.wire import build_meta, dump_item, load_item, select_inband
from shogym.mcp.session import MCPSession
from shogym.mcp.toolset import _open_session_for_spec
from shogym.serve.lifecycle import (
    FinalizationRecord,
    FinalizationStore,
    FinalizeRequest,
    LifecycleState,
    TerminalEvidence,
    args_digest,
    fail_closed_verdict,
)
from shogym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from shogym.task import ReferenceTemplate, TaskSpec, ToolManifest
from shogym.trace import (
    append_terminal_event,
    append_trace,
    step_record,
    terminal_event_record,
)
from shogym.trajectory import Step, Trajectory
from shogym.utils.uuid7 import uuid7


def _named(key: Any) -> str:
    """The schema key a validation error is *about*, or a placeholder when naming it raises.

    A schema key is the env's own object and ``repr`` is that object's code, run here only to say
    which argument the caller has to fix. Unguarded it replaces the validation error with its own
    exception, which is the one outcome this check exists to prevent: the episode is left OPEN
    with the terminal call the agent made answered by a traceback, and the harness above ends up
    composing the task's outcome from a call it never carried. The refusal is the point and the
    name is the decoration, so the name is what gives way — and nothing of the failure is echoed
    into it, because this message goes to the caller and an env's exception text is not the
    caller's business."""
    try:
        return repr(key)
    except Exception:  # noqa: BLE001 — the refusal outranks its own decoration
        return "<a key this schema cannot name>"


def _described(render: Callable[[], str]) -> str:
    """An env's own value, written into a refusal that is *about* that value — or a placeholder
    when asking for it raises.

    Same rule as :func:`_named` and for the same reason: ``repr`` is the env's code on the env's
    object, called here only while building the message that refuses the value, so the value
    would otherwise get to decide whether the refusal happens at all. The refusal is the point
    and the description is the decoration, so the description is what gives way."""
    try:
        return render()
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:  # noqa: BLE001 — the refusal outranks its own decoration
        return "<a value this refusal cannot describe>"


class _Cancellation:
    """Whether cancellation has been requested against **this task** since this was taken.

    Two unrelated things arrive as :class:`asyncio.CancelledError` and they are the same
    exception: this task being cancelled, and code this task *called* raising one. Nothing about
    the object tells them apart; ``Task.cancelling()`` does, because only the first moves it. So
    the count is taken before the calls a boundary contains and compared after them.

    Taken once for a whole boundary rather than per call, because a request against this task
    outlives whichever callee it interrupted. With no running task there is nothing a
    cancellation could have been requested *against*, so the conservative reading is kept and it
    reads as requested.

    (The same rule, and the same two classes, are what :mod:`shogym.serve.stream` holds its own
    containment boundaries to. They are restated here rather than shared because that module
    imports this one.)"""

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

    A containment boundary here is one guarding a call into the *env*, whose failure is not the
    caller's outcome: a verifier scoring already-committed evidence, a teardown, the cleanup that
    releases a half-built episode. Those catch ``BaseException`` and ask this. An
    ``except Exception`` elsewhere in this module is the opposite thing and stays that way: it
    guards this module's own coroutine, run in the caller's task, where a ``CancelledError`` is
    that caller's own cancellation and has to end it.

    At a containment boundary an env's ``CancelledError`` is third-party code failing, no
    different in kind from any other exception it raises — unless cancellation was requested
    against this task while the call was running, the one case where the exception belongs to
    whoever asked for it. ``cancellation`` is ``None`` for a boundary with no ``await`` in it:
    nothing can be delivered to a task that never suspends, so one observed there was raised
    where it was observed.

    ``SystemExit`` and ``KeyboardInterrupt`` always propagate — swallowing an interpreter-level
    signal is worse than losing what the boundary was protecting."""
    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
        return True
    if isinstance(exc, asyncio.CancelledError):
        return cancellation is not None and cancellation.requested()
    return not isinstance(exc, Exception)


_UNRENDERABLE = "<unrenderable>"


def _failure_type(exc: BaseException) -> str:
    """The class name of a failure, when even *that* runs code this module did not write.

    ``__name__`` is an attribute of the class and a metaclass may define it as a property, so it
    is guarded like the message below and for the same reason. A name that comes back as
    something other than a string is refused rather than formatted."""
    try:
        name = type(exc).__name__
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:  # noqa: BLE001 — see `_rendered_failure`
        return _UNRENDERABLE
    return name if isinstance(name, str) else _UNRENDERABLE


def _rendered_failure(exc: BaseException) -> str:
    """Describe a failure the env supplied, without running its code unguarded.

    Every diagnostic this module writes about a failure it has *caught* formats that failure, and
    formatting an exception runs code belonging to whoever raised it — a second time, and outside
    the ``except`` that just contained it. ``__str__`` is theirs, and an accident is enough: a
    message built lazily from state that is gone by the time it is asked for raises here rather
    than at the raise site. The second exception is not the one the handler caught, so it does not
    stay caught — it walks out of the handler carrying the handler's job with it. Measured, at the
    one site in this module that does this: the fail-closed verdict a crashed evaluator is owed is
    never built, so no evidence is committed, no ``FAILED`` record is written and no teardown
    runs; the episode is left sealed-but-unterminated with its durable record at ``PENDING``,
    while ``close()`` returns clean. A failure this module has already decided to contain may not
    be un-contained by the act of writing it down.

    So: the message is attempted, then the type alone, then a constant. What is never attempted
    twice is the env's code — a fallback that formatted the same object again would be the same
    bug one line down.

    ``CancelledError`` is caught here rather than let through. Nothing in this function awaits, so
    no cancellation can be *delivered* during it; one raised here was raised by the object being
    rendered, and letting it through would strand the seal exactly as the message it was
    describing does. ``SystemExit``/``KeyboardInterrupt`` still propagate."""
    name = _failure_type(exc)
    try:
        return f"{name}: {exc}"
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:  # noqa: BLE001 — a contained failure may not escape through its message
        return f"{name}: <unrenderable message>"


def _declared(value: Any, declared: Tuple[str, ...]) -> Optional[str]:
    """The **core's own** string when ``value`` *is* one of the ones this module declares, else
    ``None``.

    Two rules, and the second is why this is a function rather than an ``in``. First the type:
    ``type(value) is str``, not ``isinstance``, because a ``str`` subclass is the env's object
    with the env's ``__eq__``, and Python offers the reflected comparison first when the right
    operand's type is a proper subclass of the left's — so no arrangement of the operands keeps a
    subclass from answering for itself. A non-``str`` is worse: an object whose ``__eq__`` returns
    ``other == "ok"`` impersonates the literal from either side, and that is how an undeclared
    status was rewritten into this core's success. The type check settles both before any
    comparison happens.

    Then the identity: what comes back is the constant from ``declared``, never the value that
    matched it. A value that merely compares equal to a declared string is not that string, and
    keeping the env's object because it *said* it was equal leaks the comparison into everything
    downstream that reads the field afterwards."""
    if type(value) is not str:
        return None
    for candidate in declared:
        if value == candidate:  # both are exactly `str` by now, so this is `str.__eq__`
            return candidate
    return None


# The outcomes this core declares. An evaluator reporting anything else has not reported one.
_TERMINAL_STATUSES: Tuple[str, ...] = ("ok", "finalize_error")
# The argument the transport injects, and the JSON type the non-blank rule is about. Both are
# read off a schema the env owns, so both are matched rather than compared (see `_declared`).
_RESERVED_ARGS: Tuple[str, ...] = ("_session_id",)
_TEXT_TYPE: Tuple[str, ...] = ("string",)


def _wire(value: Any) -> Tuple[bool, Any]:
    """Whether ``value`` renders to JSON, and the rendering when it does.

    The **product** comes back with the answer, and the product is what a caller retains: this is
    never a question asked about a value that is then used anyway. A predicate followed by reuse
    of the original is a second walk, the second walk is a second question, and a container that
    permits one and refuses the next is admitted by one and committed by the other.

    The flag rather than a sentinel value, because a rendering can legitimately *be* the value it
    started as — ``json.loads(json.dumps(5))`` is ``5``, and small integers are interned — so
    identity says nothing about whether the rendering happened. Only the read belongs to the
    caller; asking the env's attribute a second time to recover what did not render would be the
    same bug one level up."""
    try:
        return True, json.loads(json.dumps(value, allow_nan=False))
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:  # noqa: BLE001 — a value that will not render is the caller's problem
        return False, None


def _wire_verdict(verdict: Any) -> Any:
    """A verdict as the wire carries it: plain JSON data, so the value this transaction *checks*
    is the value it commits, publishes and persists.

    The same round trip :func:`_core_spec` makes of the contract, for the same reason. A verdict
    is the env's object and every consumer walks that object again: the sanitized payload the
    agent is answered with, the durable record, the trace event. Nothing obliges two walks to
    agree, so a container that answers once and misbehaves after is checked as one value and
    committed as another.

    **A failed rendering raises rather than handing the value back.** Returning the env's object
    for a check further down made that check a *second walk*, which is a second question, and a
    container that refuses the first walk and permits the second was admitted on the strength of
    the walk that succeeded. It rode into the commit as the original object and was walked again
    by everything downstream: the durable write degraded and left the record with no verdict at
    all while the run answered ``correct=true``, and the trace event a step later is reached
    through a guard that catches only ``Exception``, so a cancellation raised there takes the
    whole finalization with it. A value this cannot render is not rescued by asking it a second
    time. The raise is the answer, it is given at the first asking, and it is given inside the
    evaluator guard, which turns it into the canonical fail-closed verdict."""
    return json.loads(json.dumps(verdict, allow_nan=False))


def _core_feedback(items: List[Any]) -> "Tuple[List[Dict[str, Any]], List[Any]]":
    """The env's feedback rendered once, and this module's own items rebuilt from that rendering.

    A feedback item is the env's object and every sink here serializes it again: the retained
    terminal feedback, the trace row, the in-band sidecar the caller is answered with. Three
    serializations are three questions to one object, and nothing obliges an env to answer them
    the same way. Measured on the terminal path: a name that answered the first read and raised
    cancellation on the second let the guarded render succeed, the evidence commit, and then took
    the finalization out from under a verdict already committed and already public.

    This **consciously reverses** the earlier decision to leave feedback items as the env's
    objects. That decision was defensible on the reasoning that a scribbler deceives only its own
    later reads, and that reasoning did not account for this module reading the same objects
    again itself. Rendering once and rebuilding is the rule the contract and the evidence already
    follow; feedback was the last collection still exempt from it.

    What is *not* reversed is who owns a feedback failure. Rendering can still fail, and when it
    does the exception is the layer above's to classify exactly as before.

    **Reading each field once is not the same as rendering it.** ``dump_item`` asks the item for
    its ``name``, ``value`` and ``step`` exactly once each, and then puts *those objects* into a
    new dict: the container is the core's and everything in it is still the env's. The models are
    mutable and do not validate on assignment, so an env can hang a ``str`` subclass on a
    validated item, and that subclass rode into the retained terminal feedback while the trace and
    the sidecar read the rebuilt items. Measured: a terminal ``correct`` whose name answered every
    comparison false was the object a stream headlines its row from, so a task the agent solved
    was filed sealed, with the evidence intact and ``score.success`` null: a valid answer
    recorded without its headline.

    So the dicts are rendered too, once, and **both** return values come out of that one
    rendering. The wire form a sink retains and the items a sink walks are then two shapes of the
    same plain data rather than two readings of the env's."""
    wire = json.loads(json.dumps([dump_item(item) for item in items], allow_nan=False))
    return wire, [load_item(entry) for entry in wire]


def _core_args(args: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The submission this episode seals on, rendered once and owned by this module.

    The rendering is the **same walk the durable witness already makes**: the digest normalizes
    through ``json.dumps(..., default=str)``, so taking that normalization as a value rather than
    only as bytes costs no walk that was not being taken, admits exactly what the digest admits,
    and leaves every digest this core has ever written unchanged. What it buys is that the
    submission the record witnesses and the submission everything downstream reads are one value
    instead of two readings of the caller's dictionary.

    Rendered here, at the seal, and above the transition for the reason the digest is above it:
    this walk runs the caller's values and may raise, and a failure has to be an ordinary lost
    call on an episode still ``OPEN``, never an escape from a sealed episode with no finalization
    to answer for it."""
    if args is None:
        return None
    return json.loads(json.dumps(args, default=str))


def _detached_args(args: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The sealed submission as something env code may keep, scribble on, and hand back.

    :class:`FinalizeRequest` carries the arguments so an evaluator can grade them, which means the
    evaluator holds a reference to them for the length of that call. The terminal ``Step`` the
    verifier scores, and ``evidence.args``, were then built from that same dictionary afterwards.
    A finalizer that rewrote ``req.args`` therefore changed what the trajectory said the agent
    submitted while the digest went on witnessing the call that actually arrived: the record and
    the scored trajectory described different submissions, and the run reported a clean success
    over the disagreement.

    The same corollary :func:`_detached_evidence` states, one hook earlier: the value checked is
    the value committed only if nothing foreign holds a reference to it in between. A copy of
    plain data runs nobody's code."""
    return copy.deepcopy(args) if args is not None else None


def _detached_evidence(evidence: TerminalEvidence) -> TerminalEvidence:
    """The committed evidence as something env code may keep, scribble on, and hand back.

    :meth:`Env.verify` takes the terminal evidence so a migrated env can score from it, and the
    commit that follows reads the same fields again. Handing over the object the commit reads
    made those two the same object, so a verifier could answer normally and rewrite the outcome
    on its way out. Everything here is already plain data or the core's own dictionary, so the
    copy walks nothing the env wrote: the verdict was rendered at the evaluator boundary and the
    provenance is this module's."""
    return TerminalEvidence(
        source=evidence.source,
        status=evidence.status,
        verdict=copy.deepcopy(evidence.verdict),
        args=dict(evidence.args) if evidence.args is not None else None,
        provenance=dict(evidence.provenance) if evidence.provenance is not None else None,
        finalization_id=evidence.finalization_id,
        diagnostic=evidence.diagnostic,
        schema_version=evidence.schema_version,
    )


def _core_owned(evidence: TerminalEvidence) -> TerminalEvidence:
    """The env's terminal evidence, read once and rebuilt as an object this module owns.

    ``isinstance`` admits a subclass, so what ``finalize`` hands back is only *shaped* like
    evidence: any field on it can be a property, every read of one runs the env's code again, and
    nothing obliges two reads to agree. The commit downstream reads ``verdict`` twice before it
    trusts it (once to check it is a JSON object, once to check it serializes) and ``status`` and
    ``diagnostic`` after that, all outside the guard that turns an evaluator failure into a
    fail-closed verdict. A field that answered the first read and raised on the second therefore
    walked out of the finalization with the commit half-made: the durable record left at
    ``PENDING`` with no verdict, the lifecycle at ``FINALIZING``, and ``close()`` clean over it.
    The checks were being made against a value free to be a different value by the time it was
    used.

    So every read happens here, exactly once each, inside that guard, and what comes out is this
    module's own dataclass. Two fields are normalized rather than merely copied:

    - ``verdict`` is round-tripped to plain data (:func:`_wire_verdict`), so the value checked is
      the value committed.
    - ``status`` is **matched against the two outcomes this core declares** (:func:`_declared`),
      not tested for one of them. :class:`TerminalEvidence` is a plain dataclass, so its
      ``Literal`` annotation checks nothing at runtime, and reading the field as "failure if it
      is exactly ``finalize_error``, success otherwise" made every other value an env could put
      there a *success*: a finalizer returning ``status="not-a-terminal-status"`` beside
      ``correct=True`` was published as a clean result and recorded ``FINALIZED`` with no
      diagnostic. Testing it with ``==`` was not enough either, because the env's object is on
      the dispatching side of that comparison: an object whose ``__eq__`` returns
      ``other == "ok"`` impersonated the literal and was rewritten to this core's success. The
      value has to *be* a declared string, and what is kept is this core's copy of it. An
      evaluator that cannot say which of the two outcomes it reached has not reached one, and
      reading its silence as the good one is the whole failure mode this module is written
      against, so anything else raises into the same fail-closed route a crashed evaluator takes.

    ``args`` and ``diagnostic`` are carried as read: neither decides an outcome, the diagnostic
    is private and reaches only the durable write (which is best-effort against every failure),
    and the args are the env's to hand its own verifier. ``schema_version`` is not carried at
    all, because the envelope's version is the core's to state, not the env's."""
    raw = evidence.status
    status = _declared(raw, _TERMINAL_STATUSES)
    if status is None:
        raise ValueError(
            "finalize returned a terminal status this core does not declare: "
            f"{_described(lambda: repr(raw))}"
        )
    return TerminalEvidence(
        source=evidence.source,
        status=status,  # type: ignore[arg-type]  # `_declared` returns one of the two literals
        verdict=_wire_verdict(evidence.verdict),
        args=evidence.args,
        diagnostic=evidence.diagnostic,
    )


# The strings this core declares for each marker on a published contract. A value that is not
# one of them is a contract this layer cannot read, never a value it guesses at.
_PROVENANCE: Tuple[str, ...] = ("env-mandatory", "reserved")
_TERMINAL_KINDS: Tuple[str, ...] = ("none", "score", "abort")
_TEMPLATE_ROLES: Tuple[str, ...] = ("system", "user")


def _wire_field(value: Any) -> Any:
    """One field of a published contract, as the wire carries it. **Never the value handed in.**

    The caller does the reading, exactly once, and hands the value here; this only renders it.
    That split is the point: a field recovered by *reading it again* would be a second question
    to the same object, and the whole normalization exists because nothing obliges an env to
    answer two questions the same way.

    A field that will not render **raises**, and the caller turns that into an attributed
    contract refusal. Handing the value back instead was the last of the check-then-reuse sites,
    and the reason it looked defensible was that a layer above would refuse a contract it could
    not serialize. That is only true when there *is* a layer above: this class is also the
    transport-independent engine ``evaluate()`` and ``run_stdio()`` drive directly, and there the
    kept object was the whole story. Measured on that path: a schema that refused its first walk
    stayed the env's, its ``__deepcopy__`` handed back itself once and a *different* schema the
    next time, and the episode advertised ``answer`` as an integer through ``describe()`` while
    its seal enforced a string. The agent sends exactly what it was shown and is told it is
    invalid, which is the publish-one-enforce-another failure this whole normalization exists to
    close, arrived at through the one field that was allowed to skip it.

    A rendering that comes back as a different JSON *type* than the field declares is kept, and
    kept as the **rendering**. There is plain data in hand, nothing of the env's is retained, and
    what is wrong with it is exactly what the layer above checks: a budget that is not a whole
    number, framing that is not text. Refusing it here as well would take that judgement from the
    only code that knows whether the task is servable, and there is nothing left to protect by
    doing so, because the value that reaches the contract is this module's rendering either way.
    That is what makes this one return rather than raise: the difference between a value this
    layer could not obtain and a value it obtained and does not like."""
    try:
        wire = json.loads(json.dumps(value, allow_nan=False))
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 — refused, and the refusal names what refused it
        # Rendered into the message rather than left to `__cause__`, because the refusal this
        # becomes is what an operator reads, and a refusal that does not say which value the env
        # published points at nothing.
        raise ValueError(
            f"a published contract field could not be put on the wire: {_rendered_failure(exc)}"
        ) from exc
    return wire


# One value per declared JSON type, and a second ladder for the numeric and text bounds a schema
# most often puts on them. These are only ever *candidates*: nothing here is published, enforced or
# retained, and a candidate exists solely to be offered to the env's own schema as proof that some
# call could satisfy it.
_LEAST: Dict[str, Any] = {
    "string": "x",
    "integer": 0,
    "number": 0,
    "boolean": False,
    "array": [],
    "object": {},
    "null": None,
}
_AGAIN: Dict[str, Any] = {"string": "", "integer": 1, "number": 1, "array": [{}], "object": {"x": 1}}


def _candidate_value(declaration: Any, ladder: Dict[str, Any]) -> Any:
    """A value for one declared property, taken from what the property itself says about it."""
    if not isinstance(declaration, dict):
        return ladder.get("string", "x")
    if "default" in declaration:
        return declaration["default"]
    if "const" in declaration:
        return declaration["const"]
    choices = declaration.get("enum")
    if isinstance(choices, list) and choices:
        return choices[0]
    declared = declaration.get("type")
    kinds = declared if isinstance(declared, list) else [declared]
    for kind in kinds:
        if isinstance(kind, str) and kind in ladder:
            return ladder[kind]
    return ladder.get("string", "x")


def _candidate_calls(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Objects a transport could send, built out of what this schema says it wants."""
    candidates: List[Dict[str, Any]] = [{}]
    const = schema.get("const")
    if isinstance(const, dict):
        candidates.append(const)
    choices = schema.get("enum")
    if isinstance(choices, list):
        candidates.extend(entry for entry in choices if isinstance(entry, dict))
    # `allOf` branches carry their share of the same object, so their `required` and `properties`
    # are read beside the root's rather than instead of them.
    scopes: List[Dict[str, Any]] = [schema]
    branches = schema.get("allOf")
    if isinstance(branches, list):
        scopes.extend(branch for branch in branches if isinstance(branch, dict))
    required: List[str] = []
    properties: Dict[str, Any] = {}
    for scope in scopes:
        names = scope.get("required")
        if isinstance(names, list):
            required.extend(
                name for name in names if isinstance(name, str) and name not in required
            )
        declared = scope.get("properties")
        if isinstance(declared, dict):
            for key, declaration in declared.items():
                if isinstance(key, str):
                    properties.setdefault(key, declaration)
    if required:
        for ladder in (_LEAST, _AGAIN):
            candidates.append(
                {name: _candidate_value(properties.get(name), ladder) for name in required}
            )
    return candidates


# How far into a composition an exclusion proof will look. Bounded because this is a proof about
# structure and not a solver: past a few levels the honest answer is that nothing was established.
_EXCLUSION_DEPTH = 3


def _admits_every_object(schema: Any) -> bool:
    """Whether ``schema`` accepts **every** object there is.

    The only question asked of a ``not`` subschema, because ``not X`` excludes every object exactly
    when ``X`` accepts every object. Answered only for the shapes where it is plain: a schema with
    no constraints at all, and one whose sole constraint is a ``type`` that includes ``object``.
    Anything else (``{"not": {"type": "object", "required": ["a"]}}``, say) excludes only *some*
    objects and is no proof of anything."""
    if schema is True:
        return True
    if not isinstance(schema, dict):
        return False
    if not schema:
        return True
    declared = schema.get("type")
    if len(schema) == 1 and declared is not None:
        kinds = declared if isinstance(declared, list) else [declared]
        return "object" in kinds
    return False


def _excludes_every_object(schema: Any, depth: int = 0) -> bool:
    """Whether ``schema`` **provably** admits no object at all.

    A proof, not a search. Every branch below is a structural fact about the document that no
    instance can talk this layer out of, which is what makes a refusal built on it sound:

    - ``False``, the schema that accepts nothing;
    - a ``type`` that is present and does not include ``object``;
    - a ``const`` that is not an object, since the only instance it accepts is that value;
    - an ``enum`` with no object among its members, including the empty one;
    - a ``not`` whose subschema accepts every object (:func:`_admits_every_object`);
    - an ``allOf`` with a branch that is itself provably object-excluding, since every branch has
      to hold at once;
    - an ``anyOf`` or ``oneOf`` whose branches are **all** provably object-excluding, since at
      least one has to hold.

    The composition cases recur to :data:`_EXCLUSION_DEPTH` and no further. Anything this cannot
    prove is left unproven, which the caller treats as no finding at all: general satisfiability is
    not decidable by inspection, and a layer that guessed at it would refuse working contracts."""
    if schema is False:
        return True
    if not isinstance(schema, dict):
        return False
    declared = schema.get("type")
    if declared is not None:
        kinds = declared if isinstance(declared, list) else [declared]
        if "object" not in kinds:
            return True
    if "const" in schema and not isinstance(schema["const"], dict):
        return True
    choices = schema.get("enum")
    if isinstance(choices, list) and not any(isinstance(entry, dict) for entry in choices):
        return True
    if depth >= _EXCLUSION_DEPTH:
        return False
    if "not" in schema and _admits_every_object(schema["not"]):
        return True
    branches = schema.get("allOf")
    if isinstance(branches, list) and any(
        _excludes_every_object(branch, depth + 1) for branch in branches
    ):
        return True
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list) and branches and all(
            _excludes_every_object(branch, depth + 1) for branch in branches
        ):
            return True
    return False


def _admits_a_call(name: Any, validator: Any, schema: Dict[str, Any]) -> Optional[bool]:
    """What this layer can establish about whether some call satisfies ``schema``.

    Three outcomes, and each one claims only what it proves:

    - ``True``: a candidate call validated. A witness proves satisfiability outright.
    - ``False``: no witness, **and** the document provably excludes every object
      (:func:`_excludes_every_object`). Only a structural proof may refuse.
    - ``None``: neither. The schema passes exactly as it did before this gate existed.

    The ladder is a **prover and never a refuser**, which is the correction this function exists
    to carry. Treating an exhausted ladder as proof of unsatisfiability read absence of a witness
    as a witness of absence, and the ladder is a handful of examples: an ordinary
    ``{"type": "integer", "minimum": 2}`` argument, a ``minLength``, a ``pattern``, bounds away
    from zero and one, a nested ``anyOf``, even an annotation ``default`` that does not satisfy its
    own schema, were all refused, on every advertised tool of every episode, so envs that had
    always worked stopped opening. General satisfiability is not decidable here and the fix is to
    stop claiming it, not to keep guessing at it.

    **A validator that cannot run is a different finding and it does refuse.** An exception out of
    ``is_valid`` is not an instance being rejected: it is the schema failing to resolve or compile
    (a dangling ``$ref``, a pattern that will not build), and it will do that on every real call.
    Round seventeen left that inconclusive on the grounds that the witness search had established
    nothing, which conflated two things: the *search* is inconclusive, and the *machinery* is
    proven unusable. This layer's claim is that it can enforce the published contract, and it
    cannot enforce a document it cannot execute, so it raises and the caller turns it into the
    attributed refusal with the validator's own failure as the cause. Measured before the change:
    a dangling ``$ref`` was admitted, raised on the terminal call, and was filed as a task-local
    lost call, so the run was never stopped and the same unusable env served the next task."""
    checker = validator(schema)
    for candidate in _candidate_calls(schema):
        try:
            if checker.is_valid(candidate):
                return True
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:  # noqa: BLE001 (a schema that cannot run cannot be enforced)
            raise ValueError(
                f"the advertised tool {name!r} publishes a schema this layer cannot enforce: "
                f"validating a call against it {_rendered_failure(exc)}"
            ) from exc
    return False if _excludes_every_object(schema) else None


def _core_schema(name: Any, schema: Any) -> Any:
    """One advertised tool's argument schema, proved to be a schema this episode can **enforce**.

    Rendering it proved only that it is JSON. A contract says more than that: this schema is what
    a terminal call is validated against before the episode seals, and what the transport
    advertises as the arguments of a tool, so a document that is not a schema at all is a tool
    nobody can call correctly. Measured, with the defect published consistently by every instance
    so no drift check applies: ``{"type": "definitely-not-a-json-schema-type"}`` dispensed, the
    exactly correct submission came back ``{"error": "tool schema is invalid",
    "validation_error": true}``, the run was never stopped, and an orderly shutdown filed
    ``closure="drained"`` with ``score.success=False`` and no diagnostic. A right answer recorded
    as a clean scored loss, with nothing anywhere saying the env was at fault.

    Three things are checked, and each one is a way a call cannot succeed:

    - it is a JSON **object**, because a tool takes named arguments and the transport advertises
      this document as their shape;
    - it passes ``check_schema`` on the very validator the seal will use
      (:func:`~jsonschema.validators.validator_for`, the same selection ``jsonschema.validate``
      makes for itself), so what is proved here is what is enforced there and not something near
      it;
    - the root permits an object instance, because arguments arrive as one. A root that declares
      any other type refuses every call the transport can make, and does it as a *validation*
      error, so the agent is told its correct answer is malformed for as long as it retries;
    - and **some object actually satisfies it**, shown by exhibiting one.

    That last one is the check the other three only looked like. Reading the root's ``type`` says
    what the root declares and nothing about what the rest of the document excludes:
    ``{"not": {"type": "object"}}``, a ``const`` or ``enum`` of non-objects, and an ``allOf`` whose
    branches contradict each other are all valid schemas with no ``type`` at the root that refuse
    every object there is. Measured on each of them, with the defect published by every instance so
    no drift check applies: the exactly correct submission came back ``validation_error``, the run
    was never stopped, and an orderly shutdown filed ``closure="drained"`` with
    ``score.success=false`` and no diagnostic, which is precisely the outcome the first three
    checks exist to prevent.

    **Proved by exhibition rather than by analysis**, which is the design choice here and is worth
    stating. Deciding satisfiability by inspecting keywords means implementing a solver for a
    language whose applicators compose arbitrarily, and the alternative offered, restricting
    contracts to a syntactic subset (no ``not``/``allOf``/``anyOf``/``oneOf``/``const``/``enum`` at
    the root), refuses schemas that are perfectly serviceable, such as an ``anyOf`` that admits
    objects among other shapes. So this asks the schema itself: a small ladder of candidate calls
    is built out of what the document says it wants (its own ``const``/``enum`` objects, and its
    ``required`` names filled from each property's ``default``, ``const``, ``enum`` or declared
    type), and one that validates *is* the proof. Nothing about a candidate is published,
    enforced or retained.

    What that trades is completeness in the safe direction: a schema satisfiable only by an object
    this ladder does not construct is refused, loudly and in the env's name, rather than dispensed
    as a task nobody can finish. Every env registered in this repo that constructs offline was
    checked against it and every advertised schema is admitted.

    A root that declares no type at all is left alone by the third check: it accepts an object,
    which is the whole of what that check needs to know, and inventing a constraint the env did
    not publish is the guess this layer must not make. The fourth check is what then establishes
    that the rest of the document agrees."""
    if not isinstance(schema, dict):
        raise ValueError(
            f"the advertised tool {name!r} publishes arguments that are not a JSON object: "
            f"{schema!r}"
        )
    validator = validator_for(schema)
    try:
        validator.check_schema(schema)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 (any refusal here is the contract's, not a crash)
        raise ValueError(
            f"the advertised tool {name!r} publishes arguments that are not a valid JSON Schema: "
            f"{_rendered_failure(exc)}"
        ) from exc
    declared = schema.get("type")
    if declared is not None:
        kinds = declared if isinstance(declared, list) else [declared]
        if "object" not in kinds:
            raise ValueError(
                f"the advertised tool {name!r} publishes a schema whose root is {declared!r} "
                "rather than an object, so no call this transport can make would satisfy it"
            )
    if _admits_a_call(name, validator, schema) is False:
        raise ValueError(
            f"the advertised tool {name!r} publishes a schema that no request this transport "
            "can make would satisfy, so every call it advertises would be refused as the "
            "caller's own mistake"
        )
    return schema


def _typed(value: Any, kind: type, field: str, *, optional: bool = False) -> Any:
    """A rebuilt field whose **declared type this layer depends on**, or a contract refusal.

    The rebuild is a ``model_construct`` on purpose, so a value the models would have rejected is
    carried rather than silently corrected: it is the env's published contract, and whether a task
    is *servable* belongs to whoever can tell. That reasoning holds for a field this layer only
    passes on. It does not hold for one this layer *runs on*, and the difference had become a
    disagreement between two supported surfaces: a stream refused a boolean budget while the
    transport-independent engine enforced it, so ``horizon=True`` ended the task after one
    ordinary call and filed a ``FINALIZED`` record with ``correct=False`` and no
    ``finalize_error``, which is an env contract defect wearing the shape of an agent's wrong
    answer. Where a value decides how this episode dispatches, frames or records, its declared
    type is part of the contract this layer is enforcing, so it is checked here, once, in the
    env's name, and every surface refuses the same contracts.

    ``type(value) is kind`` rather than ``isinstance``, for :func:`_declared`'s reason and for one
    more that bites here: ``bool`` is a subclass of ``int``, so an ``isinstance`` budget check
    reads ``True`` as a budget of one step. Checked on the **rendering**, where every value is
    already an exact builtin, so a subclass cannot answer for itself either."""
    if optional and value is None:
        return value
    if type(value) is not kind:
        raise ValueError(
            f"a published contract carries {field} this layer cannot enforce as published: "
            f"expected {kind.__name__}, got {value!r}"
        )
    return value


def _core_manifest(manifest: ToolManifest) -> ToolManifest:
    """One advertised tool, rebuilt as this module's own model out of fields read once each.

    A name is identity here — the score terminal is found by looking this episode's own key up
    against the name a call arrives under — so a ``str`` subclass answering that comparison its
    own way is a terminal call dispatched as an ordinary step, sealing nothing and scoring
    nothing, on the tool the endpoint advertised as the way to finish the task.

    The markers are not rendered but *matched*, and ``terminal_kind`` least optionally of all:
    this episode decides whether a tool is the scoring terminal by comparing that value against
    ``"score"``, so a value that answers the comparison its own way is a ``score`` terminal the
    serve layer never sees. It slips past the constructor's fail-loud check, leaves
    ``_seal_enabled`` False, and routes an env's advertised authoritative scoring through the
    legacy trajectory path, which is the exact downgrade that check exists to refuse. A marker
    that is not one of the strings this core declares is a contract it cannot read, and it says
    so rather than guessing."""
    provenance = _declared(manifest.provenance, _PROVENANCE)
    terminal_kind = _declared(manifest.terminal_kind, _TERMINAL_KINDS)
    if provenance is None or terminal_kind is None:
        raise ValueError("an advertised tool carries a marker this core does not declare")
    # The name is the dispatch key (this episode looks a call up against it, the transport
    # registers it, and a trace row records it as the tool that ran) and the description is the
    # instruction content the agent reads, so both are types this layer runs on: see `_typed`.
    name = _typed(_wire_field(manifest.name), str, "an advertised tool name")
    return ToolManifest.model_construct(
        name=name,
        description=_typed(
            _wire_field(manifest.description), str, f"a description for tool {name!r}"
        ),
        # Rendered, then proved to be a schema this episode can enforce: see `_core_schema`.
        input_schema=_core_schema(name, _wire_field(manifest.input_schema)),
        provenance=provenance,
        terminal_kind=terminal_kind,
    )


def _core_template(template: ReferenceTemplate) -> ReferenceTemplate:
    """One advisory template, rebuilt the same way, with its role matched rather than rendered."""
    role = _declared(template.role, _TEMPLATE_ROLES)
    if role is None:
        raise ValueError("a reference template carries a role this core does not declare")
    return ReferenceTemplate.model_construct(
        role=role,
        template=_wire_field(template.template),
        variables_schema=_wire_field(template.variables_schema),
    )


def _core_terminals(tools: List[ToolManifest]) -> None:
    """The two rules that are about the manifest **collection** rather than any one tool, checked
    on the rebuilt tools: at most one ``score`` terminal, and ``abort`` exactly on ``terminate``.

    :class:`TaskSpec` states both as a model validator, and every contract built the ordinary way
    passes through it. The rebuild does not: it is a ``model_construct``, deliberately, so that a
    field the models would have rejected still reaches the layer that can tell whether the task is
    servable. That skips the validators as well, and these two say something no per-field render
    can: pydantic models are mutable and do not validate on assignment, so an env can build a
    valid spec, hand it to the validator, and then move a marker before returning it.

    Both rules decide how a call is *dispatched*, which is why they cannot be left to the layer
    above. A second ``score`` terminal turns an ordinary mid-episode tool into a sealing one while
    the framing still describes it as ordinary: measured, a call to ``noop`` sealed the episode
    and filed a clean scored row with ``success=False``, so an agent's ordinary action became its
    terminal wrong answer. And ``abort`` is honoured by name at runtime, not by marker, so a
    marker that disagrees advertises a stop tool that does not stop or hides the one that does.

    Enforced rather than revalidated, because ``model_validate`` on the rebuilt data would also
    refuse the wrong-*typed* fields this layer deliberately carries through (a fractional budget,
    a boolean one, framing that is not text). Those are refused one layer up, by the code that
    knows whether the task is servable, and that division is what the rebuild exists to preserve.
    These two rules are checked here because there is no layer above that checks them at all.

    Every operand is the core's: the kinds are this module's own literals (:func:`_declared`) and
    the names are renderings, so no comparison here dispatches through an env's ``__eq__``."""
    score = [manifest.name for manifest in tools if manifest.terminal_kind == "score"]
    if len(score) > 1:
        raise ValueError(
            f"a published contract may advertise at most one `score` terminal, got {score!r}"
        )
    for manifest in tools:
        name = manifest.name
        kind = manifest.terminal_kind
        if name == TERMINATE_TOOL_NAME and kind != "abort":
            raise ValueError(
                f"the reserved {TERMINATE_TOOL_NAME!r} tool must be advertised with "
                f"terminal_kind='abort', got {kind!r}"
            )
        if kind == "abort" and name != TERMINATE_TOOL_NAME:
            raise ValueError(
                f"terminal_kind='abort' is reserved for the {TERMINATE_TOOL_NAME!r} tool, not "
                f"{name!r}"
            )


def _core_identity(published: Any, requested: Optional[str]) -> Any:
    """The task the contract names, reconciled with the task this episode was asked for.

    ``task_id`` looked like a publication and is not one. This episode is *handed* the identity
    (``describe(resolved_task)``), writes it on every trace row it appends, and the public
    ``evaluate()`` then reads the published field back off ``describe()`` and uses it to select
    those rows. So a contract that names a different task is a key that matches nothing this run
    wrote: measured through public ``evaluate()`` with a spec mutated to
    ``task_id=["not-the-trace-id"]``, the terminal row was on disk with ``task_id="0"`` and
    ``correct=true`` while the call returned ``terminated=False`` with no feedback at all. An
    earned verdict discarded, and a run that finished reported as one that never ended.

    That makes it a field this layer *runs on* rather than one it passes on, which is the
    distinction the budget and the framing already turn on, so it is settled the same way and in
    the same place. Exact identity, not equality: the requested value is this module's own string
    (or ``None`` when a random default picked a task the env does not index), and what may be
    published is that same string, so a value that merely compares equal to it is refused for
    :func:`_declared`'s reason.

    **``None`` is not a wrong answer, it is no answer**, and it stays admitted. The consumer reads
    an absent id as no filter at all rather than as a key that selects nothing, so a contract that
    names no task claims nothing and cannot mis-select: an env that hand-builds its contract and
    leaves the field alone reads back exactly as it always did. What is refused is a contract that
    names a *different* task, in either direction, because naming one where this episode was asked
    for none selects rows this run never wrote, which is the same empty answer from the other
    side."""
    if published is None:
        return published
    if requested is not None and type(published) is str and published == requested:
        return published
    raise ValueError(
        "a published contract names a different task than the one this episode was asked to "
        f"serve: asked for {requested!r}, published {published!r}"
    )


def _core_spec(spec: TaskSpec, task_id: Optional[str]) -> TaskSpec:
    """The published contract rebuilt as **this module's own** :class:`TaskSpec`, out of every
    field read exactly once.

    Normalizing the values was not enough, because the object holding them was still the env's.
    ``describe`` hands out a copy per reader and the copy was taken with ``spec.model_copy``,
    which is a method: a subclass overriding it can return a *different* contract every time and
    never raise at all. Measured, with a subclass that renders perfectly and rewrites one field
    per copy: the episode stored one contract, the first reader was shown another, the second
    reader a third. Containment answers a copy that fails; nothing answers a copy that succeeds
    and lies. So the class is not kept either — what comes out is built here, and the copies
    readers get are pydantic's own method on this module's own model.

    Read once, and only the product retained. A field whose read *raises* has no value to
    retain, so it is a contract this layer cannot read and the caller refuses it; a field that
    reads but will not render is kept exactly as it was read, for the layer above to refuse if it
    cannot use it. Built with ``model_construct``, so a value the models would have rejected is
    still carried rather than silently corrected: it is the env's published contract, and
    refusing it belongs to whoever can tell whether the task is servable. The two rules that
    ``model_construct`` skips and nobody else checks are re-stated here (:func:`_core_terminals`),
    because they decide how this episode dispatches a call rather than whether a task is
    servable."""
    tools = [_core_manifest(manifest) for manifest in spec.tools]
    _core_terminals(tools)
    templates = [_core_template(template) for template in spec.reference_templates]
    return TaskSpec.model_construct(
        env_name=_wire_field(spec.env_name),
        # The task this episode is serving, and not a second opinion about it: see
        # `_core_identity`.
        task_id=_core_identity(_wire_field(spec.task_id), task_id),
        # The framing an agent is handed and the budget this episode enforces: both are values
        # this layer runs on rather than passes on, so both are held to the type the contract
        # declares (see `_typed`). Everything beside them is carried as rendered.
        instructions=_typed(_wire_field(spec.instructions), str, "instructions"),
        tools=tools,
        reference_templates=templates,
        horizon=_typed(_wire_field(spec.horizon), int, "a horizon", optional=True),
        contract_version=_wire_field(spec.contract_version),
    )


class TaskContractError(RuntimeError):
    """An env published a task contract this serve layer cannot take *as* a contract.

    Its own class, because the layer above has to tell it from every other way opening an episode
    can fail. A task that could not be loaded is about that task; a contract that cannot be
    normalized or snapshotted is about the **env**, so it will fail identically on the next task
    of that env in the queue and on every one after it. A stream reads that difference: it stops
    the run for this and refuses only the dispense for the rest, which is the same line
    :meth:`TaskStream._require_published_manifest` already draws for a manifest it cannot compare.

    A ``RuntimeError`` so an existing caller that catches one still catches this."""


_CARRIED_REFUSAL = "__shogym_contract_refusal__"


def contract_refusal(exc: BaseException) -> Optional[TaskContractError]:
    """The contract refusal ``exc`` *is*, or the one it is carrying, or ``None``.

    A refusal can be discovered and then interrupted before it reaches whoever classifies it:
    :meth:`ServedEpisode.open_env` releases the sessions and the env it took ownership of before
    re-raising, and a caller cancelled during that release gets its cancellation, correctly,
    while the refusal that cancellation replaced was a fact about the env that the layer above
    still has to act on. The finding is older than the interruption, so it travels with it.

    This is the reading half of that; :func:`_carrying` is the writing half. A caller classifies
    with this instead of ``isinstance`` and gets both cases at once."""
    if isinstance(exc, TaskContractError):
        return exc
    carried = getattr(exc, _CARRIED_REFUSAL, None)
    return carried if isinstance(carried, TaskContractError) else None


_claimed_releases: "Set[asyncio.Future[None]]" = set()


def _claimed(coro: "Coroutine[Any, Any, None]") -> "asyncio.Future[None]":
    """Run ``coro`` as a task this module keeps alive until it finishes.

    A task is the only thing a cancellation of the *waiter* cannot end, which is the whole reason
    the release below is one. The set is what makes that true rather than probable: the event loop
    holds only a weak reference to a running task, so a release nobody is awaiting any more is a
    release that may simply stop existing, and the resource it was letting go of stays held. The
    entry is dropped when the task completes."""
    task: "asyncio.Future[None]" = asyncio.ensure_future(coro)
    _claimed_releases.add(task)
    task.add_done_callback(_claimed_releases.discard)
    return task


async def _release_setup(opened: List[MCPSession], env: Any) -> None:
    """Let go of everything a failed :meth:`ServedEpisode.open_env` had taken: the sessions it
    opened, then the env whose ownership transferred at the call.

    **Best-effort against every failure, each release on its own**, so one that will not close
    cannot keep the next from being tried and none of them can replace the setup failure that is
    already on its way out. ``CancelledError`` is contained here like any other: nothing cancels
    this task, so a cancellation observed inside it was raised by the env's own ``close`` and is
    that env failing, not a request against this release (see :func:`_must_propagate`, whose rule
    this is). A caller's cancellation is delivered to whoever is awaiting this task and never to
    the task itself."""
    for session in opened:
        try:
            await session.close()
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException:  # noqa: BLE001 (a session that will not close still ends here)
            pass
    try:
        await env.close()
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException:  # noqa: BLE001 (an env that will not close still ends here)
        pass


def _carrying(exc: BaseException, refusal: Optional[TaskContractError]) -> BaseException:
    """Attach a discovered refusal to the exception that is about to replace it, and return it.

    The classification rule this module holds everywhere is that a classification of something
    already discovered runs before anything cancellable. A release that has to happen cannot be
    moved above the discovery, so the discovery is carried across it instead."""
    if refusal is not None and not isinstance(exc, TaskContractError):
        try:
            setattr(exc, _CARRIED_REFUSAL, refusal)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException:  # noqa: BLE001 — an exception that will not carry it still ends the call
            pass
    return exc


def _refused(env_name: Any, exc: BaseException) -> TaskContractError:
    """The one refusal every contract failure is reported as.

    An exception raised out of ``__init__`` and through ``open_env`` carries nothing naming the
    env that could not publish a contract, and nothing the layer above can classify either, so
    the run goes on retrying an env that cannot answer and closes clean over the queue it never
    served. So the refusal is this layer's, it names the env, it says which kind of failure it is,
    and the env's failure is attached as the cause rather than being the whole answer."""
    return TaskContractError(
        f"env {_described(lambda: repr(env_name))} published a task contract this episode "
        f"cannot take as a contract: {_rendered_failure(exc)}"
    )


def _core_contract(spec: TaskSpec, env_name: Any, task_id: Optional[str]) -> TaskSpec:
    """The published contract as this module's own object, under one refusal.

    Every read of the env's contract happens inside this: the fields, the two collections, and
    the iteration of them. Reading is the env's code as much as serializing is, and the read that
    *gets* the tools to normalize was once outside the guard, so an env whose collection would
    not answer left normalization as its own bare exception, past anything the layer above could
    classify.

    A field this cannot read **or** render is refused here rather than left for the layer above.
    Those were once two answers: an unreadable field left nothing to hand on, while an
    unrenderable one still had a value somebody upstream could judge. The second half of that no
    longer holds, because keeping the env's object is exactly the retention every other boundary
    in this module has given up, and because there is not always a layer above: this class is the
    transport-independent engine too. So the rule is one rule. An empty or borrowed stand-in for
    any part of a contract would publish a task this episode cannot honestly enforce, and the
    tool collection shows why that is not survivable: a task with no tools and no scoring
    terminal, run down the legacy path, recording whatever came out.

    What comes back is therefore built entirely out of the plain data the successful walks
    produced. There is no copy step after it and there must not be one: a copy of an env's value
    is the env's code, and a call to ``__deepcopy__`` that returns *something* is not proof it
    returned a detached something."""
    try:
        return _core_spec(spec, task_id)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 — an env may not refuse an episode unattributed
        raise _refused(env_name, exc) from exc


def _snapshot(spec: TaskSpec, env_name: Any) -> TaskSpec:
    """One reader's copy of the contract this episode holds.

    Called only on the spec this module built (see :func:`_core_spec`), whose every value is
    plain data, so ``model_copy`` is pydantic's own method walking nothing an env wrote. That is
    what makes a copy meaningful here at all: detachment is established by having *rebuilt* the
    contract out of rendered data, never by a call to a foreign ``__deepcopy__`` returning
    without raising, which proves only that it returned. It is contained anyway, because a
    refusal this layer makes is the serve layer's and names the env: see :func:`_refused`."""
    try:
        return spec.model_copy(deep=True)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 — an env may not refuse an episode unattributed
        raise _refused(env_name, exc) from exc


def _score_schemas(spec: TaskSpec, env_name: Any) -> Dict[str, Dict[str, Any]]:
    """The score-terminal tool name mapped to its advertised outer schema, off the published
    contract, under the same refusal.

    Read from the rebuilt contract, so the collection and both markers are this module's own and
    the comparison below is between two exact strings. What can still be the env's is a *name*
    that would not render, and a name is used here as a dictionary key, which calls ``__hash__``
    on it. So the read stays inside the guard, and a contract whose tool names cannot be keyed is
    refused like any other this layer cannot read."""
    try:
        return {m.name: m.input_schema for m in spec.tools if m.terminal_kind == "score"}
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 — an env may not refuse an episode unattributed
        raise _refused(env_name, exc) from exc


def _promised_finalize(env: Any, env_name: Any) -> Any:
    """The terminal hook an env promised, **read** under the refusal that owns the promise.

    Asked for only once a published ``score`` terminal has been found, because that is what makes
    it part of the contract: an env that advertises no scoring terminal promises no finalizer, and
    a layer that fails closed does not go looking for hooks nobody undertook to provide.

    The read itself is the env's code. ``getattr`` with a default answers for an attribute that is
    *absent*; an attribute that raises is a different event and it came straight out of the
    constructor with nothing naming the env and nothing the layer above could classify, so a run
    met the identical failure on the next pull and closed clean over a queue it never served. The
    check on what comes back was already the attributed refusal (the promise below); the read that
    reaches it is now inside the same one."""
    try:
        return getattr(env, "finalize", None)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:  # noqa: BLE001 (an env may not refuse one unattributed)
        raise _refused(env_name, exc) from exc


@dataclass
class CallResult:
    """The outcome of one tool call: the tool's functional ``content`` (the observation the
    harness needs), the ``meta`` sidecar (feedback + terminate flag), and whether the episode
    is now over."""

    content: str
    meta: Dict[str, Any]
    terminated: bool
    # True when this call arrived after the episode had already ended and was answered with a
    # tombstone: nothing was dispatched, nothing was sealed, and ``terminated`` reports the
    # episode's state rather than anything this call did. A caller that attributes the ending
    # to the call it made — "this request is what aborted the task" — has to tell the two
    # apart, because every call after the first terminal is answered this way.
    tombstoned: bool = False


class ServedEpisode:
    """One episode of one env, driven by external tool calls.

    Construct via :meth:`start` (it loads the task and opens the tool sessions), then call
    :meth:`call` per tool invocation until :attr:`terminated`. :meth:`describe` returns the
    task contract to hand the harness at setup.
    """

    def __init__(
        self,
        env,
        env_name: str,
        task_id: Optional[str],
        session_id: str,
        task: Dict[str, Any],
        sessions: Dict[str, MCPSession],
        opened: List[MCPSession],
        trace_path: Optional[Union[str, Path]],
        spec: TaskSpec,
        finalize_deadline: Optional[float] = None,
    ) -> None:
        self._env = env
        self._env_name = env_name
        self._task_id = task_id
        self._session_id = session_id
        self._task = task
        self._sessions = sessions  # advertised tool name -> session
        self._opened = opened  # every session opened, for teardown
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._trajectory: Trajectory = []
        self._step = 0
        self._terminated = False
        # The terminal step's feedback in wire form (inference + episode), retained so
        # the in-process `evaluate()` can report the score without a trace file. Same
        # list `result_from_trace` reconstructs from the terminal row.
        self._terminal_feedback: List[Dict[str, Any]] = []
        # Serialize calls: one episode is a single sequential trajectory. `call()`
        # mutates shared step/trajectory/terminated state across an `await`, so
        # concurrent MCP requests on this session must not interleave. This same lock is the
        # lifecycle-state lock: seal, evidence commit, close-race arbitration, and teardown all
        # take it.
        self._lock = asyncio.Lock()

        # The task contract, described once and snapshotted here. Everything that publishes this
        # episode's contract or enforces it reads THIS object: `describe()` hands it out, and the
        # score schemas below — what a terminal call's arguments are validated against — are the
        # schemas in it. Describing again per reader is what that replaces: `Env.describe` is
        # ordinary env code and nothing obliges two calls to agree, so a stateful env could
        # publish one contract to whoever frames the agent and enforce another when the agent
        # acts on it. The agent then sends the arguments it was told to send, the seal refuses
        # them against a schema it was never shown, and the task is recorded as a wrong answer.
        #
        # Copied rather than kept, because `describe` hands back the env's own object and an env
        # is free to hold on to it: a snapshot the publisher can still reach is not a snapshot.
        #
        # And normalized, not merely copied, because a copy of an env's object is still the env's
        # object. What a server advertises is this contract *as JSON*, and what the terminal
        # transaction refuses a call against is the schema in here — so the two have to be the
        # same value, not two renderings of one. A JSON scalar has subclasses, and the models do
        # not validate on assignment, so a `const` that is a `str` subclass whose `__eq__` answers
        # false is advertised as ordinary text and matches nothing the agent can send: the agent
        # sends exactly what it was shown, the seal refuses it, and the row says it answered
        # wrong. The round trip strips the subclass and leaves the value the wire carries.
        #
        # Normalized per *field*, and **rebuilt**, which is the part that took the longest to get
        # right. Copying was the step that ran code the env wrote: `__deepcopy__` belongs to
        # whatever object is being copied, so a copy of the env's own spec let an env decide
        # whether an episode could be opened at all, and a copy that *returned* was taken as
        # proof it had returned something detached. It is not. A value can hand back itself once
        # and something different the next time without ever raising, and then `describe()`
        # advertises one contract while the seal enforces another: the agent sends exactly what
        # it was shown and the row says it answered wrong.
        #
        # So detachment is established by having rebuilt the whole contract out of the plain data
        # each successful walk produced, and by nothing else. A field this layer cannot read or
        # put on the wire has no rebuilt form, and it is refused rather than borrowed: deferring
        # that to whoever could compare the contract only ever worked where such a layer existed,
        # and this class is the transport-independent engine as well. See `_core_contract`, which
        # holds every read and every rendering to one attributed refusal.
        self._spec = _core_contract(spec, env_name, task_id)

        # ----- seal-before-verdict lifecycle -----
        # The score-terminal tool name -> its advertised outer schema, read off the *published
        # contract* above (the manifest is the enforcement point, not a marker scan). At most one
        # tool is `score`; empty for every non-score env, so `call()` never leaves its legacy
        # path there. Under the same refusal as the contract itself: the collection is core's
        # now, but a tool name is used here as a dictionary key, which calls `__hash__` on it.
        self._score_schemas: Dict[str, Dict[str, Any]] = _score_schemas(self._spec, env_name)
        # `finalize` is the env's terminal hook (None unless the env overrides it), read only for
        # an env whose contract promised one, and read inside the same attributed refusal that
        # judges the promise: the read is the env's own code (see `_promised_finalize`).
        self._finalize = (
            _promised_finalize(env, env_name) if self._score_schemas else None
        )
        # A published `score` terminal is a promise: this call is authoritatively sealed and
        # finalized. Enforce that promise HERE — the single boundary every served env passes
        # through — not only in the env's construction check. An env that hand-builds its
        # TaskSpec/manifest in `describe()` instead of declaring `score_terminal_tool` never runs
        # that check, and would otherwise slip a score terminal past the serve layer with no
        # callable finalize, silently leaving `_seal_enabled` False and routing its advertised,
        # authoritative scoring through the legacy marker/trajectory path — reopening the
        # grade->read->fix->grade exploit for an env that expected the seal to protect it. Refuse
        # to run, loudly, rather than downgrade.
        #
        # A `TaskContractError` like every other refusal of a published contract, because that is
        # what this is: not a task that could not be loaded but an env whose advertised terminal
        # cannot be honoured, which is equally true of the next task from that env and of every
        # one after it. Raised as a plain `TypeError` it went past the only handler that draws
        # that distinction, so the layer above refused this dispense, kept the position owed, and
        # walked into the identical failure on the next pull against a freshly built env, with a
        # clean close over a queue nothing served.
        if self._score_schemas and not callable(self._finalize):
            name = next(iter(self._score_schemas))
            raise TaskContractError(
                f"env {_described(lambda: repr(env_name))} advertises a `score` terminal "
                f"{_named(name)} but provides no callable finalize() hook; refusing to run its "
                "authoritative scoring through the legacy path (a `score` terminal must seal "
                "and finalize)"
            )
        # After the guard, a score manifest guarantees a callable finalize, so a `score`
        # terminal ALWAYS seals. An env that marks nothing `score` opts out to the unchanged
        # legacy path (`_seal_enabled` False) — the correct behaviour for a non-migrated env.
        self._seal_enabled = bool(self._score_schemas)
        self._state = LifecycleState.OPEN
        # The single in-flight finalization. Created exactly once, at seal; shielded by every
        # awaiter so a client cancellation/close can never abandon or re-dispatch it. At most
        # one per episode.
        self._finalization: Optional["asyncio.Future[CallResult]"] = None
        self._finalization_id: Optional[str] = None
        self._finalization_source: Optional[str] = None
        # The tool that entered the terminal transaction, recorded AT the seal — so a caller
        # whose terminal call was cancelled before its verdict landed can still learn how the
        # episode ended, from the episode itself.
        self._finalization_tool: Optional[str] = None
        # The committed, core-owned terminal evidence (None until FINALIZED).
        self._evidence: Optional[TerminalEvidence] = None
        # Teardown is idempotent and runs exactly once (owned by the finalizer, after
        # evidence; close() routes through the same guard). `_teardown_runs` is a test hook.
        self._torn_down = False
        self._teardown_runs = 0
        # The digest of the terminal call's arguments, taken once at the seal (see
        # `_begin_finalization`) and reused by every record and the trace event that names it.
        self._args_digest: Optional[str] = None
        # Set if a durable-record write ever failed (best-effort persistence — never fatal).
        self._persist_degraded = False
        # The in-flight evaluator task (retained so teardown drains it — see `_run_finalize`).
        self._eval_task: Optional["asyncio.Future[Any]"] = None
        # A background drain+teardown task, used only on the deadline path so the caller gets
        # the fail-closed result AT the deadline while resource cleanup waits for the evaluator.
        self._drain_task: Optional["asyncio.Future[None]"] = None
        # Optional finalize deadline (seconds): the evaluator is awaited up to this bound; on
        # timeout the episode fails closed to a `finalize_error` verdict. None disables it.
        self._finalize_deadline = finalize_deadline
        # The durable finalization store — a local directory of fsync'd JSON records next to
        # the trace (zero user setup). Built lazily only for a seal-enabled env.
        self._store: Optional[FinalizationStore] = None
        if self._seal_enabled:
            self._store = FinalizationStore(
                FinalizationStore.resolve_dir(session_id, self._trace_path)
            )
            # Restart recovery, transport-independent: resolve any finalization records left
            # dangling (SEALED/PENDING) by a crashed prior run to a fail-closed verdict —
            # the evaluator is never re-invoked. Done here at construction (not only in
            # `run_stdio`) so `evaluate()` and every in-process caller get the same guarantee.
            # This episode has not sealed yet, so it owns no record here; only prior/other
            # sessions' dangling records are resolved. Best-effort: a read-only-store I/O error
            # never blocks startup.
            try:
                self._store.recover()
            except OSError:
                pass

    @classmethod
    async def start(
        cls,
        env_name: str,
        *,
        task: Optional[Union[int, str]] = None,
        trace_path: Optional[Union[str, Path]] = None,
        env_config: Optional[Dict[str, Any]] = None,
        finalize_deadline: Optional[float] = None,
    ) -> "ServedEpisode":
        """Build the env, load the task instance, open the essential MCP sessions, and push
        per-episode state into the (in-process) tool servers."""
        return await cls.open_env(
            make(env_name, config=env_config),
            env_name=env_name,
            task=task,
            trace_path=trace_path,
            finalize_deadline=finalize_deadline,
        )

    @classmethod
    async def open_env(
        cls,
        env,
        *,
        env_name: Optional[str] = None,
        task: Optional[Union[int, str]] = None,
        trace_path: Optional[Union[str, Path]] = None,
        finalize_deadline: Optional[float] = None,
    ) -> "ServedEpisode":
        """Start an episode on an **already-constructed** env, which this episode then owns.

        Same contract as :meth:`start` except the caller supplies the env instance instead of
        a name, which lets a caller that serves several episodes at once give each one its own
        env. Ownership transfers: :meth:`close` closes this env, and a failure during setup
        closes it here — so the caller must hand over a *fresh* instance per episode rather
        than a shared one (``Env.close`` ends **every** session the instance tracks, which
        would tear down any sibling episode sharing the instance).
        """
        opened: List[MCPSession] = []
        try:
            # Inside the try, because ownership transferred at the call and `name` is the env's
            # own code: read above it, an env whose name raised was one this method had already
            # taken responsibility for closing and then never closed, which is the one promise
            # the docstring above makes about a setup that fails.
            env_name = env_name if env_name is not None else env.name
            task_idx = int(task) if task is not None else None
            task_data = env.load_task(task_idx)
            # Publish the *resolved* task identity so a random-default episode (task
            # omitted) is still attributable: an env that indexes tasks records the
            # chosen index in task_data (Wordle: "task_idx"), so a `shogym serve wordle_v1`
            # run traces a concrete task rather than null.
            if task is not None:
                resolved_task: Optional[str] = str(task)
            elif "task_idx" in task_data:
                resolved_task = str(task_data["task_idx"])
            else:
                resolved_task = None
            session_id = str(uuid7())

            sessions: Dict[str, MCPSession] = {}
            for spec in env.essential_specs():
                session = await _open_session_for_spec(spec, session_id=session_id)
                opened.append(session)
                for tool_config in await session.list_tools():
                    sessions[tool_config.name] = session
            env.begin_session(session_id, task_data)
            # The one description this episode ever asks for. Everything published about the task
            # and everything enforced on it comes off this single answer — see the snapshot the
            # constructor takes of it.
            #
            # An env that cannot answer it has published no contract at all, which is the same
            # fact about the env that a contract this layer cannot read or copy is, and it will
            # be just as true of the next task from that env. So it is refused as one, and the
            # layer above stops the run on it instead of meeting the identical failure on every
            # pull. That line is drawn around the *contract* and not around everything above it:
            # `load_task`, `essential_specs` and `begin_session` are this task's setup and its
            # session's, they can fail for reasons the next task may not share, and they still
            # cost only this dispense.
            try:
                spec = env.describe(resolved_task)
            except (SystemExit, KeyboardInterrupt):
                raise
            except BaseException as exc:  # noqa: BLE001 — a contract nobody can obtain
                raise _refused(env_name, exc) from exc
            # Construct inside the try so the cleanup below also covers the constructor's own
            # fail-loud guard (a `score` manifest with no callable finalize): the sessions
            # opened above are released before the error propagates.
            return cls(
                env,
                env_name,
                resolved_task,
                session_id,
                task_data,
                sessions,
                opened,
                trace_path,
                spec=spec,
                finalize_deadline=finalize_deadline,
            )
        except BaseException as failure:
            # Setup failed, so no ServedEpisode is returned for the caller to close:
            # release everything here. Close any opened MCP sessions, then close the
            # env (drops per-episode state begin_session may have pushed before
            # raising). Both are best-effort so the original setup error propagates.
            #
            # Best-effort against every raise, `CancelledError` included, which an env's `close`
            # can raise like anything else. Caught as `Exception` it was not best-effort against
            # that one: the cancellation replaced the setup failure as the exception this call
            # raises, leaving the real cause as `__context__` and handing the layer above an
            # unattributed cancellation — the shape it reads as its *own* request — for an env
            # that could not open a task at all and will fail the same way on every task queued
            # behind this one. The release is the decoration here and the failure being released
            # from is the point, so the release is what gives way.
            #
            # The baseline is taken inside the handler, so it asks only about the closes: a
            # cancellation delivered before them is already the exception being re-raised below.
            #
            # And a refusal already discovered travels across that release. The releases below
            # are awaits, so a caller cancelled during one of them gets its cancellation, which
            # is right and stays right. What is not right is that the cancellation *replaces* a
            # contract refusal this call had already found: the layer above classifies what it
            # is handed, so the refusal disappeared, the position stayed owed, and the run closed
            # clean over an env that cannot publish a contract at all. The classification cannot
            # be moved above a release that has to happen, so the finding is carried across it
            # instead. See `contract_refusal`, which is how the layer above reads it.
            #
            # And the release is **claimed** before it is awaited, because a caller cancelled
            # during it may end its own call but may not end the release. Run inline, the release
            # was a sequence of awaits in the caller's task: a cancellation delivered during
            # `env.close()` re-raised out of the middle of it and nothing was left holding the env
            # this method had taken ownership of. Measured, with a refusal raised after
            # `begin_session` and a close that blocks: the stop was latched (right), the caller
            # was cancelled (right), and the env was never closed and its per-episode state never
            # dropped, with no episode returned or registered for anyone to recover it from.
            #
            # So the release runs as its own task and the join is shielded, which is the shape
            # `TaskStream._released` already uses for the same reason. The cancellation reaches
            # the awaiting call and the only owner of these resources runs to completion behind
            # it.
            refusal = failure if isinstance(failure, TaskContractError) else None
            cancellation = _Cancellation()
            release = _claimed(_release_setup(opened, env))
            try:
                await asyncio.shield(release)
            except BaseException as exc:  # noqa: BLE001 — must not mask the setup failure
                if _must_propagate(exc, cancellation):
                    raise _carrying(exc, refusal)
            raise

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def seal_enabled(self) -> bool:
        """True iff this episode runs the seal transaction (the env declares a score terminal
        and a ``finalize`` hook). False for a non-seal env."""
        return self._seal_enabled

    @property
    def sealed(self) -> bool:
        """True once the lifecycle has left OPEN — the ingress gate consults this to tombstone
        post-seal traffic. Always False for a non-seal episode."""
        return self._state is not LifecycleState.OPEN

    @property
    def terminal_feedback(self) -> List[Dict[str, Any]]:
        """The terminal step's feedback (wire form), or ``[]`` until the episode ends."""
        return self._terminal_feedback

    @property
    def terminal_source(self) -> Optional[str]:
        """How the terminal transaction was entered — ``explicit_tool``, ``abort`` or
        ``horizon`` — or ``None`` while the episode is still open.

        Recorded AT the seal, so it is readable even when the call that requested it was
        cancelled before its verdict landed."""
        return self._finalization_source

    @property
    def terminal_tool(self) -> Optional[str]:
        """The tool that entered the terminal transaction (``None`` for an abort, which has no
        tool, and until the episode seals). Recorded at the seal, like :attr:`terminal_source`."""
        return self._finalization_tool

    @property
    def terminal_payload(self) -> Optional[Dict[str, Any]]:
        """The public-safe terminal payload — the same sanitized verdict a terminal call
        returns — or ``None`` until the terminal transaction has committed its evidence."""
        return None if self._evidence is None else self._sanitize_terminal(self._evidence)

    async def wait_finalized(self) -> None:
        """Wait for an in-flight terminal transaction to commit, if there is one.

        A terminal call leaves its finalization running when its own caller is cancelled, so an
        episode can be sealed with its verdict still landing. Anyone who needs to classify the
        outcome must wait for it first — the alternative is reading a not-yet-terminated episode
        and reporting an ending that never happened. The wait is shielded: waiting for the single
        evaluation never cancels it."""
        finalization = self._finalization
        if finalization is None:
            return
        try:
            await asyncio.shield(finalization)
        except Exception:  # noqa: BLE001 — a failed finalization fails closed onto the episode
            pass

    def describe(self) -> TaskSpec:
        """The contract this episode was opened on — the same snapshot for every reader, for as
        long as the episode lives.

        A copy per reader, because the snapshot is also what a terminal call's arguments are
        validated against: handed out directly, whoever framed the agent could rewrite the schema
        the seal enforces, or the tools a server had already registered, from under the episode
        that is running on them.

        A copy of **data**, which is what makes "the same snapshot" true rather than merely
        intended. A deep copy is the env's code whenever it reaches something the env still owns,
        and nothing obliged two runs of that code to agree: whoever framed the agent was shown
        one contract, the check that compares the manifest the next, and the episode enforced the
        one taken at construction — the same publish-one-enforce-another failure the snapshot
        exists to close, arrived at through the copy that was supposed to close it. The whole
        contract is normalized field by field before the snapshot is taken (see
        :func:`_core_spec`), so what this copies is plain data and copying it runs nobody's code.

        Contained for the residue that normalization documents — a tool whose schema will not
        serialize keeps the env's object — so a read that cannot be answered is refused in the
        env's name rather than by its traceback."""
        return _snapshot(self._spec, self._env_name)

    async def call(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> CallResult:
        """Execute one tool call as one step; return its result + feedback sidecar.

        For a non-seal env this is a plain step: dispatch, record, verify. For a seal-enabled
        env, a score-terminal call runs the terminal transaction (validate -> seal ->
        evaluate), and after termination (legacy terminate/horizon) or after a seal, further
        calls are tombstoned with no inward dispatch."""
        # Phase 1: decide and, for a terminal call, seal + spawn the finalization — all under
        # the lock so state transitions are atomic against concurrent ingress.
        async with self._lock:
            # Post-seal tombstone: once the lifecycle has left OPEN, no `tools/call` reaches
            # task/env state. (Read-only `describe`/`tools/list`/resource reads bypass
            # `call()` entirely, so they stay readable — see the ingress gate in server.py.)
            if self._state is not LifecycleState.OPEN:
                return self._sealed_tombstone()
            # Legacy terminate/horizon tombstone. Only reachable for a non-seal env (a seal env
            # leaves OPEN via a seal above).
            if self._terminated:
                return CallResult(
                    content="<episode already terminated>",
                    meta=build_meta(terminate=True),
                    terminated=True,
                    tombstoned=True,
                )

            args = dict(arguments or {})
            # `_session_id` is a reserved hidden field the transport injects with the real id.
            # Strip any caller-supplied value before *both* dispatch and Step construction, so
            # a forged id can't run against the real session nor land in the trajectory.
            args.pop("_session_id", None)

            if not self._seal_enabled:
                # Non-seal env: the single-step path, entirely under the lock.
                return await self._legacy_step(tool_name, args)

            # ----- seal-enabled env -----
            if tool_name in self._score_schemas:
                # validate -> seal (evaluate happens after the lock is released)
                invalid = self._validate_terminal_args(tool_name, args)
                if invalid is not None:
                    # A malformed terminal request is a NORMAL validation error while the
                    # episode is still OPEN — no seal, no verdict, no evidence. The harness
                    # may correct and re-submit.
                    return invalid
                finalization = self._begin_finalization("explicit_tool", tool_name, args)
            elif tool_name == TERMINATE_TOOL_NAME:
                # An explicit abort on a scoring env: seal with a no-score abort evidence.
                finalization = self._begin_finalization("abort", None, None)
            else:
                # An ordinary tool in a scoring env. If this call reaches the horizon it IS the
                # terminal step: dispatch it, defer its trace row, and let the horizon
                # finalization write that same step as the single terminal row (no phantom
                # ``<horizon>`` step). Otherwise it's a normal mid-episode step.
                horizon = self._budget()
                is_horizon = horizon is not None and (self._step + 1) >= horizon
                result, _ = await self._dispatch_step(
                    tool_name, args, terminated=False, reaches_budget=is_horizon
                )
                if not is_horizon:
                    return result
                # Horizon has no submission: finalize with source=horizon and no args, but keep
                # the real tool name so the terminal trace row is labelled with the call that hit
                # the budget.
                finalization = self._begin_finalization("horizon", tool_name, None)

        # Phase 2 (lock released): await the single in-flight finalization, *shielded* so a
        # cancellation/disconnect of THIS request never cancels the evaluator or re-dispatches
        # it. If we are cancelled the finalization keeps running to completion in the
        # background; a later close() awaits the same future. Exactly one evaluation.
        await asyncio.shield(finalization)
        return finalization.result()

    def _budget(self) -> Optional[int]:
        """The step budget this episode enforces — the **published** one, and only ever that one.

        The snapshot is the contract: what the agent was framed with, what a stream compared
        before dispensing the task, and what a terminal call's arguments are validated against.
        The budget is part of it, so reading the env's live ``horizon`` to decide whether *this*
        call is the one that ends the task made the enforced budget a second, unpublished
        contract. An env whose live answer is lower than the one it published has the agent's
        first ordinary call made terminal: the episode seals, the finalizer scores a submission
        that never came, and the row lands ``sealed`` with ``correct=False`` and a clean close —
        indistinguishable from an agent that spent its turns and got the answer wrong. That is a
        wrong result wearing the shape of a right one, which is the outcome this layer exists to
        make impossible.

        So the published value is returned, and the live one is still read — to *check* it, not
        to obey it. A silent correction would leave an env free to publish one budget and run on
        another for as long as nobody compared the two, and this is the one place with both
        values in hand. A disagreement is an env fault and raises: the call is refused before it
        is dispatched, the episode stays OPEN and unsealed, and the layer above files an unscored
        row naming the failure (``score=None``, which cannot be averaged into anything) rather
        than a verdict nobody earned. That is also what an env whose ``horizon`` *raises* has
        always done here, so the two failures of the same read are answered the same way.

        The comparison itself runs the env's code — the live value is the env's object and
        ``__eq__`` is its method — and a raise there is this same fault reaching the same place,
        so it is left to propagate rather than caught into a verdict."""
        published = self._spec.horizon
        live = self._env.horizon
        if live != published:
            raise RuntimeError(
                "this env does not enforce the budget it published: the task contract advertises "
                f"horizon={_described(lambda: repr(published))} and the env now answers "
                f"{_described(lambda: repr(live))}"
            )
        return published

    # ----- legacy (non-seal) step: dispatch, record, verify -----

    async def _legacy_step(self, tool_name: str, args: Dict[str, Any]) -> CallResult:
        # The budget is read (and checked against the published one) BEFORE the call is
        # dispatched, as it is on the seal path: a call refused because the env does not enforce
        # the contract it published must not first be run against the env's task state and
        # committed to the trajectory. See `_budget`.
        horizon = self._budget()
        # Prospective step: don't advance `self._step` until the call actually completes. If
        # `call_tool` is cancelled (harness timeout) or raises, the counter stays put so the
        # next call reuses this number — the trajectory stays contiguous, one Step per
        # completed call.
        step = self._step + 1
        session = self._sessions.get(tool_name)
        if session is None:
            content = f"<unknown tool {tool_name!r}>"
        else:
            result = await session.call_tool(tool_name, args, tool_call_id=f"call-{step}")
            content = result.result
        # The await completed; commit the step atomically with its Step. Everything from here
        # on is synchronous, so no cancellation point can split them.
        self._step = step
        self._trajectory.append(
            Step(index=step, tool=tool_name, arguments=args, result=content)
        )

        terminated = tool_name == TERMINATE_TOOL_NAME or (
            horizon is not None and step >= horizon
        )
        self._terminated = terminated

        feedback = self._env.verify(self._trajectory, self._task, terminated=terminated)
        # Rendered once, here, and every sink below reads the rebuild rather than the env's own
        # objects: the retained terminal feedback, the trace row, and the in-band sidecar are
        # three renderings of one value instead of three questions to one object. See
        # `_core_feedback`.
        wire, items = _core_feedback([*feedback.inference, *feedback.episode])

        if terminated:
            # Retain the terminal feedback so the no-trace `evaluate()` path can report the
            # score directly off the episode (not only via the trace).
            self._terminal_feedback = wire

        if self._trace_path is not None:
            append_trace(
                self._trace_path,
                step_record(
                    session_id=self._session_id,
                    env_name=self._env_name,
                    task_id=self._task_id,
                    step=step,
                    tool=tool_name,
                    feedback=items,  # trace records everything, in or out of band
                    terminated=terminated,
                ),
            )

        # Eval-safe default: dense inference feedback is recorded (above) but not surfaced
        # in-band. v0 exposes no per-tool opt-in, so surface_inference stays False; episode
        # feedback rides out only on the terminal result.
        inband = select_inband(items, terminal=terminated, surface_inference=False)
        return CallResult(
            content=content,
            meta=build_meta(inband, terminate=terminated),
            terminated=terminated,
        )

    # ----- seal-enabled ordinary step (mid-episode, non-terminal) -----

    async def _dispatch_step(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        terminated: bool,
        reaches_budget: bool = False,
    ) -> "tuple[CallResult, int]":
        """Dispatch one ordinary tool call in a seal-enabled env and record it as a normal
        (non-terminal) step. Termination is owned by the finalizer, so ``terminated`` is False
        here; the caller decides whether the horizon was reached. Returns the result and the
        committed step index.

        ``reaches_budget=True`` says this call is the one that spends the published budget, and it
        makes this method stop at the commit. Two things follow from it, and the second is a
        lifecycle rule rather than a tidiness one:

        - the **trace row is deferred**, so the horizon finalization can write that same step as
          the single terminal (``terminated=True``) row: the ordinary call that hits the budget IS
          the terminal step, never a fabricated extra one;
        - and **no fallible env work runs between committing that result and the caller claiming
          the horizon finalization**. Everything below the commit is the env's code (``verify``,
          then rendering what it returns), and the caller claims the seal only after this returns.
          A raise from there therefore left the budget spent, the trajectory extended, the episode
          still ``OPEN`` at ``step == horizon`` and no finalization in existence, and the *next*
          call sealed a task that had already run out of turns: measured on the public path, an
          env whose non-terminal feedback could not be rendered at the horizon bought its agent a
          fourth call and a clean ``sealed`` row with ``success=true`` and no diagnostic. An env
          fault at the exact boundary, wearing the shape of an earned result.

        The preliminary pass is not merely dangerous there, it is dead weight: its trace row is
        deferred by design, the ``CallResult`` it builds is discarded by the caller, and it
        touches no retained state. The terminal ``verify`` sees the same trajectory including this
        step and is the authoritative sink for it. So at the budget the work is skipped rather
        than reordered, and the window it opened does not exist.

        Away from the budget this is unchanged: a failure in the same window is a lost call the
        layer above records as one, on an episode that is still ``OPEN`` and has spent a turn it
        cannot get back, which is the ordinary mid-episode contract."""
        step = self._step + 1
        session = self._sessions.get(tool_name)
        if session is None:
            content = f"<unknown tool {tool_name!r}>"
        else:
            result = await session.call_tool(tool_name, args, tool_call_id=f"call-{step}")
            content = result.result
        self._step = step
        self._trajectory.append(
            Step(index=step, tool=tool_name, arguments=args, result=content)
        )
        if reaches_budget:
            # Nothing between the commit above and the caller's claim, by the rule above. The
            # result is the caller's to discard: it returns the finalization's payload.
            return (
                CallResult(content=content, meta=build_meta(), terminated=terminated),
                step,
            )
        feedback = self._env.verify(self._trajectory, self._task, terminated=terminated)
        # Rendered once and rebuilt, for the reason `_core_feedback` gives: the trace row and the
        # sidecar below must be two renderings of one value, not two questions to one object.
        _, items = _core_feedback([*feedback.inference, *feedback.episode])
        if self._trace_path is not None:
            append_trace(
                self._trace_path,
                step_record(
                    session_id=self._session_id,
                    env_name=self._env_name,
                    task_id=self._task_id,
                    step=step,
                    tool=tool_name,
                    feedback=items,
                    terminated=terminated,
                ),
            )
        inband = select_inband(items, terminal=terminated, surface_inference=False)
        return (
            CallResult(
                content=content,
                meta=build_meta(inband, terminate=terminated),
                terminated=terminated,
            ),
            step,
        )

    # ----- terminal transaction -----

    def _validate_terminal_args(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[CallResult]:
        """Validate a ``score``-terminal call's args against its advertised outer schema.

        Returns a normal validation-error :class:`CallResult` (episode stays OPEN, not sealed,
        no verdict) when the request is malformed, else ``None``. Validates the **complete**
        advertised JSON Schema BEFORE any lifecycle mutation, so a request FastMCP would reject
        downstream (a wrong type, an unknown extra field, a missing required key) is a normal
        error the harness can correct and re-submit, never a sealed finalizer that irrevocably
        scores 0. On top of the schema, a required string must be non-blank (schema
        ``type: string`` accepts ``""``).

        **Only the caller's request is answered here.** The schema is the env's own object and
        validating against it runs the env's code — a key checked against the instance, a key
        formatted into a message — so this can fail for reasons the caller could never fix. Those
        are deliberately *not* turned into a validation error the caller is invited to retry:
        nothing it sends will satisfy a contract that cannot be read, and answering as if it
        might leaves the harness above composing an outcome for a task nobody could finish. They
        propagate, and the layer that owns the task's record classifies it (see
        :meth:`shogym.serve.stream.TaskStream.dispatch`). What does not propagate is the *name* of
        an argument this refusal is about — see :func:`_named`.

        **A ``SchemaError`` is one of those and used to be answered as a validation error.** It
        says the advertised schema is not a schema, which is a fact about the env and about every
        call any caller could make: the agent submitted the exactly correct answer, was told its
        request was invalid, and an orderly shutdown filed the task as a clean scored loss with no
        diagnostic anywhere. Nothing the caller sends can fix that, so it is not offered to the
        caller to fix. The schema is now proved a schema at construction
        (:func:`_core_schema`), which is where a contract this layer cannot enforce should be
        refused, so nothing left here can raise it from an env-shaped input; it is left uncaught
        rather than caught and misfiled, because the classification is the point."""
        schema = self._score_schemas.get(tool_name, {})
        try:
            jsonschema.validate(instance=args, schema=schema)
        except jsonschema.ValidationError as exc:
            # `exc.message` describes only the caller's own (already public) input — no gold
            # answer, no verdict, no oracle.
            return self._validation_error(f"invalid arguments: {exc.message}")

        required = schema.get("required", []) if isinstance(schema, dict) else []
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        for key in required:
            # Matched, not compared. A schema that would not render is kept as the env published
            # it, so a key in it is the env's object with the env's `__eq__` on the dispatching
            # side of every comparison here. One that answers True to the reserved name excuses
            # itself from the check below, which is the only thing standing between a blank
            # submission and a seal; one that answers True to `"string"` applies the check to an
            # argument the schema never said was text. See `_declared`.
            if _declared(key, _RESERVED_ARGS) is not None:  # transport-injected, never a client's
                continue
            if _declared(props.get(key, {}).get("type"), _TEXT_TYPE) is not None:
                value = args.get(key)
                if not isinstance(value, str) or not value.strip():
                    return self._validation_error(
                        f"argument {_named(key)} must be a non-blank string"
                    )
        return None

    def _validation_error(self, message: str) -> CallResult:
        """A terminal-arg validation failure: a plain error result while still OPEN. Carries no
        seal, no terminate flag, and does not advance the step — rejected at ingress before any
        dispatch."""
        return CallResult(
            content=json.dumps({"error": message, "validation_error": True}),
            meta=build_meta(),
            terminated=False,
        )

    def _begin_finalization(
        self, source: str, tool_name: Optional[str], args: Optional[Dict[str, Any]]
    ) -> "asyncio.Future[CallResult]":
        """Atomically seal the episode (OPEN -> SEALED -> FINALIZING), persist the durable
        ``SEALED`` record, and spawn the single finalization task. Called under the lock."""
        finalization_id = str(uuid7())
        # Rendered once, here, and every use after it reads the rendering. The render walks the
        # caller's arguments, and walking a container is asking it a question: four walks of one
        # submission were four chances for a value that answers the first three to refuse the
        # fourth, from inside the commit, with the record already written. One walk, one answer,
        # and the answer is what the transaction carries.
        #
        # The rendering rather than the digest alone, because the digest was never the only
        # consumer. The same dictionary went on to the evaluator as `FinalizeRequest.args`, and
        # came back out of it into the terminal `Step` the verifier scores and into
        # `evidence.args`, so a finalizer that rewrote `req.args` changed what the trajectory
        # said the agent had submitted while the durable digest still witnessed the call that
        # arrived. Measured: the digest matched `{"answer": "4"}`, `verify` scored
        # `{"answer": "mutated-by-finalizer"}`, and the public payload reported a clean success
        # over the two. The submission is core-owned data from here down, and what the evaluator
        # gets is a detached copy of it (see `_core_args` and `_detached_args`).
        #
        # Taken **before** the seal, which is the invariant the block below states. It renders
        # the caller's values through `json.dumps(default=str)`, so it runs their `__str__`, and
        # done after the transition it raised out of a sealed episode that had no finalization to
        # join: `sealed` true, `terminated` false, no future for `wait_finalized` or `close` to
        # wait on, and not even a `SEALED` record for recovery to resolve. Ordered here, that
        # same failure is an ordinary lost call on an episode still OPEN, which the layer above
        # already records as one.
        #
        # ----- the one value that is rendered before it, and why -----
        #
        # The caller's own value, echoed into the canonical fail-closed verdict for calibration,
        # which makes it the one piece of somebody else's data the core puts in a verdict it
        # builds itself. A schema that does not constrain the argument lets a non-JSON value
        # through validation, and then the replacement is unserializable for exactly the reason
        # the thing it replaced was, so the commit that has to write it raises with no verdict
        # left to fall back to.
        #
        # What may be echoed into a verdict this core builds is narrower than what the digest
        # will witness: a value this core can put on the wire *as itself*, so the render here is
        # the strict one and a value that will not take it is not echoed at all.
        #
        # **Taken first, and substituted into the submission.** Rendering the submission and then
        # going back to the caller's object for this one key was two walks of one value, and a
        # walk is a question: a container that answers differently the second time was digested
        # as one submission and echoed as another. Measured, with a list subclass that yields a
        # different element per walk: the digest, the evaluator's request and the terminal step
        # said `[1]` while the public fail-closed payload and the durable FAILED verdict said
        # `[2]`, which is the durable-versus-public split this transaction exists to make
        # impossible. Rendered once and put back into the arguments, the submission's own walk
        # below reads plain data for this key and no caller value is traversed twice.
        raw = args.get("confidence") if isinstance(args, dict) else None
        rendered, confidence = _wire(raw) if raw is not None else (False, None)
        if not rendered:
            confidence = None
        elif args is not None:
            args = {**args, "confidence": confidence}
        submission = _core_args(args)
        digest = args_digest(submission)
        self._finalization_id = finalization_id
        self._finalization_source = source
        self._finalization_tool = tool_name
        self._args_digest = digest

        # ----- the seal, and the one invariant that makes it a transaction -----
        #
        # **Between the transition below and the finalization claim two lines later, nothing may
        # raise.** A sealed episode owes exactly one verdict and the finalization future is the
        # only thing that can pay it, so an exception in that window leaves an episode that has
        # left OPEN, will never terminate, and offers nothing for `wait_finalized()` or `close()`
        # to join. Every ingress after it is tombstoned and the run reports a task nobody scored.
        #
        # So the window holds only core-owned operations: two attribute writes, one persistence
        # call that is best-effort against every failure by contract (`_write_record`), and the
        # task creation itself. Everything that runs code this module did not write is either
        # above the transition (the render, the digest and the confidence, and in `call()` the
        # argument validation, the budget read and any dispatch) or inside `_run_finalize`, where
        # a failure becomes the canonical fail-closed verdict instead of an escape. Anything added
        # here has to satisfy one of those two, and the tests that pin it are the parametrized
        # lost-call cases in tests/test_serve_episode_fail_closed.py.
        self._state = LifecycleState.SEALED
        self._write_record("SEALED", source, digest)
        finalization: "asyncio.Future[CallResult]" = asyncio.ensure_future(
            self._run_finalize(source, tool_name, submission, finalization_id, confidence)
        )
        self._finalization = finalization
        self._state = LifecycleState.FINALIZING
        return finalization

    async def _run_finalize(
        self,
        source: str,
        tool_name: Optional[str],
        submission: Optional[Dict[str, Any]],
        finalization_id: str,
        confidence: Optional[Any],
    ) -> CallResult:
        """The single finalization: run the evaluator (``finalize``) on the *already-sealed*
        episode, stamp non-forgeable provenance onto the evidence, persist it durably, append
        the versioned ``terminal`` trace event, score via ``verify(evidence)``, tear down, and
        return the sanitized public payload.

        Runs as its own task (created at seal), shielded by every awaiter, so a cancellation of
        the awaiting request or a racing close() cannot abandon it or spawn a second evaluation.
        Fail-closed: an evaluator timeout/crash yields a ``finalize_error`` verdict
        (``correct=False``) rather than propagating, with only a private diagnostic (never
        exception text to the agent).

        ``submission`` and ``confidence`` are both renderings taken at the seal, so nothing the
        caller supplied is read again from here down: the submission is the plain data the durable
        digest witnesses (:func:`_core_args`) and the confidence is the value this core may echo
        into a verdict it builds itself."""
        self._write_record("PENDING", source, self._args_digest)
        try:
            if source == "abort":
                # An abort is a no-score path: core synthesizes the evidence directly (no
                # evaluator is invoked at teardown).
                evidence = TerminalEvidence(
                    source="abort",
                    status="ok",
                    verdict={"correct": False, "aborted": True},
                )
            else:
                req = FinalizeRequest(
                    source=source,  # type: ignore[arg-type]
                    finalization_id=finalization_id,
                    session_id=self._session_id,
                    # A copy the evaluator may keep and scribble on: the sealed submission is
                    # what this transaction commits, and an evaluator holding a reference to it
                    # rewrote the trajectory the verifier scores while the digest went on
                    # witnessing the call that arrived (see `_detached_args`).
                    args=_detached_args(submission),
                    deadline=self._finalize_deadline,
                    tool_name=tool_name,
                )
                # Run the evaluator as its own retained task so a deadline timeout can fail the
                # verdict closed WITHOUT abandoning the evaluator: `_teardown` drains this task
                # before releasing env/session resources, so a late evaluator can never touch
                # torn-down state or run a second time.
                eval_task: "asyncio.Future[Any]" = asyncio.ensure_future(
                    self._finalize(req)  # type: ignore[misc]
                )
                self._eval_task = eval_task
                if self._finalize_deadline is not None:
                    # shield inside wait_for: the deadline must not cancel the evaluator
                    # mid-flight (that could wedge the in-process client); it only bounds how
                    # long we *await* the verdict before failing closed. The evaluator keeps
                    # running and is drained by `_teardown`.
                    evidence = await asyncio.wait_for(
                        asyncio.shield(eval_task), timeout=self._finalize_deadline
                    )
                else:
                    evidence = await eval_task
                if not isinstance(evidence, TerminalEvidence):
                    raise TypeError(
                        f"finalize must return TerminalEvidence, got {type(evidence)!r}"
                    )
                # Read here and nowhere else. `isinstance` admits a subclass, so every field
                # below is still the env's code until this line rebuilds them as the core's own
                # data — inside this guard, so a field that raises while being read is an
                # evaluator failure like any other. See `_core_owned`.
                evidence = _core_owned(evidence)
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as exc:
            # Fail-closed on ANY evaluator failure — including cancellation. A cancellation here
            # can only come from the *evaluator's own* awaited work being cancelled: the
            # finalization task is shielded from every caller (`call`/`close`), so an awaiter's
            # cancellation never propagates into `_run_finalize`. Swallowing it and committing a
            # fail-closed verdict is therefore correct (the episode never strands FINALIZING with
            # a PENDING record) and preserves the caller-cancellation rule, which lives at the
            # shielded await in `call()`/`close()`, not here.
            evidence = TerminalEvidence(
                source=source,  # type: ignore[arg-type]
                status="finalize_error",
                verdict=fail_closed_verdict(confidence),
                # Through the guarded renderer: naming the failure is the decoration and the
                # fail-closed verdict is the point, so an evaluator failure that cannot be
                # written down still gets written down — see `_rendered_failure`.
                diagnostic=f"finalize failed: {_rendered_failure(exc)}",
            )

        # A verdict that is not a JSON *object* — a list, a string, a number, a null — would
        # raise mid-commit: `_sanitize_terminal`'s `dict(evidence.verdict)` at the terminal-step
        # append runs AFTER the episode reached FINALIZING with a PENDING record, so the raise
        # would strand that record and surface an exception to the client instead of the
        # documented fail-closed result. Refused here, before any commit, so the terminal
        # transaction always completes FINALIZED (fail-closed) and returns the safe result.
        #
        # Whether it *serializes* is not asked again here. It was asked once, at the boundary,
        # where the answer was taken from the value the env supplied and a failure became this
        # same fail-closed verdict (see `_wire_verdict`); what reaches this line is either the
        # rendering of that value or a verdict this module built. Asking again would be a second
        # walk of a value already settled, which is the shape that let a stateful container be
        # checked as one thing and committed as another.
        if not isinstance(evidence.verdict, dict):
            evidence = TerminalEvidence(
                source=source,  # type: ignore[arg-type]
                status="finalize_error",
                verdict=fail_closed_verdict(confidence),
                diagnostic="finalize returned a verdict that is not a JSON object",
            )

        # Core stamps the non-forgeable fields — source, finalization_id, provenance — so a
        # verdict is only ever trusted with core-owned provenance the harness cannot supply.
        evidence.source = source  # type: ignore[assignment]
        evidence.finalization_id = finalization_id
        evidence.provenance = {
            "core": "shogym-serve",
            "session_id": self._session_id,
            "finalization_id": finalization_id,
            "sealed_source": source,
        }
        if evidence.args is None and source == "explicit_tool":
            # The sealed submission, and its own copy of it: an evaluator that returned no args
            # of its own is answered with the ones the record witnesses, and this field then
            # travels to `verify` through `_detached_evidence`.
            evidence.args = _detached_args(submission) or {}

        # Commit evidence atomically under the lifecycle lock (FINALIZING -> FINALIZED). The
        # commit is wrapped so an unexpected failure in verify/trace still fails closed and,
        # via the `finally`, ALWAYS reaches teardown — a committed episode can never leak its
        # env/session state or return an exception instead of a terminal result.
        items: List[Any] = []
        try:
            async with self._lock:
                self._evidence = evidence
                self._terminated = True

                # Append the terminal step to the trajectory BEFORE calling verify, so a
                # migrated verifier sees the COMPLETE call history — including the terminal
                # action it is scoring (turn counts, a terminal inference signal, ...). The
                # horizon path already recorded its budget-reaching call as this step;
                # explicit_tool/abort append it now.
                if source == "horizon":
                    # The ordinary call that hit the budget was already appended to the
                    # trajectory (with its trace row deferred). Reuse it as the terminal step —
                    # never fabricate a phantom step or a tool invocation that didn't happen.
                    step = self._step
                    terminal_tool = tool_name or "<horizon>"
                else:
                    # explicit_tool / abort: the terminal action (submit / terminate) is a real
                    # call with no prior step — append it as the terminal step now.
                    step = self._step + 1
                    self._step = step
                    terminal_tool = tool_name or "terminate"
                    self._trajectory.append(
                        Step(
                            index=step,
                            tool=terminal_tool,
                            # The submission the record witnesses, and this step's own copy of
                            # it. Built from the dictionary the evaluator had just been handed,
                            # this said whatever that evaluator had left in it, so the verdict
                            # scored below and the digest already durable described two different
                            # calls (see `_core_args`).
                            arguments=_detached_args(submission) or {},
                            result=json.dumps(self._sanitize_terminal(evidence)),
                        )
                    )

                # A verifier bug must not strand the episode: fail closed on a verify() raise.
                #
                # Every raise, `CancelledError` included. This is a containment boundary around
                # the env's own synchronous code, and an `except Exception` here let through the
                # one exception that walks out of a finalization with everything it still owed
                # undone: no durable record (so the seal stays PENDING and recovery has to
                # resolve it), no terminal trace event, the lifecycle stuck at FINALIZING, and a
                # cancellation raised at `call()`, at `wait_finalized()` and at every `close()` —
                # an episode that cannot be shut down and a submission the layer above can only
                # file as a broker abort. An env raising cancellation is that env failing:
                # nothing here asked for it, since the finalization task is shielded from every
                # caller, so an awaiter's cancellation never reaches this code. Nothing between
                # the baseline and the call suspends either, which is why no baseline is taken —
                # see `_must_propagate`.
                #
                # The verifier is handed a **detached view** and never the object this commit
                # reads from. `verify` takes the evidence so a migrated env can score from it,
                # which means the env holds a reference to it for the length of that call, and
                # the commit below then consumes `status`, `verdict`, `provenance` and
                # `diagnostic` again without asking any of the questions `_core_owned` asked. A
                # verifier that returned ordinary feedback and rewrote `status` on its way out
                # therefore reopened the undeclared-status path this transaction closes, and the
                # same handle rewrote the provenance the core stamps precisely so a harness
                # cannot supply it: the durable record was written FINALIZED, with no diagnostic
                # and with `{"core": "not-shogym"}` in the field that exists to say otherwise.
                #
                # "The value checked is the value committed" needs its corollary, and this is it:
                # nothing foreign holds a reference to that value between the check and the
                # commit. A copy costs one walk of data this module already owns.
                try:
                    feedback = self._env.verify(
                        self._trajectory,
                        self._task,
                        terminated=True,
                        evidence=_detached_evidence(evidence),
                    )
                    items = [*feedback.inference, *feedback.episode]
                except BaseException as exc:  # noqa: BLE001 — verifier failure => fail closed
                    if _must_propagate(exc, None):
                        raise
                    evidence = TerminalEvidence(
                        source=source,  # type: ignore[arg-type]
                        status="finalize_error",
                        verdict=fail_closed_verdict(confidence),
                        provenance=evidence.provenance,
                        finalization_id=finalization_id,
                        diagnostic="verify() raised while scoring the terminal evidence",
                    )
                    self._evidence = evidence
                    items = []

                # The feedback that verify returned is serialized here, and a value the wire
                # refuses raises. Deliberately NOT contained: unlike the failures above, this one
                # has an owner. A stream catches it out of the terminating call, redacts the
                # answer to the agent, files the row `finalize_error` with `score=None` and the
                # failure named on it, and stops the run — because an env whose feedback cannot
                # be serialized will fail the same way on every task behind this one. Failing the
                # verdict closed here instead would answer with a *scored* row and let the queue
                # carry on against the same env (see
                # `test_a_forced_abort_the_env_fails_is_not_an_earned_give_up`).
                #
                # An ordinary `Exception` therefore leaves here verbatim, and that is the whole
                # of the deliberate part. `CancelledError` is not an ordinary exception and not
                # an outcome anyone can own: letting it out marks the finalization *task*
                # cancelled, and a cancelled task is control flow rather than a failure, so the
                # joins that were going to answer for it step aside instead. Measured: `call()`,
                # `wait_finalized()` and every `close()` raised cancellation, the stream saw the
                # env fail and then lost the row to the same cancelled future, `env.close()` was
                # never reached, and the record stayed PENDING.
                #
                # Nothing is awaited in this comprehension, so a cancellation observed here was
                # raised by the env's own value and is not one requested against this task, which
                # is the distinction `_must_propagate` already draws. It is translated into an
                # ordinary failure carrying the same information, so the stream owns it exactly
                # as it owns the serializer's own `ValueError`. Translating here rather than
                # teaching each join to tell a cancelled child from its own cancellation is the
                # same choice this module makes everywhere else: contain at the boundary where
                # the foreign code runs, once, instead of at every awaiter forever.
                #
                # **What the failure may not do is leave this transaction half-made.** The owner
                # above composes a *row*; the record underneath it is this module's, and on the
                # transport-independent path (`evaluate()`, `run_stdio()`) there is no owner at
                # all. Raising from here before the commit left exactly the shape this PR exists
                # to remove, one boundary later: the episode CLOSED and terminated, the evaluator's
                # `ok` verdict held in memory, and the durable record still `PENDING` with no
                # verdict, so what the run answered and what the store remembered disagreed and
                # recovery would resolve the disagreement the other way. So the commit happens
                # first, fail-closed and attributed, and the failure is handed on afterwards. The
                # stream's answer is unchanged: it still catches this out of the terminating call,
                # still files `finalize_error` with `score=None`, and now files it over a record
                # that says the same thing.
                items_failure: Optional[BaseException] = None
                try:
                    self._terminal_feedback, items = _core_feedback(items)
                except BaseException as exc:
                    if _must_propagate(exc, None):
                        raise
                    if isinstance(exc, Exception):
                        items_failure = exc
                    else:
                        items_failure = RuntimeError(
                            "verify() returned terminal feedback this episode cannot record: "
                            f"{_rendered_failure(exc)}"
                        )
                        items_failure.__cause__ = exc
                        items_failure.__suppress_context__ = True
                    self._terminal_feedback = []
                    items = []
                    evidence = TerminalEvidence(
                        source=source,  # type: ignore[arg-type]
                        status="finalize_error",
                        verdict=fail_closed_verdict(confidence),
                        provenance=evidence.provenance,
                        finalization_id=finalization_id,
                        diagnostic=(
                            "verify() returned terminal feedback this episode cannot record: "
                            f"{_rendered_failure(items_failure)}"
                        ),
                    )
                    self._evidence = evidence

                # Durable state leads the public trace. Persist the terminal record FIRST —
                # FINALIZED (ok) or FAILED (fail-closed) — so a crash can never leave a public
                # trace that says `ok` while the recoverable record still says PENDING. If we
                # crash after this fsync but before the trace event, recovery replays this
                # FINALIZED/FAILED evidence and the trace merely lacks its (derivable) terminal
                # event — never a contradiction. Confidential diagnostic + provenance live only
                # here, never in the user-readable trace.
                self._write_record(
                    "FAILED" if evidence.finalize_error else "FINALIZED",
                    source,
                    self._args_digest,
                    verdict=evidence.verdict,
                    provenance=evidence.provenance,
                    diagnostic=evidence.diagnostic,
                )
                self._state = LifecycleState.FINALIZED
                # Write the terminal step row (keeps evaluate()/result_from_trace working) then
                # the versioned terminal event after it (the terminal event is the last row).
                # Best-effort, like the durable record: a trace-store I/O failure is flagged
                # degraded, never allowed to strand a FINALIZED episode.
                if self._trace_path is not None:
                    try:
                        append_trace(
                            self._trace_path,
                            step_record(
                                session_id=self._session_id,
                                env_name=self._env_name,
                                task_id=self._task_id,
                                step=step,
                                tool=terminal_tool,
                                feedback=items,
                                terminated=True,
                            ),
                        )
                        append_terminal_event(
                            self._trace_path,
                            terminal_event_record(
                                session_id=self._session_id,
                                env_name=self._env_name,
                                task_id=self._task_id,
                                step=step,
                                source=source,
                                status=evidence.status,
                                verdict=evidence.verdict,
                                finalization_id=finalization_id,
                                args_digest=self._args_digest,
                            ),
                        )
                    except Exception:  # noqa: BLE001 — trace is best-effort; never strand
                        self._persist_degraded = True

                # Committed, so the failure can be somebody else's now. Raised inside the lock
                # and inside the `try`, so the teardown below still runs and the finalization
                # future carries the same exception the terminating call raises.
                if items_failure is not None:
                    raise items_failure
        finally:
            # Teardown ALWAYS runs after the commit — even if the commit path failed unexpectedly.
            # If the evaluator is STILL running (the deadline fired and we failed closed while it
            # keeps going, shielded), defer the drain+teardown to the background so the caller
            # receives the fail-closed result AT the deadline — resources are released only once
            # the evaluator actually finishes (`_teardown` drains it first, no use-after-free). On
            # the normal path the evaluator is already done, so teardown runs inline.
            if self._eval_task is not None and not self._eval_task.done():
                self._drain_task = asyncio.ensure_future(self._teardown())
            else:
                await self._teardown()

        public = self._sanitize_terminal(evidence)
        inband = select_inband(items, terminal=True, surface_inference=False)
        return CallResult(
            content=json.dumps(public),
            meta=build_meta(inband, terminate=True),
            terminated=True,
        )

    def _write_record(
        self,
        status: str,
        source: str,
        digest: Optional[str],
        *,
        verdict: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        diagnostic: Optional[str] = None,
    ) -> None:
        """Persist one durable record — **best-effort**. A local-file I/O failure (ENOSPC,
        permissions) must never strand the lifecycle: the seal, the finalization, and the
        in-memory verdict proceed regardless, so a sealed episode always yields an outcome. A
        persistence failure only degrades crash-recovery for *this* record (a normal crash still
        works, since the store is normally writable); it is flagged for audit, never raised.

        Never raised for **any** failure, and the ones that are not I/O are the reason. Writing a
        record copies the verdict on the way in (``asdict`` deep-copies whatever is not itself a
        dataclass), and a copy is the value's own code — an env's ``str`` subclass sitting in the
        verdict it returned is JSON-clean, passes every check above, and still gets to decide
        whether its own persistence succeeds. Caught as ``Exception`` this was best-effort
        against every failure but one: a value whose copy raised cancellation left the record at
        ``PENDING`` and took the *verdict* with it, out through the commit to ``call()`` and
        ``close()``, so a graded answer the env had already returned reached nobody. A record
        this store cannot write is a degraded record; it is never also a lost outcome."""
        if self._store is None or self._finalization_id is None:
            return
        try:
            self._store.write(
                FinalizationRecord(
                    session_id=self._session_id,
                    finalization_id=self._finalization_id,
                    status=status,  # type: ignore[arg-type]
                    source=source,  # type: ignore[arg-type]
                    args_digest=digest,
                    verdict=verdict,
                    provenance=provenance,
                    diagnostic=diagnostic,
                    owner_pid=os.getpid(),
                )
            )
        except BaseException as exc:  # noqa: BLE001 — best-effort; never strand the seal
            if _must_propagate(exc, None):  # nothing here awaits — see `_must_propagate`
                raise
            self._persist_degraded = True

    def _sanitize_terminal(self, evidence: TerminalEvidence) -> Dict[str, Any]:
        """The public-safe terminal payload: the core-stamped verdict + a ``finalize_error``
        flag. The verdict an env returns is already public-safe (the RFC forbids returning
        judge reasoning / extracted answers / exception text); the private diagnostic and
        provenance are dropped here and live only in the durable store."""
        public = dict(evidence.verdict)
        public["finalize_error"] = evidence.finalize_error
        return public

    def _sealed_tombstone(self) -> CallResult:
        """The generic post-seal tombstone: no inward dispatch, no verdict re-exposed. Used for
        every `tools/call` (repeat terminal, unknown tool) after the seal."""
        return CallResult(
            content="<episode sealed; no further tool calls are dispatched>",
            meta=build_meta(terminate=True),
            terminated=True,
            tombstoned=True,
        )

    async def _teardown(self) -> None:
        """Drop the env's per-session state. Idempotent, runs exactly once. Owned by the
        finalizer (after evidence); close() routes through it too. The MCP client sessions are
        disposed by :meth:`close` in its own task (anyio requires the client to be exited in
        the task that entered it), so this touches only pure-Python env state and is safe to
        run from the finalization background task."""
        # Drain any still-running evaluator BEFORE dropping env state. On the deadline path the
        # verdict was already committed fail-closed while the (shielded) evaluator kept running;
        # it must finish before `end_session` clears the per-session state it may still touch, so
        # a timed-out evaluator can never use-after-free or run a second time.
        eval_task = self._eval_task
        if eval_task is not None and not eval_task.done():
            try:
                await asyncio.shield(eval_task)
            except BaseException:  # incl. the evaluator's own CancelledError — never block teardown
                pass
        async with self._lock:
            if self._torn_down:
                return
            self._torn_down = True
            self._teardown_runs += 1
            self._state = LifecycleState.TEARING_DOWN
        # Best-effort against every raise, `CancelledError` included. This runs in the
        # finalization's `finally`, so a raise here does not merely fail the cleanup — it
        # replaces the finalization's return value. Caught as `Exception` it was not best-effort
        # against that one: an env whose `end_session` raised cancellation took an already
        # committed, durably `FINALIZED`, correct verdict and turned it into a cancellation at
        # `call()`, at `wait_finalized()` and at every `close()`, so the record on disk and the
        # result the run reported disagreed and the run filed an earned answer as a broker abort.
        # Nothing here awaits between the call and the catch, so a cancellation observed is one
        # this call raised — see `_must_propagate`.
        try:
            self._env.end_session(self._session_id)
        except BaseException as exc:  # noqa: BLE001 — cleanup may not discard a decided verdict
            if _must_propagate(exc, None):
                raise
        self._state = LifecycleState.CLOSED

    async def close(self) -> None:
        # close() participates in the lifecycle for a seal env. If a finalization is in flight,
        # WAIT for it to commit evidence + tear down before disposing anything (so the
        # evaluator's live session isn't reclaimed mid-finalize); otherwise atomically claim an
        # abort and own teardown. Then dispose the MCP client sessions and let the env drop
        # residual state. For a non-seal env none of this engages (state stays OPEN with no
        # finalization), so close() is just the plain teardown below.
        finalization: Optional["asyncio.Future[CallResult]"] = None
        if self._seal_enabled:
            async with self._lock:
                finalization = self._finalization
                if finalization is None and self._state is LifecycleState.OPEN:
                    # No seal happened (a score env closed before submitting): claim an abort
                    # finalization so this close owns teardown + records a no-score verdict.
                    finalization = self._begin_finalization("abort", None, None)
            if finalization is not None:
                # Shielded: close() must not cancel the single in-flight finalization.
                try:
                    await asyncio.shield(finalization)
                except Exception:
                    pass
            # A deadline-path finalization returns before its background drain+teardown finishes;
            # wait for it so the evaluator has drained and env state is released before we dispose
            # the MCP sessions below.
            if self._drain_task is not None:
                try:
                    await asyncio.shield(self._drain_task)
                except Exception:
                    pass
            await self._teardown()  # idempotent: a no-op if the finalizer already ran it

        # Close every MCP session opened for this episode, then let the env tear down its own
        # per-session state. `env.close()` drops in-process server state via `end_session`;
        # out-of-process sessions are reaped by their `close()` above (one subprocess/session).
        for session in self._opened:
            try:
                await session.close()
            except Exception:
                pass
        await self._env.close()
