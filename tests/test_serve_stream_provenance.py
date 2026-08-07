"""Provenance spans: extra, namespaced provenance per dispensed task.

The contract under test is containment. An extension may add to a row and may fail in any way
it likes — raise, hang, return junk — and in no case may it suppress a row, duplicate one,
change a score, or reach anything but the redacted outcome it is handed.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from shogym.serve import stream as stream_module
from shogym.serve.stream import (
    _TASK_OVER,
    Closure,
    CompletedTask,
    ProvenanceError,
    ProvenanceSpan,
    TaskRef,
    TaskStream,
    read_dispenses,
    read_results,
    reconcile,
)
from shogym.types import EpisodeFeedback, FeedbackCollection
from tests._fixtures.score_env import ENV_NAME, SUBMIT_TOOL, _FixtureScoreEnv

TASKS = [
    {"id": "q0", "question": "2+2?", "answer": "4"},
    {"id": "q1", "question": "3+3?", "answer": "6"},
]


def _stream(tmp_path: Path, indices: List[int], **kwargs: Any) -> TaskStream:
    return TaskStream(
        lambda _name: _FixtureScoreEnv(tasks=TASKS),
        [TaskRef(ENV_NAME, i) for i in indices],
        prov_dir=tmp_path / "prov",
        **kwargs,
    )


class _Snapshots:
    """A well-behaved extension: a hash before, a hash after, and the closure it saw."""

    namespace = "test.snapshots"

    def __init__(self) -> None:
        self.begins: List[TaskRef] = []
        self.completed: List[CompletedTask] = []

    async def begin(self, ref: TaskRef) -> ProvenanceSpan:
        self.begins.append(ref)
        return _SnapshotSpan(self, len(self.begins))


class _SnapshotSpan:
    def __init__(self, owner: _Snapshots, nth: int) -> None:
        self._owner = owner
        self._nth = nth

    @property
    def dispensed(self) -> Dict[str, Any]:
        return {"hash": f"before-{self._nth}"}

    async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
        self._owner.completed.append(completed)
        return {"hash": f"after-{self._nth}", "closure": completed.closure}


class _Misbehaving:
    """An extension that fails however the test asks it to."""

    namespace = "test.bad"

    def __init__(
        self,
        *,
        begin_error: Optional[BaseException] = None,
        begin_hangs: bool = False,
        begin_returns: Any = None,
        finalize_error: Optional[BaseException] = None,
        finalize_hangs: bool = False,
        finalize_returns: Any = None,
    ) -> None:
        self.__dict__.update(locals())
        del self.__dict__["self"]
        self.finalize_calls = 0

    async def begin(self, ref: TaskRef) -> ProvenanceSpan:
        if self.begin_hangs:
            await asyncio.sleep(60)
        if self.begin_error is not None:
            raise self.begin_error
        return self  # type: ignore[return-value]

    @property
    def dispensed(self) -> Dict[str, Any]:
        return self.begin_returns if self.begin_returns is not None else {"ok": True}

    async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
        self.finalize_calls += 1
        if self.finalize_hangs:
            await asyncio.sleep(60)
        if self.finalize_error is not None:
            raise self.finalize_error
        return self.finalize_returns if self.finalize_returns is not None else {"ok": True}


class _RepeatsAName(dict):
    """A mapping that answers for itself, and answers with one name twice.

    The encoder asks a ``dict`` subclass for its ``items()``, so this is the one way two of the
    same name reach a JSON object after every key has been checked to be exact text — and what
    it holds is not what it says it holds, so a record built from either half of that
    disagreement is a value no reader could ever have got from the object."""

    def items(self) -> Any:  # type: ignore[override]
        return [("shot", "first"), ("shot", "second")]


class _NotQuiteAName(str):
    """A key that is text on the wire and is not equal to the plain key of the same text."""

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# Payloads whose names a JSON encoder will happily coerce into text, each one carrying two keys
# that collapse onto a single name — so exactly one of the two values survives and nothing says
# which. Every value is the same marker, so a test can assert that no part of the payload reached
# the row under any spelling.
_COERCED = "coerced-onto-the-row"
_COERCIBLE_NAMES: List[Any] = [
    {"1": _COERCED, 1: _COERCED},
    {"true": _COERCED, True: _COERCED},
    {"1.0": _COERCED, 1.0: _COERCED},
    {"outer": {"1": _COERCED, 1: _COERCED}},
    {"outer": [{"1": _COERCED, 1: _COERCED}]},
    {_NotQuiteAName("a"): _COERCED, "a": _COERCED},
]
_COERCIBLE_IDS = [
    "int-name",
    "bool-name",
    "float-name",
    "nested-name",
    "name-inside-an-array",
    "str-subclass-name",
]

# A JSON object that contains itself. The encoder finds this too; the check in front of it has
# to reach the same answer rather than recursing until the interpreter stops it.
_SELF_REFERENTIAL: Dict[str, Any] = {}
_SELF_REFERENTIAL["self"] = _SELF_REFERENTIAL


# ----- the happy path -----


async def test_a_span_nests_its_output_under_its_namespace(tmp_path: Path) -> None:
    snapshots = _Snapshots()
    async with _stream(tmp_path, [0], provenance=[snapshots]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    (row,) = stream.results
    assert row.extensions == {
        "test.snapshots": {
            "dispensed": {"hash": "before-1"},
            "sealed": {"hash": "after-1", "closure": "sealed"},
        }
    }
    # Authoritative fields are untouched by the extension.
    assert row.score is not None and row.score.success is True
    assert read_results(tmp_path / "prov")[0].extensions == row.extensions


async def test_reading_an_extension_s_output_cannot_rewrite_it(tmp_path: Path) -> None:
    # `extensions` is a dict of dicts on a frozen row, so a reader handed the run's own row can
    # edit an extension's output — and every later read, and anything scoring the run in memory,
    # would show the edit while `results.jsonl` said something else. That is the same disagreement
    # between the public and the durable view that the row's own fields are detached against, and
    # provenance is where it is easiest to miss: nothing else in the run reads these values, so a
    # rewrite is invisible until someone compares the two records.
    #
    # So the run keeps the wire form the file holds — extensions included — and hands out copies.
    snapshots = _Snapshots()
    async with _stream(tmp_path, [0], provenance=[snapshots]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    durable = read_results(tmp_path / "prov")[0].extensions
    first = stream.results[0]
    assert first.extensions == durable
    assert first.extensions is not stream.results[0].extensions, "two reads share one mapping"
    assert (
        first.extensions["test.snapshots"]
        is not stream.results[0].extensions["test.snapshots"]
    ), "two reads share one namespace's output"

    first.extensions["test.snapshots"]["sealed"]["hash"] = "invented"
    first.extensions["test.snapshots"]["dispensed"] = {"invented": True}
    first.extensions["test.invented"] = {"whatever": True}

    assert stream.results[0].extensions == durable, "an extension's output was rewritten"
    assert read_results(tmp_path / "prov")[0].extensions == durable
    # And the row is still the one the file holds, whole.
    assert stream.results[0].to_wire() == read_results(tmp_path / "prov")[0].to_wire()


async def test_spans_pair_correctly_across_a_repeated_task_index(tmp_path: Path) -> None:
    # The whole reason a span exists: TaskRef is not a dispense identity, so a before/after pair
    # cannot be keyed on it when the same index is queued twice.
    snapshots = _Snapshots()
    async with _stream(tmp_path, [1, 1], provenance=[snapshots]) as stream:
        for _ in range(2):
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "6"})
    pairs = [
        (row.extensions["test.snapshots"]["dispensed"], row.extensions["test.snapshots"]["sealed"])
        for row in stream.results
    ]
    assert pairs == [
        ({"hash": "before-1"}, {"hash": "after-1", "closure": "sealed"}),
        ({"hash": "before-2"}, {"hash": "after-2", "closure": "sealed"}),
    ]
    assert snapshots.begins == [TaskRef(ENV_NAME, 1), TaskRef(ENV_NAME, 1)]


async def test_a_span_finalizes_over_a_row_a_lost_call_left_unscored(tmp_path: Path) -> None:
    # A call the harness lost leaves the row unscored (`finalize_error`) without stopping the
    # queue, and a span is open across the whole of that: `begin` ran before the call was made and
    # `finalize` runs after the stream ended the task itself. The extension is told what the row
    # says — an unearned closure and no score — rather than the `drained` seal the task would have
    # been filed as, so provenance and the row cannot disagree about how the task ended.
    class _LosesTheCall(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            if not terminated:
                raise RuntimeError("the session dropped the call")
            return super()._verify(trajectory, task, terminated=terminated, evidence=evidence)

    snapshots = _Snapshots()
    stream = TaskStream(
        lambda _name: _LosesTheCall(tasks=TASKS),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
        provenance=[snapshots],
    )
    async with stream:
        await stream.get_task()
        with pytest.raises(RuntimeError, match="dropped the call"):
            await stream.dispatch("noop", {})
        await stream.get_task()  # the stream ends the abandoned task itself
        await stream.dispatch(SUBMIT_TOOL, {"answer": "6"})

    lost, played = stream.results
    assert lost.closure == "finalize_error" and lost.score is None
    assert "the agent never played it out" in (lost.diagnostic or "")
    assert not stream.stopped, "one lost call ended a queue the agent could still play"
    # The span closed over that row, and was told the same thing the row says.
    assert lost.extensions["test.snapshots"] == {
        "dispensed": {"hash": "before-1"},
        "sealed": {"hash": "after-1", "closure": "finalize_error"},
    }
    assert [c.closure for c in snapshots.completed] == ["finalize_error", "sealed"]
    assert snapshots.completed[0].score is None, "an extension was shown a score nobody earned"
    assert played.score is not None and played.score.success is True
    assert [row.extensions for row in read_results(tmp_path / "prov")] == [
        lost.extensions,
        played.extensions,
    ]


async def test_an_extension_sees_only_the_redacted_outcome(tmp_path: Path) -> None:
    snapshots = _Snapshots()
    async with _stream(tmp_path, [1], provenance=[snapshots]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "6"})
    (completed,) = snapshots.completed
    assert completed == CompletedTask(
        position=0, closure="sealed", score=stream.results[0].score
    )
    # No episode, no env, no lease, no task index, no target — and it cannot be mutated.
    assert set(vars(completed)) == {"position", "closure", "score"}
    with pytest.raises(Exception):
        completed.position = 5  # type: ignore[misc]


async def test_extensions_compose_and_namespaces_must_be_unique(tmp_path: Path) -> None:
    class _Other(_Snapshots):
        namespace = "test.other"

    async with _stream(tmp_path, [0], provenance=[_Snapshots(), _Other()]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert set(stream.results[0].extensions) == {"test.snapshots", "test.other"}

    with pytest.raises(ValueError, match="unique"):
        _stream(tmp_path, [0], provenance=[_Snapshots(), _Snapshots()])


async def test_a_namespace_rebound_after_construction_cannot_take_another_extensions_key(
    tmp_path: Path,
) -> None:
    # `namespace` is an ordinary attribute of an object the *caller* owns. Checking it once and
    # then reading it again at every dispense validates one set of names and records under
    # another: two extensions agreeing on a name after construction would share one key, so the
    # later span overwrites the earlier one, one extension's `finalize` is never called at all,
    # and the row shows neither.
    first, second = _Snapshots(), _Snapshots()
    second.namespace = "test.other"  # distinct, and validated as distinct
    stream = _stream(tmp_path, [0], provenance=[first, second])
    first.namespace = second.namespace = "test.collision"
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    (row,) = stream.results
    assert set(row.extensions) == {"test.snapshots", "test.other"}
    assert len(first.completed) == 1 and len(second.completed) == 1
    assert read_results(tmp_path / "prov")[0].extensions == row.extensions


async def test_a_namespace_emptied_after_construction_does_not_key_a_row(
    tmp_path: Path,
) -> None:
    # The other half of the same check: an empty namespace is refused at construction, and
    # rebinding one to "" afterwards must not get it back in through the dispense.
    snapshots = _Snapshots()
    stream = _stream(tmp_path, [0], provenance=[snapshots])
    snapshots.namespace = ""
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    (row,) = stream.results
    assert set(row.extensions) == {"test.snapshots"}


@pytest.mark.parametrize(
    "namespace",
    [None, b"test.bytes", "", 0],
    ids=["none", "bytes", "empty", "int"],
)
async def test_a_namespace_must_be_non_empty_text(tmp_path: Path, namespace: Any) -> None:
    snapshots = _Snapshots()
    snapshots.namespace = namespace
    with pytest.raises(ValueError, match="non-empty string namespace"):
        _stream(tmp_path, [0], provenance=[snapshots])


class _Shifty(str):
    """A namespace whose equality and hash answer one way, then another.

    Every line of it is ordinary Python: ``str`` is subclassable, ``__eq__`` and ``__hash__``
    are overridable, and nothing obliges either to answer the same way twice."""

    collapsed = False
    _collapses_to: str

    def __new__(cls, value: str, collapses_to: str) -> "_Shifty":
        held = super().__new__(cls, value)
        held._collapses_to = collapses_to
        return held

    def _key(self) -> str:
        return self._collapses_to if _Shifty.collapsed else str.__str__(self)

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: Any) -> bool:
        return self._key() == (other._key() if isinstance(other, _Shifty) else other)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class _Exploding(str):
    """A namespace that hashes cleanly until something asks it to key a row."""

    armed = False

    def __hash__(self) -> int:
        if _Exploding.armed:
            raise RuntimeError("namespace hash exploded")
        return str.__hash__(self)


async def test_a_namespace_whose_identity_could_change_is_refused_where_it_is_checked(
    tmp_path: Path,
) -> None:
    # `isinstance` accepts a `str` subclass, and what the constructor then keeps is that same
    # object. Uniqueness is one round of `__hash__`/`__eq__` taken here, and the writes it
    # protects — `spans[namespace]`, `dispensed[namespace]`, `extensions[namespace]` — happen
    # once per dispense and once per seal, on the far side of every containment boundary this
    # module draws. So a namespace only has to answer differently *later*:
    #
    #   * two that pass as distinct and then compare equal share one key, so the second
    #     extension's output is filed under the first one's name, the first extension's
    #     `finalize` is never called, and the row lands with `success` true and no error at all;
    #   * one whose `__hash__` starts raising raises where the row is being keyed, outside the
    #     boundary that contains a failing extension — so a task whose score terminal had
    #     already succeeded gets no row.
    #
    # Neither is repairable after the fact, and both look like a clean run. The identity is
    # refused here instead, exactly as a queue entry's env is (see `_require_task_ref`).
    first, second = _Snapshots(), _Snapshots()
    first.namespace = _Shifty("test.one", "test.same")
    second.namespace = _Shifty("test.two", "test.same")
    assert first.namespace != second.namespace, "these two pass the uniqueness check as written"
    with pytest.raises(ValueError, match="non-empty string namespace"):
        _stream(tmp_path, [0], provenance=[first, second])

    detonator = _Snapshots()
    detonator.namespace = _Exploding("test.boom")
    assert hash(detonator.namespace) == hash("test.boom"), "it is well behaved right now"
    with pytest.raises(ValueError, match="non-empty string namespace"):
        _stream(tmp_path, [0], provenance=[detonator])

    assert not (tmp_path / "prov").exists(), "nothing was served"


# ----- failure isolation: begin -----


@pytest.mark.parametrize(
    "kwargs",
    [
        {"begin_error": RuntimeError("no snapshot for you")},
        {"begin_hangs": True},
        {"begin_returns": ["not", "an", "object"]},
        {"begin_returns": {"path": Path("/tmp")}},
        {"begin_returns": {"nan": float("nan")}},
        {"begin_returns": {"1": "text", 1: "integer"}},
        {"begin_returns": _RepeatsAName({"shot": "held"})},
        {"begin_returns": {"cycle": _SELF_REFERENTIAL}},
    ],
    ids=[
        "raises",
        "hangs",
        "not-an-object",
        "unserialisable",
        "nan",
        "non-text-name",
        "repeated-name",
        "contains-itself",
    ],
)
async def test_a_failed_span_refuses_the_dispense_and_owes_the_position(
    tmp_path: Path, kwargs: Dict[str, Any]
) -> None:
    # Nothing has been exposed when begin runs, so the honest failure is to not hand the task
    # out at all: no episode, no dispense record, and the position is still owed.
    bad = _Misbehaving(**kwargs)
    stream = _stream(tmp_path, [0, 1], provenance=[bad], provenance_timeout=0.05)
    async with stream:
        with pytest.raises(ProvenanceError, match="test.bad"):
            await stream.get_task()
        assert read_dispenses(tmp_path / "prov") == []
        assert stream.queue_info().consumed == 0
        assert stream.queue_info().remaining == 2
    assert stream.results == ()
    assert not (tmp_path / "prov" / "results.jsonl").exists()


class _UnrenderableFailure(Exception):
    """An extension failure whose own message is a second failure.

    Not exotic on purpose: an exception whose message is built lazily from state the failure has
    already torn down raises exactly here — when the harness asks for it, not when it was
    raised. The harness formats every failure it catches, and formatting one runs the
    extension's code again, outside the ``except`` that just contained it."""

    def __str__(self) -> str:
        raise RuntimeError("formatting this failure failed")


class _UnrenderableCancellation(asyncio.CancelledError):
    """The same, on the branch that tells an extension's cancellation from the seal's own."""

    def __str__(self) -> str:
        raise RuntimeError("formatting this failure failed")


class _NamelessMeta(type):
    @property
    def __name__(cls) -> str:  # type: ignore[override]
        raise RuntimeError("even the class name is not readable")


class _NamelessFailure(Exception, metaclass=_NamelessMeta):
    """The fallback's own fallback: a class whose ``__name__`` raises too.

    Deliberately not a test *parameter*. ``reprlib`` — which is what pytest's own safe-repr is
    built on — reads ``type(x).__name__`` before anything else, so an instance of this reaching
    a reported traceback frame's arguments crashes the reporter rather than the test. The one
    test that uses it therefore keeps it out of every frame it could be reported from."""

    def __str__(self) -> str:
        raise RuntimeError("formatting this failure failed")


def test_a_failure_whose_class_has_no_readable_name_still_renders() -> None:
    # The type-only fallback is not automatically safe either: `__name__` is an attribute of the
    # class, so a metaclass can make reading it the same second failure. The constant is what is
    # left, and it is still a string the row can carry.
    try:
        rendered = stream_module._rendered_failure(_NamelessFailure())
    except BaseException as escaped:  # noqa: BLE001 — the regression is that this escapes
        rendered = f"escaped as {type(escaped).__name__}"
    assert rendered == "<unrenderable>: <unrenderable message>"


async def test_a_span_whose_failure_cannot_be_formatted_still_refuses_as_a_provenance_error(
    tmp_path: Path,
) -> None:
    # `ProvenanceError` is what every caller of `get_task` is told a span that would not open
    # raises. Building its message formats the extension's exception, so an unformattable one
    # replaced the promised error with the formatter's — a caller catching `ProvenanceError` to
    # retire the position saw an unrelated `RuntimeError` escape instead.
    bad = _Misbehaving(begin_error=_UnrenderableFailure())
    stream = _stream(tmp_path, [0, 1], provenance=[bad])
    async with stream:
        with pytest.raises(ProvenanceError) as refused:
            await stream.get_task()
        assert "test.bad" in str(refused.value)
        assert "_UnrenderableFailure: <unrenderable message>" in str(refused.value)
        assert read_dispenses(tmp_path / "prov") == []
        assert stream.queue_info().remaining == 2
    assert stream.results == ()


# ----- failure isolation: finalize -----


@pytest.mark.parametrize(
    "kwargs",
    [
        {"finalize_error": RuntimeError("snapshot failed")},
        {"finalize_hangs": True},
        {"finalize_returns": "not an object"},
        {"finalize_returns": {"inf": float("inf")}},
        {"finalize_returns": {"cycle": _SELF_REFERENTIAL}},
    ],
    ids=["raises", "hangs", "not-an-object", "not-serialisable", "contains-itself"],
)
async def test_a_failing_finalize_can_neither_suppress_nor_change_a_row(
    tmp_path: Path, kwargs: Dict[str, Any]
) -> None:
    # The episode is already sealed and scored by now, so the extension gets a structured error
    # in its own namespace and the row is written regardless.
    bad = _Misbehaving(**kwargs)
    stream = _stream(tmp_path, [0], provenance=[bad], provenance_timeout=0.05)
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    (row,) = stream.results
    assert len(read_results(tmp_path / "prov")) == 1  # not suppressed, not duplicated
    assert row.score is not None and row.score.success is True  # not changed
    entry = row.extensions["test.bad"]
    assert entry["dispensed"] == {"ok": True}
    assert "sealed" not in entry
    assert entry["error"]


@pytest.mark.parametrize("returned", _COERCIBLE_NAMES, ids=_COERCIBLE_IDS)
async def test_a_name_that_is_not_text_is_refused_rather_than_coerced_into_one(
    tmp_path: Path, returned: Any
) -> None:
    # "Strict JSON" is the documented contract for what an extension hands back, and the encoder
    # that proves the *values* is deliberately permissive about the **names**: it writes `1`,
    # `True` and `1.0` as names a JSON object can hold rather than refusing them, and a `str`
    # subclass renders as its plain text however its own `__eq__` answers. So two keys the
    # extension's dict holds apart become one name, the decoder keeps whichever was written last,
    # and the other value is gone — with no error on the row, `success` true, and the extension
    # told its output was recorded. A record that quietly holds half of what it was given is
    # worse than one that says it holds nothing, so the name is refused and the refusal is what
    # lands.
    bad = _Misbehaving(finalize_returns=returned)
    stream = _stream(tmp_path, [0], provenance=[bad])
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    (row,) = stream.results
    assert row.score is not None and row.score.success is True, "the row is unchanged"
    entry = row.extensions["test.bad"]
    assert entry["dispensed"] == {"ok": True}
    assert "sealed" not in entry, "a coerced name was recorded as if it were the value given"
    assert "must be text" in entry["error"]
    assert _COERCED not in json.dumps(entry), "half the payload was recorded anyway"
    assert read_results(tmp_path / "prov")[0].extensions == row.extensions


async def test_a_mapping_that_answers_for_itself_cannot_rewrite_what_it_recorded(
    tmp_path: Path,
) -> None:
    # The narrow case the name check alone would not catch, and the worst one: every name here is
    # exact text, and the *fabrication* comes from the mapping rather than the encoder. The
    # encoder asks a `dict` subclass for its `items()`, so what is written is that answer and not
    # what the object holds — and when the answer repeats a name, the decoder keeps one of them.
    # The row would then carry a value that is neither what the object holds nor all of what it
    # reported. The read is taken once and refused when it names anything twice.
    held = _RepeatsAName({"shot": "held"})
    assert held["shot"] == "held", "what the object holds"
    assert dict(held.items()) == {"shot": "second"}, "what it says it holds"

    bad = _Misbehaving(finalize_returns=held)
    stream = _stream(tmp_path, [0], provenance=[bad])
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    (row,) = stream.results
    assert row.score is not None and row.score.success is True
    entry = row.extensions["test.bad"]
    assert "sealed" not in entry
    assert "twice" in entry["error"]
    assert "first" not in json.dumps(entry) and "second" not in json.dumps(entry)


@pytest.mark.parametrize(
    "error, expected",
    [
        (_UnrenderableFailure(), "_UnrenderableFailure: <unrenderable message>"),
        (_UnrenderableCancellation(), "_UnrenderableCancellation: <unrenderable message>"),
    ],
    ids=["error", "cancelled"],
)
async def test_a_failure_the_row_cannot_format_still_cannot_suppress_the_row(
    tmp_path: Path, error: BaseException, expected: str
) -> None:
    # The containment above is written in a message, and writing it runs the extension's code a
    # second time — `__str__` belongs to whoever raised. That second exception is not the one
    # the handler caught, so it walked out of `_finalize_spans` and `_compose_row` on an episode
    # already sealed: no row was composed, the run reported "a dispensed task could not be
    # recorded" — the storage failure it was not — and `results.jsonl` did not exist at all.
    # Both arms are covered because both format: the cancellation branch is the one an
    # extension reaches by raising `CancelledError`, and it strands the seal the same way.
    good = _Snapshots()
    bad = _Misbehaving(finalize_error=error)
    stream = _stream(tmp_path, [0], provenance=[bad, good])
    async with stream:
        await stream.get_task()
        answer = await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
        assert answer.content[0].text == _TASK_OVER  # type: ignore[union-attr]

    (row,) = stream.results
    assert len(read_results(tmp_path / "prov")) == 1, "the row was suppressed or duplicated"
    assert row.score is not None and row.score.success is True, "the outcome was changed"
    entry = row.extensions["test.bad"]
    assert "sealed" not in entry
    # The class is still named, which is the part of a failure that never needed the extension.
    assert entry["error"] == expected
    assert row.extensions["test.snapshots"]["sealed"]["closure"] == "sealed"
    assert bad.finalize_calls == 1
    assert not stream.stopped, "an extension stopped the stream"
    assert reconcile(tmp_path / "prov") == [], "the durable dispense went unanswered"


async def test_one_failing_extension_does_not_take_out_the_others(tmp_path: Path) -> None:
    good = _Snapshots()
    bad = _Misbehaving(finalize_error=RuntimeError("boom"))
    async with _stream(tmp_path, [0], provenance=[bad, good]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    extensions = stream.results[0].extensions
    assert extensions["test.bad"]["error"]
    assert extensions["test.snapshots"]["sealed"] == {"hash": "after-1", "closure": "sealed"}


async def test_a_span_is_finalized_exactly_once_when_paths_race(tmp_path: Path) -> None:
    # The deadline, a terminal call and the drain all race for the same task. Exactly one seal
    # happens, so exactly one finalize does.
    bad = _Misbehaving()
    stream = _stream(tmp_path, [0], provenance=[bad], deadline=0.05)
    async with stream:
        await stream.get_task()
        await asyncio.sleep(0.15)  # let the deadline fire while the drain is coming
    assert bad.finalize_calls == 1
    assert len(stream.results) == 1
    assert len(read_results(tmp_path / "prov")) == 1


async def test_a_dispense_waits_for_a_seal_that_is_still_composing_its_row(
    tmp_path: Path,
) -> None:
    # A seal is claimed long before it has anything to show for it: the episode is still open,
    # its closure is undecided and no row exists. A dispense that only skips entries *no* seal
    # has claimed steps over all of that — it hands out a second episode while the first is
    # still ending, and writes its durable dispense ahead of the row the running seal is about
    # to land.
    entered, release = asyncio.Event(), asyncio.Event()

    class _Blocking:
        namespace = "test.blocking"

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            entered.set()
            await release.wait()
            return {}

    stream = _stream(tmp_path, [0, 1], provenance=[_Blocking()], provenance_timeout=None)
    await stream.get_task()
    call = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    await asyncio.wait_for(entered.wait(), timeout=5)
    nxt = asyncio.ensure_future(stream.get_task())
    await asyncio.sleep(0.05)

    assert not nxt.done(), "a task was dispensed while the previous seal was still running"
    assert stream.results == ()
    assert len(read_dispenses(tmp_path / "prov")) == 1

    release.set()
    assert await asyncio.wait_for(nxt, timeout=5) is not None
    await call
    assert len(stream.results) == 1, "the first row must land before the second task is out"
    assert len(read_dispenses(tmp_path / "prov")) == 2
    await stream.aclose()


async def test_a_row_write_that_failed_does_not_finalize_a_span_a_second_time(
    tmp_path: Path,
) -> None:
    # The other way a seal is retried. A failed durable append hands the claim back on purpose,
    # so the row is not lost to one bad write — but what is retryable is the *write*. Composing
    # the row again would call `finalize` again, and its snapshots and commits already happened
    # out in the world with no row to say they happened twice. Repeated closes must not keep
    # spending them either.
    prov = tmp_path / "prov"
    prov.mkdir(parents=True)
    (prov / "results.jsonl").mkdir()  # a directory: no row can be appended to it

    bad = _Misbehaving()
    stream = TaskStream(
        lambda _name: _FixtureScoreEnv(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=prov,
        provenance=[bad],
    )
    await stream.get_task()
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert bad.finalize_calls == 1
    for attempt in ("first", "second", "third"):
        with pytest.raises(RuntimeError, match="record is incomplete"):
            await stream.aclose()
        assert bad.finalize_calls == 1, f"the {attempt} close finalized the span again"
    assert stream.results == ()


async def test_a_retried_row_write_still_records_the_ending_the_stream_imposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # And composing the row again would not merely repeat the callbacks: it would re-read an
    # episode the first attempt already force-terminated, so `not episode.terminated` is false
    # the second time and the drain the stream imposed is adopted as an ending the agent
    # produced — a scored closure, from a storage failure. The row a transiently failed write
    # finally lands must be the row the control lands.
    async def drained(prov: Path) -> tuple[_Misbehaving, List[str]]:
        bad = _Misbehaving()
        stream = TaskStream(
            lambda _name: _FixtureScoreEnv(tasks=TASKS),
            [TaskRef(ENV_NAME, 0)],
            prov_dir=prov,
            provenance=[bad],
        )
        await stream.get_task()  # the agent stops short; the drain ends the task for it
        for _ in range(2):  # the second close is the retry the hand-back exists for
            try:
                await stream.aclose()
            except RuntimeError:
                pass
        return bad, [row.closure for row in read_results(prov)]

    control, control_closures = await drained(tmp_path / "clean")

    real_append = stream_module._append_jsonl
    left = [1]  # one transient failure, then the storage works again

    def flaky(path: Path, record: Dict[str, Any], *, durable: bool = False) -> None:
        if path.name == "results.jsonl" and left[0]:
            left[0] -= 1
            raise OSError("no space left on device")
        real_append(path, record, durable=durable)

    monkeypatch.setattr(stream_module, "_append_jsonl", flaky)
    retried, retried_closures = await drained(tmp_path / "flaky")

    assert left == [0], "the append never failed, so nothing was retried"
    assert control_closures == ["drained"]
    assert retried_closures == control_closures
    assert control.finalize_calls == 1
    assert retried.finalize_calls == 1  # the retry wrote the row it had already composed


async def test_a_cancelled_seal_does_not_finalize_a_span_a_second_time(tmp_path: Path) -> None:
    # Exactly-once has to survive the caller going away. A shutdown cancelled while an extension
    # is inside `finalize` must not let the retry start that callback again — the first call's
    # side effects already happened out in the world, and no row records that it ran twice.
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: List[Closure] = []

    class _Blocking:
        namespace = "test.blocking"

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            seen.append(completed.closure)
            entered.set()
            await release.wait()
            return {"calls": len(seen)}

    stream = _stream(tmp_path, [0], provenance=[_Blocking()], provenance_timeout=None)
    await stream.__aenter__()
    await stream.get_task()
    closing = asyncio.ensure_future(stream.aclose())
    await entered.wait()  # inside the extension's finalize
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()
    await stream.aclose()  # the retry joins the seal already running, it does not restart it

    assert seen == ["drained"]
    (row,) = stream.results
    assert row.closure == "drained"
    assert row.extensions["test.blocking"]["sealed"] == {"calls": 1}
    assert len(read_results(tmp_path / "prov")) == 1


# ----- containment -----


@pytest.mark.parametrize(
    "tamper",
    [
        lambda feedback: feedback.clear(),
        lambda feedback: feedback.__setitem__(slice(None), [{"name": "correct", "value": False}]),
        lambda feedback: [item.update({"value": "TAMPERED"}) for item in feedback],
        lambda feedback: feedback.append({"not": object()}),
    ],
    ids=["clear", "replace", "mutate-item", "append-unserialisable"],
)
async def test_an_extension_cannot_rewrite_a_row_through_the_summary_it_is_handed(
    tmp_path: Path, tamper: Any
) -> None:
    # `frozen=True` on `CompletedTask` and `Score` stops an attribute being rebound and nothing
    # else: the list behind `Score.feedback` is the row's own `observed`, and its items are the
    # row's items. An extension handed that object could empty it, rewrite it in place, or append
    # a value the durable append cannot serialise — which took the whole row down, not just the
    # extension's namespace. Each span gets its own detached copy instead.
    class _Tampering:
        namespace = "test.tampering"

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            assert completed.score is not None
            tamper(completed.score.feedback)
            return {"tampered": True}

    class _Watching:
        """Runs after the tamperer: one extension may not decide what the next one observes."""

        namespace = "test.watching"

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            assert completed.score is not None
            return {"observed": list(completed.score.feedback)}

    async with _stream(tmp_path, [0], provenance=[_Tampering(), _Watching()]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    intact = [{"name": "correct", "value": True, "level": "episode"}]
    (row,) = stream.results
    assert row.score is not None
    assert row.score.success is True
    assert row.score.feedback == intact
    assert row.observed == intact
    # Durable, not just in memory: the row is written after the extensions run, so a mutation
    # that reached it would have been written rather than merely observed.
    (durable,) = read_results(tmp_path / "prov")
    assert durable.score is not None and durable.score.feedback == intact
    assert durable.observed == intact
    assert not stream.stopped
    # The second extension saw the env's items, not the first extension's edit of them.
    assert row.extensions["test.watching"]["sealed"] == {"observed": intact}
    assert row.extensions["test.tampering"]["sealed"] == {"tampered": True}


@pytest.mark.parametrize(
    ("shape", "timeout"),
    [("raised", None), ("from-a-cancelled-child", None), ("raised", 30.0)],
    ids=["raised", "from-a-cancelled-child", "raised-under-a-timeout"],
)
async def test_a_cancelling_extension_cannot_strand_the_task_it_spanned(
    tmp_path: Path, shape: str, timeout: Optional[float]
) -> None:
    # `CancelledError` is a BaseException, so it walks through the `except Exception` that
    # contains every other extension failure — and because the seal runs as its own task, one
    # raised inside a callback cancels *the seal*: no row, no stop, an entry left sealed with a
    # cancelled claim that every retry re-awaits, and a durable dispense that reconciles as a
    # crash after an orderly shutdown. It is the extension's failure and is recorded as one.
    #
    # Run with and without `provenance_timeout`, because the bound is where this gets subtle: a
    # bound that expired a hung callback by cancelling the seal task would move the very count
    # the two cases are told apart by, on a path that has nothing to do with either of them.
    class _Cancelling:
        namespace = "test.cancelling"

        def __init__(self) -> None:
            self.calls = 0

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            self.calls += 1
            if shape == "raised":
                raise asyncio.CancelledError()
            child = asyncio.ensure_future(asyncio.sleep(60))
            await asyncio.sleep(0)
            child.cancel()
            await child  # the cancellation propagates out of the callback
            raise AssertionError("unreachable")

    extension = _Cancelling()
    stream = _stream(tmp_path, [0], provenance=[extension], provenance_timeout=timeout)
    async with stream:
        await stream.get_task()
        answer = await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    # The agent still gets the one redacted payload every other outcome gets.
    assert answer.content[0].text == _TASK_OVER  # type: ignore[union-attr]
    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True
    assert row.extensions["test.cancelling"]["error"].startswith("CancelledError")
    assert "sealed" not in row.extensions["test.cancelling"]
    assert len(read_results(tmp_path / "prov")) == 1
    assert extension.calls == 1
    # The stream is not stopped, and the durable dispense is answered by a row rather than
    # reconciling as a crash the orderly shutdown never had.
    assert not stream.stopped
    assert len(read_dispenses(tmp_path / "prov")) == 1
    assert reconcile(tmp_path / "prov") == []


async def test_cancelling_the_seal_task_itself_is_still_not_swallowed(tmp_path: Path) -> None:
    # The other half of the rule above, and the reason it is decided by `Task.cancelling()`
    # rather than by the exception's type: a cancellation actually requested against the seal
    # task is not an extension failure, and must not be turned into one. Swallowing it would
    # write a row for a seal whose owner cancelled it.
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: List[Closure] = []

    class _Reporting:
        namespace = "test.reporting"

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            calls.append(completed.closure)
            entered.set()
            await release.wait()
            return {}

    stream = _stream(tmp_path, [0], provenance=[_Reporting()], provenance_timeout=None)
    await stream.__aenter__()
    await stream.get_task()
    call = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    await entered.wait()
    # The seal task itself, taken from the entry that holds the claim — the stream's own handle
    # on it, rather than whatever task the callback happens to be running in.
    (live,) = stream._live.values()
    sealing = live.sealing
    assert sealing is not None
    sealing.cancel()  # requested against the seal itself, from outside the extension
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert sealing.cancelled()
    assert calls == ["sealed"]  # not restarted, so not finalized twice either
    assert stream.results == ()
    assert not stream.stopped  # cancellation is never an integrity failure
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    with pytest.raises(asyncio.CancelledError):
        await stream.aclose()  # the claim is not restarted; the release still runs


async def test_a_cancelling_extension_cannot_cancel_the_dispense_either(tmp_path: Path) -> None:
    # The same containment on the way in. A `begin` that raises `CancelledError` is that
    # extension refusing to open its span — the loud `ProvenanceError` every other begin failure
    # produces — and not a cancellation of whoever is dispensing.
    bad = _Misbehaving(begin_error=asyncio.CancelledError())
    stream = _stream(tmp_path, [0, 1], provenance=[bad])
    async with stream:
        with pytest.raises(ProvenanceError, match="test.bad"):
            await stream.get_task()
        assert read_dispenses(tmp_path / "prov") == []
        assert stream.queue_info().remaining == 2
    assert stream.results == ()


async def _until(ready: Callable[[], bool], timeout: float = 5.0) -> None:
    """Let the loop run until an abandoned callback has got where the test needs it."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not ready():
        assert asyncio.get_running_loop().time() < deadline, "the callback never got there"
        await asyncio.sleep(0.005)


class _Resistant:
    """An extension that catches the cancellation its bound expires it with, and carries on.

    ``CancelledError`` is an ordinary catchable exception, so this is what a bound has to hold
    against: not a callback that is slow, but one that declines to stop being slow. It keeps
    running until the test releases it, so the bound is under test while the callback is still
    going rather than after it has quietly finished.
    """

    namespace = "test.resistant"

    def __init__(self, *, where: str, release: asyncio.Event) -> None:
        self._where = where
        self._release = release
        self.resisted = 0
        self.returned = 0

    async def _resist(self) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.resisted += 1
            await self._release.wait()  # still running, and still not returning
            self.returned += 1

    async def begin(self, ref: TaskRef) -> ProvenanceSpan:
        if self._where == "begin":
            await self._resist()
        return self  # type: ignore[return-value]

    @property
    def dispensed(self) -> Dict[str, Any]:
        return {"ok": True}

    async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
        if self._where == "finalize":
            await self._resist()
        return {"ok": True}


async def test_a_callback_that_resists_cancellation_still_stops_at_its_bound(
    tmp_path: Path,
) -> None:
    # The bound is on how long the stream *waits*, and that is the only bound it can be: the
    # callback cannot be killed, only cancelled, and cancellation is catchable. So expiring it by
    # cancelling the task the callback runs in and then awaiting that cancellation bounds
    # nothing — the callback catches it and either returns late, which is accepted as a success
    # it did not earn, or never returns, which holds the dispense lock (the whole queue) or the
    # seal (so `aclose` never returns either). Here it is still inside the callback, refusing to
    # end, for every assertion below.
    release = asyncio.Event()
    resistant = _Resistant(where="begin", release=release)
    stream = _stream(tmp_path, [0, 1], provenance=[resistant], provenance_timeout=0.01)
    try:
        with pytest.raises(ProvenanceError, match="test.resistant"):
            await asyncio.wait_for(stream.get_task(), timeout=5)
        await _until(lambda: resistant.resisted == 1)
        assert resistant.returned == 0, "the bound waited for the callback after all"
        # The queue is not wedged behind it: nothing was dispensed, the position is still owed,
        # and the next caller is answered rather than queued behind the callback.
        assert read_dispenses(tmp_path / "prov") == []
        assert stream.queue_info().remaining == 2
        with pytest.raises(ProvenanceError, match="test.resistant"):
            await asyncio.wait_for(stream.get_task(), timeout=5)
    finally:
        release.set()
        await _until(lambda: resistant.returned == resistant.resisted)


async def test_a_resisting_finalize_cannot_hold_the_seal_or_land_its_late_value(
    tmp_path: Path,
) -> None:
    # The same on the way out, where the wedge is worse: the seal is what `aclose` is waiting
    # for. The row lands at the bound with the callback recorded as failed, and the value that
    # callback finally produces is not read — a row is not rewritten by a callback that missed it.
    release = asyncio.Event()
    resistant = _Resistant(where="finalize", release=release)
    stream = _stream(tmp_path, [0], provenance=[resistant], provenance_timeout=0.01)
    try:
        await stream.get_task()
        answer = await asyncio.wait_for(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}), timeout=5)
        await asyncio.wait_for(stream.aclose(), timeout=5)
        await _until(lambda: resistant.resisted == 1)
        assert resistant.returned == 0, "the row waited for the callback after all"
    finally:
        release.set()
        await _until(lambda: resistant.returned == 1)
    assert resistant.returned == 1  # it did finish, well after the row it was meant to be on

    assert answer.content[0].text == _TASK_OVER  # type: ignore[union-attr]
    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True
    entry = row.extensions["test.resistant"]
    assert entry["error"].startswith("TimeoutError")
    assert "sealed" not in entry
    assert read_results(tmp_path / "prov")[0].extensions == row.extensions
    assert not stream.stopped
    assert reconcile(tmp_path / "prov") == []


async def test_an_extension_runs_in_its_own_task_and_can_only_cancel_that(
    tmp_path: Path,
) -> None:
    # What isolating the callback buys beyond the bound: `asyncio.current_task()` inside a
    # callback is the callback's own task, so an extension reaching for it cancels itself. That
    # used to be the seal, where cancelling it left the entry claimed with no row, no stop and a
    # durable dispense that reconciled an orderly shutdown as a crash.
    inside: List["asyncio.Task[Any]"] = []
    sealing: List[Optional["asyncio.Task[Any]"]] = []

    class _SelfCancelling:
        namespace = "test.selfish"

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            task = asyncio.current_task()
            assert task is not None
            inside.append(task)
            (live,) = stream._live.values()
            sealing.append(live.sealing)
            task.cancel()
            await asyncio.sleep(0)
            raise AssertionError("unreachable")

    stream = _stream(tmp_path, [0], provenance=[_SelfCancelling()], provenance_timeout=None)
    async with stream:
        await stream.get_task()
        answer = await asyncio.wait_for(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}), timeout=5)

    assert sealing[0] is not None and inside[0] is not sealing[0]
    assert inside[0].cancelled() and not sealing[0].cancelled()
    # Contained exactly like every other extension failure: the row lands, the agent gets the
    # one redacted payload, and the dispense is answered.
    assert answer.content[0].text == _TASK_OVER  # type: ignore[union-attr]
    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.extensions["test.selfish"]["error"].startswith("CancelledError")
    assert not stream.stopped
    assert reconcile(tmp_path / "prov") == []


class _Unwrappable(str):
    """A feedback value that is a JSON scalar to the record and a landmine to anything that
    tries to *copy* or *render* it.

    Reachable, not exotic: ``shogym.feedback.wire`` types a feedback value as ``float | bool |
    str``, the feedback models do not validate on assignment (``dump_item`` says so in as many
    words), and the wire dict carries the object itself. So an env that builds a value one way
    and rewrites it another puts a scalar *subclass* on the row, and it serialises like any
    other string."""

    calls: List[str] = []

    def __deepcopy__(self, memo: Any) -> "_Unwrappable":
        _Unwrappable.calls.append("__deepcopy__")
        raise RuntimeError("this value refuses to be copied")

    def __copy__(self) -> "_Unwrappable":
        _Unwrappable.calls.append("__copy__")
        raise RuntimeError("this value refuses to be copied")

    def __reduce__(self) -> Any:
        _Unwrappable.calls.append("__reduce__")
        raise RuntimeError("this value refuses to be pickled")

    def __repr__(self) -> str:
        _Unwrappable.calls.append("__repr__")
        raise RuntimeError("this value refuses to be rendered")


def _publishes_an_unwrappable_value() -> Callable[[str], _FixtureScoreEnv]:
    """An env whose ordinary, non-summary feedback carries one such value."""

    class _Published(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> Any:
            feedback = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
            if terminated:
                item = EpisodeFeedback(name="note", value="ordinary")
                item.value = _Unwrappable("ordinary")
                feedback.episode.append(item)
            return feedback

    return lambda _name: _Published(tasks=TASKS)


@pytest.mark.parametrize("with_provenance", [False, True], ids=["off", "on"])
async def test_a_summary_value_the_harness_cannot_copy_still_lands_its_row(
    tmp_path: Path, with_provenance: bool
) -> None:
    # The row is authoritative and an extension may not suppress it — and *enabling* one is not
    # allowed to suppress it either. Detaching the summary for a span used to deep-copy it, above
    # the boundary that contains a failing extension, so the env's own value object got to decide
    # whether the seal survived: with no extension configured this task scores and the run is
    # clean, and adding a no-op extension turned it into no row at all, a stopped stream, and a
    # failure the harness only meets at `aclose` — the redacted constant still went to the agent.
    _Unwrappable.calls.clear()
    spans = _Snapshots()
    stream = TaskStream(
        _publishes_an_unwrappable_value(),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        provenance=[spans] if with_provenance else (),
    )
    await stream.get_task()
    answer = await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert answer.content[0].text == _TASK_OVER  # type: ignore[union-attr]
    await stream.aclose()  # clean, both ways: nothing here is an integrity failure

    assert not stream.stopped
    (row,) = read_results(tmp_path / "prov")
    assert row.closure == "sealed" and row.score is not None and row.score.success is True
    assert {item["name"]: item["value"] for item in row.observed} == {
        "correct": True,
        "note": "ordinary",
    }
    assert reconcile(tmp_path / "prov") == []
    # Nothing ran the value's own code — not to copy it, not to describe it. Serialising a JSON
    # scalar reads its concrete representation, which is why the detachment is total: there is
    # no hook left for a value to fail or block in.
    assert _Unwrappable.calls == []


async def test_a_span_is_handed_the_summary_in_the_form_the_row_keeps_it(
    tmp_path: Path,
) -> None:
    # What detachment now hands over is the row re-parsed out of its own JSON, so the copy is
    # plain data: the subclass identity is gone, which is exactly the behaviour that could reach
    # back into the harness — and is also exactly what the durable row does not preserve either,
    # so an extension reasoning about it was reasoning about something the file never held.
    _Unwrappable.calls.clear()
    seen: Dict[str, Any] = {}

    class _Inspecting:
        namespace = "test.inspecting"

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            assert completed.score is not None
            value = completed.score.feedback[-1]["value"]
            seen["type"] = type(value).__name__
            seen["value"] = str(value)
            return {}

    stream = TaskStream(
        _publishes_an_unwrappable_value(),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        provenance=[_Inspecting()],
    )
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    assert seen == {"type": "str", "value": "ordinary"}
    assert _Unwrappable.calls == []
    assert stream.results[0].extensions["test.inspecting"]["sealed"] == {}


async def test_a_summary_no_span_can_copy_still_finalizes_each_span_exactly_once(
    tmp_path: Path,
) -> None:
    # The second half of the same defect, and the worse one. A raise from detaching escaped
    # `_finalize_spans` entirely, so `_run_seal` never got a row to retain and the claim went
    # back for a drain to retry — and the retry re-enters `_compose_row`, which re-drives the
    # terminal on an episode the first attempt already ended and calls `finalize` a second time
    # on every span that had already closed, with its side effects already out in the world.
    _Unwrappable.calls.clear()
    first, second = _Snapshots(), _Snapshots()
    second.namespace = "test.snapshots.2"  # type: ignore[misc]
    stream = TaskStream(
        _publishes_an_unwrappable_value(),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        provenance=[first, second],
    )
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    assert len(first.completed) == 1 and len(second.completed) == 1
    assert len(first.begins) == 1 and len(second.begins) == 1
    (row,) = read_results(tmp_path / "prov")
    assert set(row.extensions) == {"test.snapshots", "test.snapshots.2"}
    assert all("error" not in entry for entry in row.extensions.values())
    assert not stream.stopped


async def test_returned_provenance_is_detached_from_the_extension(tmp_path: Path) -> None:
    # A returned dict is round-tripped through strict JSON, so a later mutation of the object
    # the extension kept cannot rewrite a row that was already recorded.
    shared: Dict[str, Any] = {"hash": "after"}

    class _Aliasing(_Snapshots):
        namespace = "test.alias"

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {"hash": "before"}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            return shared

    async with _stream(tmp_path, [0], provenance=[_Aliasing()]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    shared["hash"] = "TAMPERED"
    assert stream.results[0].extensions["test.alias"]["sealed"] == {"hash": "after"}
    assert json.loads((tmp_path / "prov" / "results.jsonl").read_text())["extensions"][
        "test.alias"
    ]["sealed"] == {"hash": "after"}


async def test_extension_callbacks_run_with_the_registry_free(tmp_path: Path) -> None:
    # An extension is arbitrary user code that may block for a long time; holding the stream's
    # registry lock across it would wedge every other request behind it.
    entered = asyncio.Event()
    release = asyncio.Event()
    observed: Dict[str, Any] = {}

    class _Blocking(_Snapshots):
        namespace = "test.blocking"

        async def begin(self, ref: TaskRef) -> ProvenanceSpan:
            return self  # type: ignore[return-value]

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: CompletedTask) -> Dict[str, Any]:
            entered.set()
            await release.wait()
            return {}

    stream = _stream(tmp_path, [0], provenance=[_Blocking()], provenance_timeout=None)
    async with stream:
        await stream.get_task()
        sealing = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        await entered.wait()
        # The registry answers while the extension is still blocked.
        probe = await asyncio.wait_for(stream.dispatch("noop", {}), timeout=1.0)
        observed["probe"] = json.loads(probe.content[0].text)  # type: ignore[union-attr]
        release.set()
        await sealing
    assert observed["probe"]["error"] == "no_active_task"
    assert len(stream.results) == 1


async def test_rejects_a_provenance_timeout_that_cannot_be_enforced(tmp_path: Path) -> None:
    # Nonpositive is the obvious half; the non-finite half is worse and splits in two. A timer
    # treats infinity as no bound at all, so the hang this argument exists to stop goes unbounded
    # again — and it expires against NaN immediately, so every extension on every task would be
    # filed as having timed out while the run reported fine. `None` is how a caller asks to wait
    # indefinitely, and it should be the only way to get it.
    for value in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="provenance_timeout must be a finite positive"):
            _stream(tmp_path, [0], provenance_timeout=value)


# ----- durability: the half of a span a crash can leave behind -----
#
# `broker_abort` is a dispensed-task outcome like any other, and it is the one outcome no
# orderly path can produce. The dispense record is the only artifact guaranteed to exist for
# it, so anything an extension had already produced when the task went out has to be on that
# record or it is lost — while the snapshot it names is still sitting out in the world.


async def test_an_unsealed_dispense_still_carries_what_its_extension_observed(
    tmp_path: Path,
) -> None:
    prov = tmp_path / "prov"
    stream = _stream(tmp_path, [0], provenance=[_Snapshots()])
    await stream.get_task()  # and then nothing at all: the process "dies" holding the task

    (record,) = read_dispenses(prov)
    assert record["extensions"] == {"test.snapshots": {"hash": "before-1"}}
    assert not (prov / "results.jsonl").exists(), "nothing orderly ran; only the dispense exists"

    (row,) = reconcile(prov)
    assert row.closure == "broker_abort" and row.score is None
    assert row.extensions == {"test.snapshots": {"dispensed": {"hash": "before-1"}}}
    assert row.lease == record["lease"], "the provenance stays attached to its own dispense"


async def test_a_reconciled_entry_is_told_apart_from_a_sealed_one_without_guessing(
    tmp_path: Path,
) -> None:
    # Two dispenses of one extension, one sealed and one abandoned. A consumer must not have to
    # infer which is which: `closure` says it for the row, and the namespace members follow —
    # exactly one of `sealed`/`error` on a row the stream wrote, neither on one `reconcile`
    # built. Nothing is invented for the half that never happened.
    prov = tmp_path / "prov"
    stream = _stream(tmp_path, [0, 1], provenance=[_Snapshots()])
    await stream.get_task()
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})  # position 0, sealed
    await stream.get_task()  # position 1, dispensed and then abandoned

    (orderly,) = read_results(prov)
    (reconciled,) = reconcile(prov)
    assert (orderly.closure, reconciled.closure) == ("sealed", "broker_abort")

    sealed_entry = orderly.extensions["test.snapshots"]
    abandoned_entry = reconciled.extensions["test.snapshots"]
    assert sealed_entry == {
        "dispensed": {"hash": "before-1"},
        "sealed": {"hash": "after-1", "closure": "sealed"},
    }
    assert abandoned_entry == {"dispensed": {"hash": "before-2"}}
    assert len({"sealed", "error"} & set(sealed_entry)) == 1
    assert not {"sealed", "error"} & set(abandoned_entry)
    # And the pairing is per DISPENSE, not per task index: the abandoned row carries its own
    # span's observation rather than the sealed one's.
    assert sealed_entry["dispensed"] != abandoned_entry["dispensed"]


async def test_a_replayed_position_does_not_take_over_the_snapshot_the_crash_left(
    tmp_path: Path,
) -> None:
    # Where the disconnection bites hardest. A resumed run replays the abandoned position and its
    # extension opens a *new* span and takes a *new* snapshot, so with nothing on the dispense
    # the crashed attempt is a `broker_abort` that cannot say which of the two snapshots on disk
    # was its. Both rows name their own, and the leases keep them apart.
    prov = tmp_path / "prov"
    snapshots = _Snapshots()
    crashed = _stream(tmp_path, [0], provenance=[snapshots])
    await crashed.get_task()  # the process "dies" here; the stream is deliberately not closed
    (abandoned,) = read_dispenses(prov)

    replay = TaskStream(
        lambda _name: _FixtureScoreEnv(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=prov,
        resume=True,
        provenance=[snapshots],
    )
    async with replay:
        await replay.get_task()
        await replay.dispatch(SUBMIT_TOOL, {"answer": "4"})

    (replayed,) = replay.results
    (reconciled,) = reconcile(prov)
    assert reconciled.position == replayed.position == 0
    assert reconciled.lease == abandoned["lease"] != replayed.lease
    assert reconciled.extensions == {"test.snapshots": {"dispensed": {"hash": "before-1"}}}
    assert replayed.extensions["test.snapshots"] == {
        "dispensed": {"hash": "before-2"},
        "sealed": {"hash": "after-2", "closure": "sealed"},
    }


async def test_a_dispense_recorded_without_provenance_reconciles_to_an_empty_map(
    tmp_path: Path,
) -> None:
    # Two shapes, one answer. A stream with no extensions writes an empty map, and a record
    # written before this field existed has no member at all — neither may make `reconcile`
    # raise, because reading a directory a crashed run left behind is the one job it has.
    prov = tmp_path / "prov"
    stream = _stream(tmp_path, [0])
    await stream.get_task()
    (record,) = read_dispenses(prov)
    assert record["extensions"] == {}
    assert reconcile(prov)[0].extensions == {}

    older = tmp_path / "older"
    older.mkdir()
    predating = {"lease": "x", "seq": 1, "position": 0, "env": ENV_NAME, "task_idx": 0}
    (older / "dispenses.jsonl").write_text(json.dumps(predating) + "\n")
    (row,) = reconcile(older)
    assert row.closure == "broker_abort" and row.extensions == {}


async def test_a_dispense_whose_commit_failed_leaves_no_provenance_behind_either(
    tmp_path: Path,
) -> None:
    # The payload rides on the record, so it is committed by the same append and rolled back by
    # the same failure. A dispense whose commit could not be confirmed must leave the log exactly
    # as it found it — an extension's observation surviving a dispense that did not would be a
    # `broker_abort` manufactured against a task nobody was ever handed.
    prov = tmp_path / "prov"
    target = prov / "dispenses.jsonl"
    real_fsync = os.fsync

    def _fail_the_commit(fd: int) -> None:
        real_fsync(fd)
        try:
            info, wanted = os.fstat(fd), target.stat()
        except OSError:
            return
        if (info.st_dev, info.st_ino) != (wanted.st_dev, wanted.st_ino):
            return
        if target.read_bytes()[-1:] != b"\n":
            return  # the record itself, not the byte that commits it
        raise OSError("cannot commit dispenses.jsonl")

    snapshots = _Snapshots()
    stream = _stream(tmp_path, [0], provenance=[snapshots])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", _fail_the_commit)
        with pytest.raises(OSError):
            await stream.get_task()
        assert stream.stopped
        with pytest.raises(RuntimeError, match="could not record a dispense"):
            await stream.aclose()

    assert read_dispenses(prov) == [], "a rolled-back dispense left its provenance behind"
    assert reconcile(prov) == []
    # The span was opened — the failure is downstream of it — and it is dropped unfinalized,
    # exactly as every other refused dispense drops one.
    assert len(snapshots.begins) == 1 and snapshots.completed == []
