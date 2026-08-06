"""Serving more than one env from one stream.

Native tool names collide across envs — every ``ToolUsingEnv`` carries ``terminate``, and real
envs share ``done`` and ``submit_answer`` with different schemas behind them — while an endpoint
registers one schema per name. These tests pin that the collision is resolved by explicit
routing, not by hoping the names differ.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastmcp import Client
from mcp.shared.tool_name_validation import TOOL_NAME_REGEX, validate_tool_name

from hgym.serve.stream import (
    _TOOL_NAME_CHAR,
    _TOOL_NAME_MAX,
    TaskRef,
    TaskStream,
    build_stream_server,
    read_results,
)
from hgym.task import ToolManifest
from tests._fixtures.choice_env import _FixtureChoiceEnv
from tests._fixtures.score_env import _FixtureScoreEnv

ANSWER_TASKS = [{"id": "q0", "question": "2+2?", "answer": "4"}]
CHOICE_TASKS = [{"id": "c0", "choice": 7}]

ANSWERS = "answers"
CHOICES = "choices"


def _env_for(name: str) -> Any:
    if name == ANSWERS:
        return _FixtureScoreEnv(tasks=ANSWER_TASKS)
    if name == CHOICES:
        return _FixtureChoiceEnv(tasks=CHOICE_TASKS)
    raise AssertionError(f"unexpected env key {name!r}")


def _stream(tmp_path: Path, refs: List[TaskRef], **kwargs: Any) -> TaskStream:
    return TaskStream(_env_for, refs, prov_dir=tmp_path / "prov", **kwargs)


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[0].text)


def _renaming(base: Any, names: Dict[str, str]) -> Any:
    """``base``, publishing the same tools under different names."""

    class _Renamed(base):  # type: ignore[misc, valid-type]
        def describe(self, task_id: Any = None) -> Any:
            spec = super().describe(task_id)
            spec.tools = [
                tool.model_copy(update={"name": names.get(tool.name, tool.name)})
                for tool in spec.tools
            ]
            return spec

    return _Renamed


BOTH = [TaskRef(ANSWERS, 0), TaskRef(CHOICES, 0)]


async def test_colliding_native_names_are_registered_once_per_env(tmp_path: Path) -> None:
    # Both envs expose `submit`, `noop` and `terminate`, and `submit` has a different schema in
    # each. One endpoint cannot publish two schemas under one name, so both are prefixed.
    async with _stream(tmp_path, BOTH) as stream:
        advertised = {tool.name: tool for tool in stream.tools}
        assert "answers__submit" in advertised and "choices__submit" in advertised
        assert "answers__terminate" in advertised and "choices__terminate" in advertised
        assert "submit" not in advertised
        assert "answer" in advertised["answers__submit"].input_schema["properties"]
        assert "choice" in advertised["choices__submit"].input_schema["properties"]


async def test_each_task_is_told_only_its_own_tools(tmp_path: Path) -> None:
    async with _stream(tmp_path, BOTH) as stream:
        first = await stream.get_task()
        assert first is not None
        assert first.env == ANSWERS
        assert {tool["name"] for tool in first.tools} == {
            "answers__submit",
            "answers__noop",
            "answers__terminate",
        }


async def test_calls_route_to_the_env_the_task_belongs_to(tmp_path: Path) -> None:
    async with _stream(tmp_path, BOTH) as stream:
        first = await stream.get_task()
        assert first is not None and first.env == ANSWERS
        done = _payload(await stream.dispatch("answers__submit", {"answer": "4"}))
        assert done["terminated"] is True

        second = await stream.get_task()
        assert second is not None and second.env == CHOICES
        done = _payload(await stream.dispatch("choices__submit", {"choice": 7}))
        assert done["terminated"] is True

    assert [(row.env, row.task_idx) for row in stream.results] == [(ANSWERS, 0), (CHOICES, 0)]
    assert all(row.score is not None and row.score.success for row in stream.results)


async def test_a_lease_cannot_reach_another_envs_tool(tmp_path: Path) -> None:
    # The multi-env failure the binding exists for: with every tool exposed at once, a valid
    # lease naming task A must not be able to seal and score A through a tool advertised for B.
    async with _stream(tmp_path, BOTH, max_in_flight=2) as stream:
        answers = await stream.get_task()
        choices = await stream.get_task()
        assert answers is not None and choices is not None

        refused = _payload(
            await stream.dispatch("choices__submit", {"choice": 7, "lease": answers.lease})
        )
        assert refused["error"] == "wrong_env" and refused["stream_error"] is True
        # Neither task was touched: both are still live and both still seal normally.
        assert stream.queue_info().in_flight == 2

        also_refused = _payload(
            await stream.dispatch("answers__terminate", {"lease": choices.lease})
        )
        assert also_refused["error"] == "wrong_env"

        assert _payload(
            await stream.dispatch("answers__submit", {"answer": "4", "lease": answers.lease})
        )["terminated"] is True
        assert _payload(
            await stream.dispatch("choices__submit", {"choice": 7, "lease": choices.lease})
        )["terminated"] is True

    by_env = {row.env: row for row in stream.results}
    assert by_env[ANSWERS].closure == "sealed" and by_env[CHOICES].closure == "sealed"
    assert all(row.score is not None and row.score.success for row in stream.results)


async def test_results_are_grouped_by_env(tmp_path: Path) -> None:
    # 0.7 on one env and 0.7 on another are not the same quantity, so per-env is the default
    # presentation — without forbidding a caller from aggregating anyway.
    async with _stream(tmp_path, BOTH) as stream:
        await stream.get_task()
        await stream.dispatch("answers__submit", {"answer": "4"})
        await stream.get_task()
        await stream.dispatch("choices__submit", {"choice": 0})  # wrong

    grouped = stream.results_by_env
    assert set(grouped) == {ANSWERS, CHOICES}
    assert [row.task_idx for row in grouped[ANSWERS]] == [0]
    assert grouped[ANSWERS][0].score is not None and grouped[ANSWERS][0].score.success is True
    assert grouped[CHOICES][0].score is not None and grouped[CHOICES][0].score.success is False
    assert len(stream.results) == 2  # the ungrouped view is still there


async def test_reading_the_recorded_rows_cannot_rewrite_them(tmp_path: Path) -> None:
    # Two public views onto one list, and both used to hand back the run's own rows:
    # `results[0] is results_by_env[env][0]`. A `ResultRow` is frozen and *shallow*, so what a
    # reader got was a handle on `extensions`, on `observed`, and on the one list that
    # `score.feedback` also is — and an edit through any of them changed what the run reported
    # while the file it had already committed said something else. That is the shape of the worst
    # version: an in-memory row headlining `success=True` beside an `observed` item saying the
    # answer was wrong, with the record on disk agreeing with neither.
    #
    # So the run keeps one canonical row — the wire form the file holds — and every read is a
    # copy of it, exactly as the frozen tool contract is read (`TaskStream.tools`).
    async with _stream(tmp_path, BOTH) as stream:
        await stream.get_task()
        await stream.dispatch("answers__submit", {"answer": "4"})
        await stream.get_task()
        await stream.dispatch("choices__submit", {"choice": 0})  # wrong

    durable = [row.to_wire() for row in read_results(tmp_path / "prov")]
    assert [row.to_wire() for row in stream.results] == durable, (
        "what the run reports in memory and what it committed are one record"
    )
    first = stream.results[0]
    assert first is not stream.results[0], "two reads of the record share one row"
    assert first is not stream.results_by_env[first.env][0], "two accessors share one row"
    assert first.score is not None
    assert first.observed is not first.score.feedback, "one list is both halves of the row"

    # A reader edits everything the frozen row leaves reachable, through both accessors.
    grouped = stream.results_by_env
    for row in (*stream.results, *(r for rows in grouped.values() for r in rows)):
        row.observed[0]["value"] = "invented"
        row.extensions["invented"] = True
        if row.score is not None:
            row.score.feedback[0]["value"] = "invented"

    assert [row.to_wire() for row in stream.results] == durable, "the record was rewritten"
    assert [row.to_wire() for row in read_results(tmp_path / "prov")] == durable
    assert {
        row.lease: row.to_wire()
        for env_rows in stream.results_by_env.values()
        for row in env_rows
    } == {row["lease"]: row for row in durable}, "the grouped view disagrees with the record"


async def test_separators_are_rejected_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        TaskStream(
            _env_for,
            [TaskRef("two__words", 0), TaskRef(CHOICES, 0)],
            prov_dir=tmp_path / "prov",
        )

    class _Separated(_FixtureScoreEnv):
        def describe(self, task_id=None):
            spec = super().describe(task_id)
            spec.tools = [
                tool.model_copy(update={"name": "sub__mit"}) if tool.name == "submit" else tool
                for tool in spec.tools
            ]
            return spec

    with pytest.raises(ValueError, match="ambiguous"):
        TaskStream(
            lambda name: _Separated(tasks=ANSWER_TASKS) if name == ANSWERS else _env_for(name),
            [TaskRef(ANSWERS, 0), TaskRef(CHOICES, 0)],
            prov_dir=tmp_path / "prov",
        )


async def test_one_env_is_joined_to_nothing_so_neither_half_is_restricted(
    tmp_path: Path,
) -> None:
    # The separator rules exist to keep a *join* unambiguous, and a single-env stream performs
    # no join: it advertises the env's own names and its env key never reaches the wire at all.
    # So both halves are free there — a key holding `__` was legal before this stream could
    # serve several envs and stays legal, and `sub__mit` is an ordinary tool name that one
    # endpoint can serve perfectly well.
    separated = _renaming(_FixtureScoreEnv, {"submit": "sub__mit"})
    stream = TaskStream(
        lambda _name: separated(tasks=ANSWER_TASKS),
        [TaskRef("single__env", 0)],
        prov_dir=tmp_path / "prov",
    )
    async with stream:
        assert [tool.name for tool in stream.tools] == ["terminate", "sub__mit", "noop"]
        task = await stream.get_task()
        assert task is not None
        assert task.env == "single__env"
        assert task.tool_naming is None
        assert set(task.to_wire()) == {"env", "instructions", "budget", "tools"}
        assert _payload(await stream.dispatch("sub__mit", {"answer": "4"}))["terminated"] is True
    assert stream.results[0].score is not None and stream.results[0].score.success is True


async def test_a_name_the_stream_makes_is_one_a_tool_may_be_called(tmp_path: Path) -> None:
    # A joined name is the stream's own, and it goes on the wire. The protocol bounds what a
    # tool name may be, and nothing downstream of here enforces that bound: FastMCP warns about
    # a name outside it and registers the tool anyway, so an endpoint built from one looks
    # healthy in-process and is refused where the harness cannot see it. Refused at
    # construction instead, checked on the joined string.
    def _queue(key: str, factory: Any = None) -> Any:
        """``key`` beside a second env, served by a factory that answers to any key."""
        made = factory or (lambda _name: _FixtureScoreEnv(tasks=ANSWER_TASKS))
        return made, [TaskRef(key, 0), TaskRef(CHOICES, 0)]

    # The env key is a caller's private label until it becomes part of a tool name — with one
    # exception, which is refused a step earlier and whichever way the queue is served: an empty
    # key is not a label at all, and it is what every row of that env is filed under (see
    # `_require_task_ref`), so it never reaches the join.
    for key, expected in [
        ("answer key", r"env key 'answer key' .*contains ' '"),
        ("", r"env must be a non-empty string"),
        # Non-ASCII too, which also keeps two keys that normalise to one string off the wire:
        # they are distinct dict keys here and a client is free to fold them together.
        ("réponse", r"env key 'r\wponse' .*contains '\w'"),
    ]:
        made, queue = _queue(key)
        with pytest.raises(ValueError, match=expected):
            TaskStream(made, queue, prov_dir=tmp_path / f"key{len(key)}")

    # Length belongs to neither half. This key and every tool name in the fixture are well
    # inside the limit; only the join is over it, so only the join can find it.
    long_key = "a" * 120
    assert len(long_key) <= _TOOL_NAME_MAX and len("terminate") <= _TOOL_NAME_MAX
    made, queue = _queue(long_key)
    with pytest.raises(ValueError, match=r"131 characters long, over the 128"):
        TaskStream(made, queue, prov_dir=tmp_path / "long")

    # The same bound, reached from the other half: a legal key and an over-long tool name.
    made, queue = _queue(
        ANSWERS,
        lambda name: (
            _renaming(_FixtureScoreEnv, {"submit": "s" * 120})(tasks=ANSWER_TASKS)
            if name == ANSWERS
            else _env_for(name)
        ),
    )
    with pytest.raises(ValueError, match=r"over the 128 a tool name may be"):
        TaskStream(made, queue, prov_dir=tmp_path / "long-tool")

    # And every name a stream does advertise passes the protocol's own validator, not just
    # this module's reading of it.
    async with _stream(tmp_path, BOTH) as stream:
        advertised = [tool.name for tool in stream.tools]
        assert advertised, "nothing was advertised, so nothing was checked"
        assert [name for name in advertised if not validate_tool_name(name).is_valid] == []


def test_the_tool_name_rule_is_the_one_the_protocol_states() -> None:
    # The rule is restated in `stream.py` rather than imported from the MCP package's internals,
    # so that what this module promises cannot change under it on a dependency bump. This is
    # what makes that restatement checkable rather than a copy that quietly drifts.
    assert TOOL_NAME_REGEX.pattern == f"^{_TOOL_NAME_CHAR.pattern}{{1,{_TOOL_NAME_MAX}}}$"


async def test_two_tools_cannot_share_one_public_name(tmp_path: Path) -> None:
    # `__` is refused in either half, but that is not what makes a public name unambiguous:
    # ("a", "_x") and ("a_", "x") hold no `__` between them and both join to `a___x`. The route
    # map is the check that catches it — and the same check catches one env publishing a name
    # twice, which one endpoint cannot serve either.
    def factory(name: str) -> Any:
        if name == "a":
            return _renaming(_FixtureScoreEnv, {"submit": "_x"})(tasks=ANSWER_TASKS)
        if name == "a_":
            return _renaming(_FixtureChoiceEnv, {"submit": "x"})(tasks=CHOICE_TASKS)
        raise AssertionError(name)

    with pytest.raises(ValueError, match=r"would both be advertised as 'a___x'"):
        TaskStream(factory, [TaskRef("a", 0), TaskRef("a_", 0)], prov_dir=tmp_path / "joined")

    class _Doubled(_FixtureScoreEnv):
        def describe(self, task_id: Any = None) -> Any:
            spec = super().describe(task_id)
            spec.tools = [*spec.tools, spec.tools[-1].model_copy(update={"description": "again"})]
            return spec

    with pytest.raises(ValueError, match="publishes two tools named"):
        TaskStream(
            lambda _name: _Doubled(tasks=ANSWER_TASKS),
            [TaskRef(ANSWERS, 0)],
            prov_dir=tmp_path / "doubled",
        )


async def test_the_framing_never_names_a_tool_the_endpoint_does_not_serve(
    tmp_path: Path,
) -> None:
    # The env's prose names the env's own tools ("call `submit`"), and prefixing makes that name
    # uncallable — a literal instruction-follower is handed two incompatible commands and the
    # endpoint refuses one of them before the stream is even reached. The prose belongs to the
    # env author and ships verbatim, so the framing says what the tools are called *beside* it,
    # naming only what this endpoint really registers.
    async with _stream(tmp_path, BOTH, max_in_flight=2) as stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            registered = {tool.name for tool in await client.list_tools()}
            task = _payload(await client.call_tool("get_task", {}))
            assert task["env"] == ANSWERS

            authored = _FixtureScoreEnv(tasks=ANSWER_TASKS).describe().instructions
            assert task["instructions"] == authored, "the env's own prose was edited"
            assert "`submit`" in task["instructions"] and "submit" not in registered

            mapped = dict(re.findall(r"`([^`]+)` is called as `([^`]+)`", task["tool_naming"]))
            assert mapped == {
                "submit": "answers__submit",
                "noop": "answers__noop",
                "terminate": "answers__terminate",
            }
            # Nothing the framing quotes is a name the agent cannot use: every one is either a
            # name the env's own prose uses or a name this endpoint serves.
            quoted = set(re.findall(r"`([^`]+)`", task["tool_naming"]))
            assert quoted - set(mapped) - registered == {"tools"}

            # The instruction-follower's call, made under the name the framing gives it.
            done = _payload(
                await client.call_tool(mapped["submit"], {"answer": "4", "lease": task["lease"]})
            )
            assert done["terminated"] is True
    assert stream.results[0].score is not None and stream.results[0].score.success is True


async def test_a_single_env_stream_is_not_prefixed(tmp_path: Path) -> None:
    async with _stream(tmp_path, [TaskRef(ANSWERS, 0)]) as stream:
        assert {tool.name for tool in stream.tools} == {"submit", "noop", "terminate"}
        task = await stream.get_task()
        assert task is not None
        # Nothing was renamed, so the framing says nothing about naming and the wire an agent
        # sees is the one it saw before this stream could serve several envs at all.
        assert task.tool_naming is None
        assert set(task.to_wire()) == {"env", "instructions", "budget", "tools"}
        assert _payload(await stream.dispatch("submit", {"answer": "4"}))["terminated"] is True


async def test_multi_env_over_mcp(tmp_path: Path) -> None:
    async with _stream(tmp_path, BOTH, max_in_flight=2) as stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            names = {tool.name for tool in await client.list_tools()}
            assert {"answers__submit", "choices__submit", "get_task", "queue_info"} <= names

            first = _payload(await client.call_tool("get_task", {}))
            second = _payload(await client.call_tool("get_task", {}))
            assert {first["env"], second["env"]} == {ANSWERS, CHOICES}

            for task in (first, second):
                tool = f"{task['env']}__submit"
                args = {"answer": "4"} if task["env"] == ANSWERS else {"choice": 7}
                out = _payload(await client.call_tool(tool, {**args, "lease": task["lease"]}))
                assert out["terminated"] is True
    assert set(stream.results_by_env) == {ANSWERS, CHOICES}
    assert all(row.score is not None and row.score.success for row in stream.results)


async def test_a_drifting_episode_is_checked_on_its_native_names(tmp_path: Path) -> None:
    # With two envs the endpoint publishes `<env>__<tool>`, but the env itself publishes plain
    # `submit`. The frozen contract is therefore compared on the NATIVE manifest, and the
    # prefixing is derived from it afterwards — so an instance that adds `hint` is refused by
    # that name rather than by `answers__hint`, and the other env's tools are not dragged into
    # the comparison.
    class _AddsATool(_FixtureScoreEnv):
        def __init__(self, drift: bool, **kwargs: Any) -> None:
            self._drift = drift
            super().__init__(**kwargs)

        def describe(self, task_id: Any = None) -> Any:
            spec = super().describe(task_id)
            if self._drift:
                spec.tools = [
                    *spec.tools,
                    ToolManifest(
                        name="hint",
                        description="Ask for a hint.",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ]
            return spec

    answer_envs: List[Any] = []

    def factory(name: str) -> Any:
        if name != ANSWERS:
            return _env_for(name)
        # The catalog instance is clean; every episode instance after it drifts.
        env = _AddsATool(drift=bool(answer_envs), tasks=ANSWER_TASKS)
        answer_envs.append(env)
        return env

    stream = TaskStream(factory, BOTH, prov_dir=tmp_path / "prov")
    assert "answers__submit" in {tool.name for tool in stream.tools}  # really prefixed
    with pytest.raises(RuntimeError, match=r"env 'answers' published .*\(added \['hint'\]\)"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()


def _catalog_teardown(behaviour: Any) -> Any:
    """A factory whose *catalog* env for :data:`ANSWERS` tears down through ``behaviour``.

    Only the catalog instances are affected: they are the ones the stream's own release lets go,
    and an episode's env is closed by its seal instead."""
    catalog: List[str] = []
    closed: List[str] = []

    def factory(name: str) -> Any:
        first = name not in catalog
        catalog.append(name)
        base = type(_env_for(name))

        class _Teardown(base):  # type: ignore[misc, valid-type]
            async def close(self) -> None:
                if first and name == ANSWERS:
                    await behaviour()
                await super().close()
                closed.append(name)

        return _Teardown(tasks=ANSWER_TASKS if name == ANSWERS else CHOICE_TASKS)

    return factory, closed


async def test_a_catalog_close_that_hangs_does_not_hold_another_env_open(
    tmp_path: Path,
) -> None:
    # `Env.close()` is third-party code that can block for as long as it likes. Closed one at a
    # time, the first env that does not return decides whether any env after it is closed at
    # all: they keep their sessions and subprocesses for exactly as long as it hangs, and a
    # queue of several envs is what makes that reachable.
    gate = asyncio.Event()
    factory, closed = _catalog_teardown(gate.wait)

    stream = TaskStream(factory, BOTH, prov_dir=tmp_path / "prov")
    closing = asyncio.ensure_future(stream.aclose())
    try:
        for _ in range(200):
            await asyncio.sleep(0.01)
            if closed:
                break
        assert closed == [CHOICES], "a hung close held every later catalog env open"
        assert not closing.done(), "the hung env was not still hanging, so nothing was tested"
    finally:
        # Released before any assertion that could fail above, so a regression fails rather
        # than leaving the shutdown blocked on a gate nobody sets.
        gate.set()
    await asyncio.wait_for(closing, timeout=5)
    assert sorted(closed) == [ANSWERS, CHOICES]


async def test_a_catalog_close_that_fails_still_closes_the_others(tmp_path: Path) -> None:
    # Teardown is best-effort and stays best-effort: one env's failure is not another's, and it
    # is not the run's outcome either.
    async def raiser() -> None:
        raise RuntimeError("this env cannot be closed")

    factory, closed = _catalog_teardown(raiser)

    stream = TaskStream(factory, BOTH, prov_dir=tmp_path / "prov")
    await stream.aclose()
    assert closed == [CHOICES]
    assert not stream.stopped, "a teardown failure is not the run's outcome"
