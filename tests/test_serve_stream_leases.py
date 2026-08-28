"""Leases: how a native call names the episode it belongs to when several are live.

The property under test is that a lease is *identity*, not a hint. A call that names nothing,
names a task that never existed, or names a finished one is refused by the stream — which costs
the agent no budget and leaves no trace in any trajectory — and two live episodes interleave
without touching each other's state.

At capacity 1 the property is the opposite one: there is no lease anywhere, so nothing about
the published schemas or the arguments an env receives may differ from a stream that had never
heard of leases — including for an env that spends the word ``lease`` on an argument of its own.
"""

from __future__ import annotations

import asyncio
import gc
import json
import secrets as _real_secrets
import weakref
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastmcp import Client

from shogym.mcp import MCPServerSpec
from shogym.serve import stream as stream_module
from shogym.serve.stream import (
    TaskRef,
    TaskStream,
    build_stream_server,
    read_dispenses,
    reconcile,
)
from shogym.types import FeedbackCollection
from tests._fixtures.score_env import ENV_NAME, SUBMIT_TOOL, _FixtureScoreEnv

TASKS = [
    {"id": "q0", "question": "2+2?", "answer": "4"},
    {"id": "q1", "question": "3+3?", "answer": "6"},
    {"id": "q2", "question": "5+5?", "answer": "10"},
]

LEASE_ARG_TOOL = "lookup_lease"
_LEASE_ARG_SPEC = MCPServerSpec(
    name="fixture_lease_arg",
    transport="in_process",
    module="tests._fixtures.lease_arg_mcp",
)


class _EnvWithNativeLeaseArg(_FixtureScoreEnv):
    """An env one of whose own tools takes an argument named ``lease``.

    Legal, and unremarkable to the env: ``lease`` is an ordinary word. It only matters here
    because the stream spends the same word on routing when several episodes are live."""

    mcp_servers = (*_FixtureScoreEnv.mcp_servers, _LEASE_ARG_SPEC)


def _stream(tmp_path: Path, indices: List[int], **kwargs: Any) -> TaskStream:
    return TaskStream(
        lambda _name: _FixtureScoreEnv(tasks=TASKS),
        [TaskRef(ENV_NAME, i) for i in indices],
        prov_dir=tmp_path / "prov",
        **kwargs,
    )


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[0].text)


def _rows(tmp_path: Path) -> List[Dict[str, Any]]:
    path = tmp_path / "prov" / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ----- capacity 1 is unchanged -----


async def test_capacity_one_publishes_no_lease(tmp_path: Path) -> None:
    # The whole point of gating the wrapper on capacity: an existing single-slot client sees the
    # env's own schemas and no extra argument.
    async with _stream(tmp_path, [0]) as stream:
        native = _FixtureScoreEnv(tasks=TASKS).describe().tools
        assert [tool.input_schema for tool in stream.tools] == [
            tool.input_schema for tool in native
        ]
        task = await stream.get_task()
        assert task is not None and task.lease is None
        assert "lease" not in task.to_wire()


async def test_a_native_lease_argument_reaches_the_env_at_capacity_one(tmp_path: Path) -> None:
    # `lease` is only the stream's word above capacity 1. At capacity 1 the env's own schemas
    # are advertised verbatim, so an env tool may legitimately take an argument of that name —
    # and the stream must route on nothing and pass it through untouched, exactly as it did
    # before leases existed.
    seen: List[Dict[str, Any]] = []
    stream = TaskStream(
        lambda _name: _EnvWithNativeLeaseArg(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    async with stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            schema = {tool.name: tool for tool in await client.list_tools()}[
                LEASE_ARG_TOOL
            ].inputSchema
            # Advertised exactly as the env wrote it: the env's `lease` is a required string of
            # its own, and nothing was wrapped around it.
            assert schema["properties"]["lease"] == {"type": "string"}
            assert schema["required"] == ["lease"]

            await client.call_tool("get_task", {})
            live = next(iter(stream._live.values()))  # noqa: SLF001 - inspecting the trajectory
            original_call = live.episode.call

            async def record(tool: str, arguments: Any = None):
                seen.append(dict(arguments or {}))
                return await original_call(tool, arguments)

            live.episode.call = record  # type: ignore[method-assign]

            out = _payload_from_tool(
                await client.call_tool(LEASE_ARG_TOOL, {"lease": "LX-1"})
            )
            # The call reached the env and succeeded; it was not refused as a routing mistake.
            assert "error" not in out
            assert out["terminated"] is False
            assert json.loads(out["content"])["looked_up"] == "LX-1"

            done = _payload_from_tool(await client.call_tool(SUBMIT_TOOL, {"answer": "4"}))
            assert done["terminated"] is True

    # The argument arrived verbatim, and the step was recorded as the env's own.
    assert seen[0] == {"lease": "LX-1"}
    assert stream.results[0].closure == "sealed"


async def test_capacity_one_ignores_a_lease_instead_of_binding_on_it(tmp_path: Path) -> None:
    # A refusal is the loud half of reading a lease that isn't one. The quiet half is an argument
    # that disappears: whatever the stream reads it also removes, so a call could reach the env
    # missing a key the agent actually sent — including a value the env's schema allows and the
    # routing code does not (`null`). At capacity 1 nothing is read and nothing is removed, and a
    # lease handed in by a caller is ignored rather than bound on.
    seen: List[Dict[str, Any]] = []
    stream = TaskStream(
        lambda _name: _EnvWithNativeLeaseArg(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    async with stream:
        await stream.get_task()
        live = next(iter(stream._live.values()))  # noqa: SLF001 - inspecting the trajectory
        original_call = live.episode.call

        async def record(tool: str, arguments: Any = None):
            seen.append(dict(arguments or {}))
            return await original_call(tool, arguments)

        live.episode.call = record  # type: ignore[method-assign]

        assert "error" not in _payload(await stream.dispatch(LEASE_ARG_TOOL, {"lease": None}))
        assert "error" not in _payload(
            await stream.dispatch(LEASE_ARG_TOOL, {"lease": "LX-2"}, lease="0" * 32)
        )
    # Both arrived exactly as sent — whether the env then accepts the value is the env's call to
    # make, not the stream's. (The trailing entries are the drain's own forced terminal.)
    assert seen[:2] == [{"lease": None}, {"lease": "LX-2"}]


# ----- capacity > 1 -----


class _AnswersOnce(str):
    """A lease that answers one lookup truthfully and would answer the next one differently.

    Ordinary Python: ``str`` is subclassable and ``__hash__`` is overridable, and nothing
    obliges a caller's object to give the same answer twice. It counts the lookups made on it,
    so a test can say how many the stream took."""

    hashes = 0

    def __hash__(self) -> int:
        type(self).hashes += 1
        return str.__hash__(self) if type(self).hashes == 1 else 0


async def test_a_lease_is_looked_up_once_and_the_entry_found_is_the_one_used(
    tmp_path: Path,
) -> None:
    # The routing key is the caller's object, and `_resolve` only requires it to be a `str`. A
    # membership test followed by a subscript would be two reads of it: the test passes, the
    # subscript misses, and `KeyError` leaves `dispatch` where nothing catches it. The call the
    # agent made is dropped, the stream does not stop, and the task stays live for the drain to
    # end — so the row says the agent played the task out and got it wrong, over the correct
    # answer it had in fact submitted. An unearned wrong answer, produced by a routing key.
    _AnswersOnce.hashes = 0
    async with _stream(tmp_path, [0, 1], max_in_flight=2) as stream:
        first = await stream.get_task()
        second = await stream.get_task()
        assert first is not None and second is not None
        assert first.lease and second.lease

        done = _payload(
            await stream.dispatch(
                SUBMIT_TOOL, {"answer": "6"}, lease=_AnswersOnce(second.lease)
            )
        )
        assert done["terminated"] is True
        assert stream.stopped is False

        # One read, so the answer it gave is the one that was used.
        assert _AnswersOnce.hashes == 1

        away = _payload(
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": first.lease})
        )
        assert away["terminated"] is True

    # The submitted answer is what the record holds — not the `drained` loss a dropped call
    # would have left behind.
    assert [(row.task_idx, row.closure) for row in stream.results] == [
        (1, "sealed"),
        (0, "sealed"),
    ]
    assert all(row.score is not None and row.score.success for row in stream.results)
    # And what was recorded is the stream's own minted lease, never the caller's object.
    assert all(type(row.lease) is str for row in stream.results)
    assert {row.lease for row in stream.results} == {first.lease, second.lease}


async def test_two_live_episodes_interleave(tmp_path: Path) -> None:
    # Two episodes of the same env class, live at once. Each owns its own env instance, so
    # sealing one must not disturb the other — the failure this design exists to prevent, since
    # Env.close() ends every session its instance tracks.
    async with _stream(tmp_path, [0, 1], max_in_flight=2) as stream:
        first = await stream.get_task()
        second = await stream.get_task()
        assert first is not None and second is not None
        assert first.lease and second.lease and first.lease != second.lease
        assert stream.queue_info().in_flight == 2

        # Interleave ordinary work on both.
        for lease in (first.lease, second.lease, first.lease):
            step = _payload(await stream.dispatch("noop", {"lease": lease}))
            assert step["terminated"] is False

        # Seal the first; the second must be untouched and still answerable.
        done = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": first.lease}))
        assert done["terminated"] is True
        assert stream.queue_info().in_flight == 1

        still_live = _payload(await stream.dispatch("noop", {"lease": second.lease}))
        assert still_live["terminated"] is False
        second_done = _payload(
            await stream.dispatch(SUBMIT_TOOL, {"answer": "6", "lease": second.lease})
        )
        assert second_done["terminated"] is True

    assert [(row.task_idx, row.closure) for row in stream.results] == [
        (0, "sealed"),
        (1, "sealed"),
    ]
    assert all(row.score is not None and row.score.success for row in stream.results)


def _loses_the_call_for(task_idx: int) -> Any:
    """An env that fails every call made while *one* of its tasks is still live. The other task
    of the same queue is ordinary, so a failure attributed to the wrong episode is visible as a
    wrong number on both rows at once."""

    class _LosesOneTasksCalls(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            if not terminated and task["task_idx"] == task_idx:
                raise RuntimeError(f"the session dropped a call for task {task_idx}")
            return super()._verify(trajectory, task, terminated=terminated, evidence=evidence)

    return lambda _name: _LosesOneTasksCalls(tasks=TASKS)


async def test_a_lost_call_is_recorded_against_the_episode_that_made_it(
    tmp_path: Path,
) -> None:
    # With one slot, "the live entry" and "the entry this call belongs to" were the same thing.
    # They are not any more, and a loss kept against the wrong one is a wrong number in both
    # directions at once: the episode that never failed is filed unscored for a call it never
    # made, while the one whose call the harness dropped seals as though nothing happened and its
    # forced closure is averaged in as an earned answer.
    #
    # Neither episode ends itself, so the stream ends both — which is the only case where a lost
    # call changes a row, and therefore the only case where attributing it wrongly is visible.
    # The one that loses a call is the SECOND dispensed, so the oldest live entry and the entry
    # the call was routed to are different objects throughout.
    stream = TaskStream(
        _loses_the_call_for(1),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
        max_in_flight=2,
    )
    async with stream:
        healthy = await stream.get_task()
        losing = await stream.get_task()
        assert healthy is not None and losing is not None

        # The healthy episode plays, and its call reaches a result.
        played = _payload(await stream.dispatch("noop", {"lease": healthy.lease}))
        assert played["terminated"] is False
        with pytest.raises(RuntimeError, match="dropped a call for task 1"):
            await stream.dispatch("noop", {"lease": losing.lease})
        # Neither is ended by its agent; the drain ends both.

    rows = {row.task_idx: row for row in stream.results}
    kept, lost = rows[0], rows[1]
    assert lost.closure == "finalize_error", "the lost call was filed as an answer"
    assert lost.score is None, "a call the harness dropped may not be scored"
    assert "the agent never played it out" in (lost.diagnostic or "")
    assert "dropped a call for task 1" in (lost.diagnostic or "")
    # The episode that made no failing call is scored on what the stream's own terminal produced
    # for it — an outcome it earned by playing — and says nothing about the other one's failure.
    assert kept.closure == "drained", "another episode's lost call rewrote this one's ending"
    assert kept.score is not None, "another episode's lost call unscored this one"
    assert kept.diagnostic is None
    assert not stream.stopped, "one lost call ended a queue the agent could still play"
    assert {row["task_idx"]: row["closure"] for row in _rows(tmp_path)} == {
        0: "drained",
        1: "finalize_error",
    }


async def test_a_terminal_that_fails_does_not_unscore_another_live_episode(
    tmp_path: Path,
) -> None:
    # The same question about the other half of the entry: a terminal the *stream* drove and the
    # env then failed stops the run, and the row it lands is unscored. Both belong to the episode
    # whose terminal it was. Attributed to whichever entry the seal happened to reach first, the
    # stop would be right by accident and the unscored row would be wrong twice over.
    class _FailsTheForcedTerminal(_FixtureScoreEnv):
        score_terminal_tool = None  # `verify` runs inline on the terminating call

        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            if terminated and task["task_idx"] == 1:
                raise RuntimeError("the evaluator exploded on its way out")
            return super()._verify(trajectory, task, terminated=terminated, evidence=evidence)

    stream = TaskStream(
        lambda _name: _FailsTheForcedTerminal(tasks=TASKS),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
        max_in_flight=2,
    )
    healthy = await stream.get_task()
    failing = await stream.get_task()
    assert healthy is not None and failing is not None
    # Neither episode ends itself: the drain ends both, and only one of the two terminals fails.
    with pytest.raises(RuntimeError, match="failed while the stream ended a task"):
        await stream.aclose()

    rows = {row.task_idx: row for row in stream.results}
    assert rows[1].closure == "finalize_error" and rows[1].score is None
    assert "while the stream ended the task" in (rows[1].diagnostic or "")
    assert "exploded on its way out" in (rows[1].diagnostic or "")
    # The other episode's terminal worked, so its row is the outcome that terminal produced —
    # the stop belongs to the run, not to this row.
    assert rows[0].closure == "drained", "the other episode's ending was rewritten"
    assert rows[0].score is not None, "one episode's failed terminal unscored another's row"
    assert rows[0].diagnostic is None


async def test_calls_on_two_leases_run_concurrently(tmp_path: Path) -> None:
    # The stream must not serialise two episodes behind one lock: the second call has to make
    # progress while the first is still awaiting its env.
    entered = asyncio.Event()
    release = asyncio.Event()

    class _Blocking(_FixtureScoreEnv):
        async def finalize(self, req: Any) -> Any:
            entered.set()
            await release.wait()
            return await super().finalize(req)

    stream = TaskStream(
        lambda _name: _Blocking(tasks=TASKS),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
        max_in_flight=2,
        provenance_timeout=None,
    )
    async with stream:
        first = await stream.get_task()
        second = await stream.get_task()
        assert first is not None and second is not None
        blocked = asyncio.ensure_future(
            stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": first.lease})
        )
        await entered.wait()
        # The other episode answers while the first is stuck inside its evaluator.
        other = await asyncio.wait_for(
            stream.dispatch("noop", {"lease": second.lease}), timeout=1.0
        )
        assert _payload(other)["terminated"] is False
        release.set()
        await blocked


async def test_the_lease_never_reaches_the_env(tmp_path: Path) -> None:
    # The routing capability must not become part of the agent's recorded action.
    seen: List[Dict[str, Any]] = []

    async with _stream(tmp_path, [0], max_in_flight=2) as stream:
        task = await stream.get_task()
        assert task is not None
        live = next(iter(stream._live.values()))  # noqa: SLF001 - inspecting the trajectory
        original_call = live.episode.call

        async def record(tool: str, arguments: Any = None):
            seen.append(dict(arguments or {}))
            return await original_call(tool, arguments)

        live.episode.call = record  # type: ignore[method-assign]
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": task.lease})
    assert seen and all("lease" not in args for args in seen)
    assert seen[-1] == {"answer": "4"}


async def test_a_lease_is_never_reused(tmp_path: Path) -> None:
    # A recycled lease would let a delayed call from a finished task act on its successor.
    async with _stream(tmp_path, [0, 1, 2], max_in_flight=1) as stream:
        for _ in range(3):
            await stream.get_task()
    leases = [row.lease for row in stream.results]
    assert len(set(leases)) == 3


class _ScriptedTokens:
    """A ``secrets`` stand-in that hands out chosen values first, then the real thing.

    A 128-bit CSPRNG will not collide on its own, and "never reused" is not a claim about how
    unlikely a collision is — it is a claim the stream *enforces*. So the source is scripted to
    produce one, which is also the only way to test the enforcement rather than the odds."""

    def __init__(self, *scripted: str) -> None:
        self.scripted = list(scripted)
        self.drawn: List[str] = []

    def token_hex(self, nbytes: int) -> str:
        value = self.scripted.pop(0) if self.scripted else _real_secrets.token_hex(nbytes)
        self.drawn.append(value)
        return value


def _dispense_record(
    lease: str, *, seq: int, position: int, task_idx: int, max_in_flight: int = 1
) -> Dict[str, Any]:
    """A dispense record in the shape this module writes one.

    The identity is carried because a record that names none is a record from before the member
    existed, and continuing one is a migration a caller has to ask for. This record is standing in
    for one *this* module wrote, so it says what such a record says: the run facts, blank where
    nobody named anything."""
    return {
        "lease": lease,
        "seq": seq,
        "position": position,
        "env": ENV_NAME,
        "task_idx": task_idx,
        "dispensed_at": 1.0,
        "extensions": {},
        "run_identity": {
            "caller": "",
            "envs": {},
            "deadline": None,
            "max_in_flight": max_in_flight,
        },
    }


async def test_a_resumed_run_will_not_mint_a_lease_its_record_already_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `_issued` is what makes a lease unrepeatable, and a resumed run starts with an empty one
    # while the leases it must not repeat are in the directory it is resuming. A repeat is
    # invisible in exactly the place the record cannot afford it: `reconcile` pairs a dispense
    # with a result BY LEASE, so the older run's result answers the newer run's dispense, and the
    # crash the newer one is owed — a position dispensed and never sealed — is reported as
    # nothing at all. A row goes missing from the record while the record looks complete.
    prov = tmp_path / "prov"
    repeat = "a" * 32
    tokens = _ScriptedTokens(repeat, repeat, "b" * 32)
    monkeypatch.setattr(stream_module, "secrets", tokens)

    async with _stream(tmp_path, [0, 1], max_in_flight=2) as first:
        task = await first.get_task()
        assert task is not None and task.lease == repeat
        await first.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": task.lease})
    assert [row.position for row in first.results] == [0]

    resumed = _stream(tmp_path, [0, 1], max_in_flight=2, resume=True)
    replay = await resumed.get_task()  # dispensed, never sealed — this run "crashes" here
    assert replay is not None
    assert replay.lease != repeat, "a resumed run minted a lease its own record already holds"
    assert [record["lease"] for record in read_dispenses(prov)] == [repeat, replay.lease]
    # The dispense the crash left behind is visible as one, which is only true while the two
    # dispenses are distinguishable.
    assert [(row.position, row.closure) for row in reconcile(prov)] == [(1, "broker_abort")]
    await resumed.aclose()


async def test_a_record_that_holds_one_lease_twice_is_refused_at_resume(
    tmp_path: Path,
) -> None:
    # The same ambiguity already committed to disk. Two dispenses under one lease cannot be
    # paired with their results — whichever result exists answers both — so a run continuing
    # that directory would append to a record that has already lost a row. Refused at
    # construction, before anything is spent, like every other unreadable provenance.
    prov = tmp_path / "prov"
    prov.mkdir(parents=True)
    (prov / "dispenses.jsonl").write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                _dispense_record("c" * 32, seq=1, position=0, task_idx=0, max_in_flight=2),
                _dispense_record("c" * 32, seq=2, position=1, task_idx=1, max_in_flight=2),
            )
        )
    )
    with pytest.raises(ValueError, match="records two dispenses under one lease"):
        _stream(tmp_path, [0, 1], max_in_flight=2, resume=True)


# ----- refusals are stream errors, never env steps -----


@pytest.mark.parametrize(
    "supplied, expected",
    [
        (None, "missing_lease"),
        ("0" * 32, "unknown_lease"),
        ("", "unknown_lease"),
    ],
)
async def test_a_call_that_names_no_live_task_is_refused(
    tmp_path: Path, supplied: Any, expected: str
) -> None:
    async with _stream(tmp_path, [0], max_in_flight=2) as stream:
        task = await stream.get_task()
        assert task is not None
        args = {"lease": supplied} if supplied is not None else {}
        payload = _payload(await stream.dispatch("noop", args))
        assert payload["error"] == expected
        assert payload["stream_error"] is True
        # Refused before the episode: no step was recorded, so the task still submits normally.
        done = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": task.lease}))
        assert done["terminated"] is True
    assert stream.results[0].closure == "sealed"


async def test_a_stale_lease_is_refused_rather_than_rebound(tmp_path: Path) -> None:
    # The task the lease named is over. It must be told so — not silently routed to whichever
    # task is live now, and not reported as if the lease had never existed.
    async with _stream(tmp_path, [0, 1], max_in_flight=2) as stream:
        first = await stream.get_task()
        assert first is not None
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": first.lease})
        second = await stream.get_task()
        assert second is not None

        payload = _payload(await stream.dispatch("noop", {"lease": first.lease}))
        assert payload["error"] == "sealed_lease"
        # The live task is untouched by the stale call.
        assert stream.queue_info().in_flight == 1
        done = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "6", "lease": second.lease}))
        assert done["terminated"] is True
    assert [row.task_idx for row in stream.results] == [0, 1]


async def test_a_tool_the_task_never_advertised_is_refused(tmp_path: Path) -> None:
    # Reaching the episode with an unknown tool would record an unknown-tool step, spending a
    # step of the budget on a routing mistake.
    async with _stream(tmp_path, [0], max_in_flight=2) as stream:
        task = await stream.get_task()
        assert task is not None
        payload = _payload(await stream.dispatch("api_fetch", {"lease": task.lease}))
        assert payload["error"] == "tool_not_in_task"
        # The budget is intact: the fixture's horizon is 3 and all of it is still available.
        for _ in range(2):
            step = _payload(await stream.dispatch("noop", {"lease": task.lease}))
            assert step["terminated"] is False
        done = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": task.lease}))
        assert done["terminated"] is True
    assert stream.results[0].score is not None and stream.results[0].score.success is True


# ----- over the wire -----


async def test_wrapper_schemas_carry_the_lease(tmp_path: Path) -> None:
    async with _stream(tmp_path, [0, 1], max_in_flight=2) as stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            schema = tools[SUBMIT_TOOL].inputSchema
            assert "lease" in schema["properties"]
            assert "lease" in schema["required"]
            assert "answer" in schema["properties"]  # the env's own argument survives
            assert schema.get("additionalProperties") is False  # still a closed schema

            first = _payload_from_tool(await client.call_tool("get_task", {}))
            second = _payload_from_tool(await client.call_tool("get_task", {}))
            assert first["lease"] != second["lease"]
            assert "lease" in {
                key for tool in first["tools"] for key in tool["input_schema"]["properties"]
            }

            out = _payload_from_tool(
                await client.call_tool(
                    SUBMIT_TOOL, {"answer": "4", "lease": first["lease"]}
                )
            )
            assert out["terminated"] is True
            refused = _payload_from_tool(
                await client.call_tool("noop", {"lease": first["lease"]})
            )
            assert refused["error"] == "sealed_lease"
    assert [row.task_idx for row in stream.results] == [0, 1]


def _payload_from_tool(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[0].text)


async def test_a_tool_named_lease_cannot_be_wrapped(tmp_path: Path) -> None:
    from shogym.task import TaskSpec, ToolManifest

    class _Colliding(_FixtureScoreEnv):
        def describe(self, task_id=None) -> TaskSpec:
            spec = super().describe(task_id)
            spec.tools = [
                *spec.tools,
                ToolManifest(
                    name="lease_check",
                    description="d",
                    input_schema={
                        "type": "object",
                        "properties": {"lease": {"type": "string"}},
                        "required": ["lease"],
                    },
                ),
            ]
            return spec

    with pytest.raises(ValueError, match="already takes an argument named"):
        TaskStream(
            lambda _name: _Colliding(tasks=TASKS),
            [TaskRef(ENV_NAME, 0)],
            prov_dir=tmp_path / "prov",
            max_in_flight=2,
        )


def _env_whose_submit_schema_is(schema: Dict[str, Any]) -> Any:
    """A factory for the fixture env with one hand-written schema on its score terminal. Only
    the *advertised* schema changes: the tool, the handler and the grading are the fixture's."""
    from shogym.task import TaskSpec

    class _CustomSchema(_FixtureScoreEnv):
        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            spec.tools = [
                tool.model_copy(update={"input_schema": json.loads(json.dumps(schema))})
                if tool.name == SUBMIT_TOOL
                else tool
                for tool in spec.tools
            ]
            return spec

    return lambda _name: _CustomSchema(tasks=TASKS)


_UNWRAPPABLE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # Valid, and its closed object is reached through the root `$ref`. Writing root `properties`
    # beside it either loses the native arguments outright (a `$ref` sibling is ignored) or
    # leaves the referenced `additionalProperties: false` refusing the very `lease` the root
    # requires — an unsatisfiable tool.
    "ref_rooted": {
        "$defs": {
            "submission": {
                "type": "object",
                "properties": {"answer": {"type": "string"}, "_session_id": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "$ref": "#/$defs/submission",
    },
    # Each branch applies to the WHOLE instance, so a closed branch refuses the added argument.
    "allof_rooted": {
        "type": "object",
        "allOf": [
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "additionalProperties": False,
            }
        ],
    },
    "oneof_rooted": {
        "type": "object",
        "oneOf": [
            {"type": "object", "required": ["answer"]},
            {"type": "object", "required": ["_session_id"]},
        ],
    },
    # Constrains every name the object may carry — including the one being added.
    "property_names": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "propertyNames": {"maxLength": 6},
    },
    # One more required name than the object may hold.
    "max_properties": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "maxProperties": 1,
    },
    # Not proved to be an object schema at all: nothing says the root takes named arguments.
    "no_type": {"properties": {"answer": {"type": "string"}}},
    "not_an_object": {"type": "string"},
    # `required` is read and rewritten, so a value that is not an array is not something the
    # lease can be added to: `[*"answer", "lease"]` requires six single letters and a lease.
    "required_is_text": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": "answer",
    },
    # The env owns the word, and says so only in `required`. Adding it there twice is not a
    # schema this endpoint may register, and stripping it would take the env's own argument.
    "lease_required_by_the_env": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["lease"],
    },
}


@pytest.mark.parametrize("shape", sorted(_UNWRAPPABLE_SCHEMAS))
async def test_a_schema_the_lease_cannot_be_added_to_is_refused(
    tmp_path: Path, shape: str
) -> None:
    # The wrapper writes `properties`/`required` at the schema root. That is sound for a plain
    # root object schema and for nothing else: against these shapes it advertises a contract the
    # episode does not enforce, so a client that sends exactly what the endpoint permits seals as
    # an earned-looking zero, or the tool cannot be satisfied at all. Refused at construction —
    # before an env is opened or a task is spent — rather than transformed into something whose
    # meaning nobody can prove.
    with pytest.raises(ValueError, match="cannot be wrapped|already takes an argument named"):
        TaskStream(
            _env_whose_submit_schema_is(_UNWRAPPABLE_SCHEMAS[shape]),
            [TaskRef(ENV_NAME, 0)],
            prov_dir=tmp_path / "prov",
            max_in_flight=2,
        )


@pytest.mark.parametrize("shape", sorted(_UNWRAPPABLE_SCHEMAS))
async def test_capacity_one_serves_a_schema_no_wrapper_could_carry(
    tmp_path: Path, shape: str
) -> None:
    # The refusal is the wrapper's, so it belongs to the wrapper's capacity. With one slot
    # nothing is added to any schema and the env is served exactly as it wrote itself — the same
    # promise `test_capacity_one_publishes_no_lease` makes, held for the schemas above.
    if shape == "lease_required_by_the_env":
        pytest.skip("the env's own `lease` argument is only a collision above capacity 1")
    async with TaskStream(
        _env_whose_submit_schema_is(_UNWRAPPABLE_SCHEMAS[shape]),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    ) as stream:
        published = {tool.name: tool.input_schema for tool in stream.tools}
        assert published[SUBMIT_TOOL] == _UNWRAPPABLE_SCHEMAS[shape]


async def test_a_schema_the_lease_can_be_added_to_keeps_every_native_argument(
    tmp_path: Path,
) -> None:
    # The other half of the refusal: what it accepts, it must carry whole. A root object schema
    # may still carry a definitions container, annotations and a closed `additionalProperties`,
    # and the argument the env actually grades has to survive to the registered contract.
    native: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "submit",
        "description": "the env's own framing",
        "$defs": {"text": {"type": "string"}},
        "type": "object",
        "properties": {"answer": {"$ref": "#/$defs/text"}, "_session_id": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    stream = TaskStream(
        _env_whose_submit_schema_is(native),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        max_in_flight=2,
    )
    async with stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            schema = {tool.name: tool for tool in await client.list_tools()}[
                SUBMIT_TOOL
            ].inputSchema
            assert "lease" in schema["properties"] and "lease" in schema["required"]
            assert "answer" in schema["properties"], "the env's own argument was dropped"
            assert schema["required"] == ["answer", "lease"]
            assert schema.get("additionalProperties") is False

            task = _payload_from_tool(await client.call_tool("get_task", {}))
            out = _payload_from_tool(
                await client.call_tool(SUBMIT_TOOL, {"answer": "4", "lease": task["lease"]})
            )
            assert out["terminated"] is True
    assert stream.results[0].score is not None and stream.results[0].score.success is True


# ----- capacity -----


async def test_a_pull_beyond_capacity_seals_the_oldest(tmp_path: Path) -> None:
    # Capacity is a promise about how many tasks may be worked at once, not a promise that a
    # task can be forgotten: the displaced one is sealed and lands its row.
    async with _stream(tmp_path, [0, 1, 2], max_in_flight=2) as stream:
        first = await stream.get_task()
        await stream.get_task()
        third = await stream.get_task()
        assert first is not None and third is not None
        assert stream.queue_info().in_flight == 2
        assert [row.task_idx for row in stream.results] == [0]
        assert stream.results[0].closure == "drained"

        stale = _payload(await stream.dispatch("noop", {"lease": first.lease}))
        assert stale["error"] == "sealed_lease"
    assert len(stream.results) == 3


async def test_a_pull_with_no_task_left_displaces_nothing(tmp_path: Path) -> None:
    # A pull is a request for a *slot*, and a slot is only worth taking if there is a task to
    # put in it. Displacing the oldest episode before asking whether the queue still holds one
    # turns the call whose only purpose is to learn that the run is over into a forced terminal:
    # the oldest unfinished task is scored as an ordinary agent-driven loss, over an answer the
    # agent was still free to submit.
    async with _stream(tmp_path, [0, 1], max_in_flight=2) as stream:
        first = await stream.get_task()
        second = await stream.get_task()
        assert first is not None and second is not None

        assert await stream.get_task() is None
        assert stream.queue_info().in_flight == 2, "an exhausted pull sealed a live episode"
        assert stream.results == (), "an exhausted pull recorded a row"

        # Both are still the agent's to finish, and both are earned.
        for lease, answer in ((first.lease, "4"), (second.lease, "6")):
            done = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": answer, "lease": lease}))
            assert done["terminated"] is True
    assert [(row.position, row.closure) for row in stream.results] == [
        (0, "sealed"),
        (1, "sealed"),
    ]
    assert all(row.score is not None and row.score.success for row in stream.results)
    assert [row["closure"] for row in _rows(tmp_path)] == ["sealed", "sealed"]


async def test_the_end_of_the_queue_is_not_told_as_nothing_being_live(tmp_path: Path) -> None:
    # `done` is a statement about the QUEUE. Answering it beside `in_flight: 0` while a lease is
    # still callable tells a worker both that no task is coming and that it has nothing left to
    # finish — so it stops, and the task it was working lands as a forced `drained` loss it
    # could have answered.
    async with _stream(tmp_path, [0, 1], max_in_flight=2) as stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            first = _payload_from_tool(await client.call_tool("get_task", {}))
            second = _payload_from_tool(await client.call_tool("get_task", {}))
            over = _payload_from_tool(await client.call_tool("get_task", {}))
            assert over == {
                "done": True,
                "remaining": 0,
                "consumed": 2,
                "in_flight": 2,
            }
            for payload, answer in ((first, "4"), (second, "6")):
                out = _payload_from_tool(
                    await client.call_tool(SUBMIT_TOOL, {"answer": answer, "lease": payload["lease"]})
                )
                assert out["terminated"] is True
            # Nothing is live now, and the same call says so.
            assert _payload_from_tool(await client.call_tool("get_task", {})) == {
                "done": True,
                "remaining": 0,
                "consumed": 2,
                "in_flight": 0,
            }
    assert all(row.closure == "sealed" for row in stream.results)
    assert all(row.score is not None and row.score.success for row in stream.results)


async def test_a_capacity_that_is_not_a_whole_number_is_refused(tmp_path: Path) -> None:
    # A capacity is read three ways downstream — sliced against the live entries a pull
    # displaces, and compared to 1 to decide both whether a lease is advertised and whether a
    # call must carry one — and a value that is not a whole number answers each of them
    # differently. So the stream half-works instead of refusing: `1.5` hands out a task and then
    # raises `TypeError` from the *next* pull, and `nan` hands out a task with no lease and then
    # refuses every call on it as `missing_lease`.
    built: List[str] = []

    def _factory(name: str) -> _FixtureScoreEnv:
        built.append(name)
        return _FixtureScoreEnv(tasks=TASKS)

    for value in (1.5, float("nan"), float("inf"), "2", None):
        with pytest.raises(ValueError, match="max_in_flight must be a whole number of slots"):
            TaskStream(
                _factory,
                [TaskRef(ENV_NAME, 0)],
                prov_dir=tmp_path / "prov",
                max_in_flight=value,  # type: ignore[arg-type]
            )
    # Refused before anything was built: a constructor that raises hands back no object, so an
    # env opened on the way to the refusal is one nobody can ever close.
    assert built == []


# ----- the deadline, with more than one episode live -----


async def _block_the_forced_terminal(live: Any, gate: asyncio.Event) -> asyncio.Event:
    """Make this episode's next tool call — the terminal the stream drives on the agent's
    behalf — wait for `gate`, and report when it is inside."""
    entered = asyncio.Event()
    original = live.episode.call

    async def blocked(tool: str, arguments: Any = None) -> Any:
        entered.set()
        await gate.wait()
        return await original(tool, arguments)

    live.episode.call = blocked
    return entered


def _arm_deadline(stream: TaskStream, deadline: float) -> None:
    """Turn the deadline on *after* the test's setup is in place.

    An episode's clock starts the moment it is dispensed, so a stream constructed with a live
    deadline is racing the rest of the setup below: the watchdog can reap the first episode,
    dropping its entry from `_live`, before the test has taken the handle it installs its block
    on. That window is not made safe by a faster machine; it depends on the scheduler never
    handing the watchdog a turn at the wrong moment, which is exactly what a loaded runner does
    differently (a 60ms yield between the two dispenses reproduces the `KeyError` every time).

    Arming here instead means no wall clock decides whether the setup completes. It costs the
    tests nothing: what they are about is *which* tasks the watchdog claims in one scan and in
    what order it waits on them, and they already choose the moment of expiry themselves, so the
    deadline only has to be enforced once the episodes they mean to expire exist.
    """
    stream._deadline = deadline  # noqa: SLF001
    stream._start_watchdog()  # noqa: SLF001


async def test_every_expired_task_is_claimed_before_any_seal_is_waited_on(
    tmp_path: Path,
) -> None:
    # Sealing is not a step the watchdog can take one task at a time: a seal drives an env
    # terminal, waits on its finalizer and runs every extension, any of which may block. If the
    # claim were taken inside that wait, the tasks after the blocked one would be expired and
    # still unclaimed — and an unclaimed task is one `_resolve` keeps routing calls to, so the
    # agent could earn a scored, `sealed` row on a task whose clock ran out.
    gate = asyncio.Event()
    async with _stream(tmp_path, [0, 1], max_in_flight=2) as stream:
        first = await stream.get_task()
        second = await stream.get_task()
        assert first is not None and second is not None
        entered = await _block_the_forced_terminal(
            stream._live[first.lease], gate  # noqa: SLF001
        )
        # Two clocks that ran out together, so the watchdog meets both in one scan. Waiting them
        # out in real time would not say this: the two dispenses are milliseconds apart, so the
        # watch would usually reach the first on an earlier tick than the second, and the test
        # would be about the tick rather than about the scan.
        started = min(live.started for live in stream._live.values())  # noqa: SLF001
        for live in stream._live.values():  # noqa: SLF001
            live.started = started - 1.0
        # Both are already out of time before the clock exists, so the first scan meets both.
        _arm_deadline(stream, 0.05)
        await asyncio.wait_for(entered.wait(), timeout=5)  # the watchdog is stuck on the first

        payload = _payload(
            await stream.dispatch(SUBMIT_TOOL, {"answer": "6", "lease": second.lease})
        )
        gate.set()  # released before anything can fail, so a regression fails rather than hangs
        assert payload.get("error") == "sealed_lease", (
            "a task past its deadline was still answerable because another task's seal blocked"
        )
    closures = sorted((row.position, row.closure, row.score) for row in stream.results)
    assert closures == [(0, "timeout", None), (1, "timeout", None)]


async def test_a_blocked_seal_does_not_stop_the_clock_for_a_later_task(
    tmp_path: Path,
) -> None:
    # The same defect one step along, and the one claiming-before-waiting does not reach: with a
    # free slot the queue keeps moving while a seal is stuck, so a watch that waited in its own
    # loop would never even look at the task dispensed after it. At capacity 1 there is nothing
    # to lose — a stuck seal blocks the next pull too — which is why this only exists here.
    gate = asyncio.Event()
    async with _stream(tmp_path, [0, 1, 2], max_in_flight=3) as stream:
        first = await stream.get_task()
        assert first is not None
        entered = await _block_the_forced_terminal(
            stream._live[first.lease], gate  # noqa: SLF001
        )
        # The clock starts here, on an episode this test still holds: the block is installed, so
        # the seal the deadline forces is the one that gets stuck. `first` was dispensed a moment
        # ago, so its deadline is a real one; what the arming removes is only the race between
        # the watchdog and the lookup above.
        _arm_deadline(stream, 0.05)
        await asyncio.wait_for(entered.wait(), timeout=5)

        later = await stream.get_task()  # dispensed while the first seal is stuck
        assert later is not None
        for _ in range(200):
            await asyncio.sleep(0.01)
            if later.lease not in stream._live:  # noqa: SLF001 — its seal claimed and finished
                break
        payload = _payload(
            await stream.dispatch(SUBMIT_TOOL, {"answer": "6", "lease": later.lease})
        )
        gate.set()  # released before anything can fail, so a regression fails rather than hangs
        assert payload.get("error") == "sealed_lease", (
            "a task dispensed behind a blocked seal was never clocked"
        )
    assert [row.closure for row in stream.results] == ["timeout", "timeout"]
    assert all(row.score is None for row in stream.results)


# ----- shutdown, with more than one episode live -----


async def test_every_live_task_is_claimed_before_the_drain_waits_on_any_seal(
    tmp_path: Path,
) -> None:
    # The shutdown twin of the deadline test above, and the same hazard one caller along. The
    # drain marks the stream closed and then seals what is live; `_resolve` routes on the entry
    # rather than on that flag, so claims taken one at a time — inside each seal — would leave
    # every episode after the blocked one answerable while the stream already read closed. The
    # agent's submission would then be accepted mid-shutdown and recorded as an ordinary earned
    # `sealed` row: a stop that did not happen, filed as a score a run would average in.
    gate = asyncio.Event()
    stream = _stream(tmp_path, [0, 1], max_in_flight=2)
    await stream.__aenter__()
    first = await stream.get_task()
    second = await stream.get_task()
    assert first is not None and second is not None
    entered = await _block_the_forced_terminal(stream._live[first.lease], gate)  # noqa: SLF001

    closing = asyncio.ensure_future(stream.aclose())
    await asyncio.wait_for(entered.wait(), timeout=5)  # the drain is stuck on the first seal
    assert stream._closed is True, "the drain has not begun; this test is not testing anything"

    payload = _payload(
        await stream.dispatch(SUBMIT_TOOL, {"answer": "6", "lease": second.lease})
    )
    # Read before the gate is released, so it is what the record held *during* the shutdown: no
    # seal has finished, so no row exists at all.
    mid_shutdown = (tmp_path / "prov" / "results.jsonl").exists()
    gate.set()  # released before anything can fail, so a regression fails rather than hangs
    await closing
    assert payload.get("error") == "sealed_lease", (
        "a submission was accepted after the stream closed, because another task's seal blocked"
    )
    assert not mid_shutdown, "a task the shutdown had already claimed still recorded a row"
    # What the record holds for both is the ending the *stream* imposed. A submission that
    # arrived after the shutdown began earns nothing: it costs no budget, enters no trajectory,
    # and the row its task lands is the drain's own.
    assert sorted((row.position, row.closure) for row in stream.results) == [
        (0, "drained"),
        (1, "drained"),
    ]
    assert all(
        row.score is not None and row.score.success is False for row in stream.results
    )
    assert [row["closure"] for row in _rows(tmp_path)] == ["drained", "drained"]


# ----- what a finished task leaves behind -----


async def test_a_finished_task_leaves_only_its_lease(tmp_path: Path) -> None:
    # A sealed entry has to answer a late call with `sealed_lease` rather than `unknown_lease`,
    # and that is the whole of what it is still owed. Keeping the entry to say it would keep the
    # episode behind it — its env, its sessions, the task payload and the whole trajectory —
    # reachable for the length of the run, so a long queue's memory would grow with every task it
    # had already scored rather than with its rows.
    held: List[Any] = []
    async with _stream(tmp_path, [0, 1, 2], max_in_flight=2) as stream:
        for answer in ("4", "6", "10"):
            task = await stream.get_task()
            assert task is not None
            live = stream._live[task.lease]  # noqa: SLF001
            held.append((task.lease, weakref.ref(live.episode)))
            del live
            await stream.dispatch(SUBMIT_TOOL, {"answer": answer, "lease": task.lease})

        assert stream.queue_info().in_flight == 0
        assert stream._live == {}, "a finished episode is still reachable from the registry"

        # Still refused as over, not as never dispensed.
        stale = _payload(await stream.dispatch("noop", {"lease": held[0][0]}))
        assert stale["error"] == "sealed_lease"
        assert _payload(await stream.dispatch("noop", {"lease": "0" * 32}))["error"] == (
            "unknown_lease"
        )

    gc.collect()
    assert [ref() for _, ref in held] == [None, None, None]
    assert [row.closure for row in stream.results] == ["sealed", "sealed", "sealed"]


# ----- a task the stream has ended, whose row could not be written -----


def _unwritable_results(tmp_path: Path) -> None:
    """Make the real append fail without reaching inside the stream: a *directory* where the
    results file goes, so `open("a")` raises where a full or read-only volume would."""
    (tmp_path / "prov" / "results.jsonl").mkdir(parents=True, exist_ok=True)


class _TrackedEnv(_FixtureScoreEnv):
    """A fixture env whose release is observable, for a refusal nothing else holds."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        await super().close()


def _watch_calls(live: Any) -> List[str]:
    """Record every tool name that actually reaches this episode."""
    reached: List[str] = []
    original = live.episode.call

    async def watched(tool: str, arguments: Any = None) -> Any:
        reached.append(tool)
        return await original(tool, arguments)

    live.episode.call = watched
    return reached


async def test_a_row_that_could_not_be_written_does_not_leave_its_task_answerable(
    tmp_path: Path,
) -> None:
    # A seal that fails on the storage hands its claim back so a later drain can retry the
    # *append* — and the claim is what `_resolve` was reading to know a task is over. So the one
    # task whose record already failed became the one task a late call is routed to: the stream
    # has force-terminated the episode, composed its row and finalized every span, and the agent
    # is still being let in. What comes back is the terminating payload rather than the refusal
    # every other finished task gives, which also tells the agent, from the shape of the answer,
    # that this task's record went wrong.
    stream = _stream(tmp_path, [0, 1], max_in_flight=2)
    first = await stream.get_task()
    second = await stream.get_task()
    assert first is not None and second is not None
    _unwritable_results(tmp_path)

    reached = _watch_calls(stream._live[first.lease])  # noqa: SLF001
    ended = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": first.lease}))
    assert ended["terminated"] is True
    assert stream.stopped, "a row that could not be written must stop the stream"
    after_terminal = list(reached)

    late = _payload(await stream.dispatch("noop", {"lease": first.lease}))
    assert late["error"] == "sealed_lease", "a task the stream already ended was answerable"
    assert reached == after_terminal, "a late call reached an episode the stream had ended"

    # The task next to it is untouched: this refusal is about one entry, not about a stopped
    # stream refusing everything.
    healthy = _payload(await stream.dispatch("noop", {"lease": second.lease}))
    assert healthy["terminated"] is False

    # And the refusal is the one a task whose row *did* land gives, word for word: what a late
    # call may tell the agent is that its task is over, never that the record failed.
    async with _stream(tmp_path / "clean", [0], max_in_flight=2) as healthy_stream:
        task = await healthy_stream.get_task()
        assert task is not None
        await healthy_stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": task.lease})
        sealed = _payload(await healthy_stream.dispatch("noop", {"lease": task.lease}))
    assert late == sealed

    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()
    assert stream.results == ()


async def test_a_task_the_drain_has_claimed_is_not_answerable_when_its_row_fails(
    tmp_path: Path,
) -> None:
    # The same window during shutdown, which is where it matters most: `aclose` claims every
    # unsettled task in the critical section that closes the stream, precisely so no episode is
    # left routable — and the first seal to *fail* hands its claim straight back, reopening that
    # window from the other end while the drain is still waiting on another task's seal. A
    # submission accepted there is work taken on after the shutdown began.
    gate = asyncio.Event()
    stream = _stream(tmp_path, [0, 1], max_in_flight=2)
    await stream.__aenter__()
    first = await stream.get_task()
    second = await stream.get_task()
    assert first is not None and second is not None
    _unwritable_results(tmp_path)

    live_first = stream._live[first.lease]  # noqa: SLF001
    reached = _watch_calls(live_first)
    # Hold the *second* task's forced terminal so the drain is still inside its join loop after
    # the first task's seal has failed and handed its claim back.
    entered = await _block_the_forced_terminal(stream._live[second.lease], gate)  # noqa: SLF001

    closing = asyncio.ensure_future(stream.aclose())
    await asyncio.wait_for(entered.wait(), timeout=5)
    for _ in range(500):  # both seals were claimed together; wait for the first one to fail
        await asyncio.sleep(0.01)
        if live_first.sealing is None:
            break
    assert stream._closed is True, "the drain has not begun; this test is not testing anything"
    assert live_first.sealing is None, "the seal never failed; this test is not testing anything"
    during = list(reached)

    refused = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": first.lease}))
    gate.set()  # released before anything can fail, so a regression fails rather than hangs
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await closing
    assert refused.get("error") == "sealed_lease", (
        "a task the drain had already ended accepted a submission mid-shutdown"
    )
    assert reached == during, "a mid-shutdown call reached an episode the drain had ended"
    # ...and the window does not reopen once the drain has returned, either.
    after = _payload(await stream.dispatch("noop", {"lease": first.lease}))
    assert after.get("error") == "sealed_lease"
    assert reached == during
    assert stream.results == ()


async def test_a_handed_back_seal_is_still_retried_after_its_lease_is_refused(
    tmp_path: Path,
) -> None:
    # The other half of the contract, and the reason the claim is handed back at all: refusing
    # the lease may not cost the task the row it is still owed. A later drain has to retry the
    # *append* — the retained row, not a recomposed one, so the closure still says the stream
    # ended this task and every extension's `finalize` still ran exactly once.
    stream = _stream(tmp_path, [0, 1], max_in_flight=2)
    first = await stream.get_task()
    assert first is not None
    _unwritable_results(tmp_path)
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": first.lease})
    assert stream.results == () and stream.stopped

    live = stream._live[first.lease]  # noqa: SLF001
    assert live.pending_row is not None, "the composed row was not retained for the retry"
    assert _payload(await stream.dispatch("noop", {"lease": first.lease}))["error"] == (
        "sealed_lease"
    )

    (tmp_path / "prov" / "results.jsonl").rmdir()  # the storage comes back
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()  # still reports the stop, and lands the row it was holding
    assert [(row.position, row.closure) for row in stream.results] == [(0, "sealed")]
    assert [row["closure"] for row in _rows(tmp_path)] == ["sealed"]


async def test_a_lease_that_cannot_be_minted_afresh_is_refused_rather_than_retried_forever(
    tmp_path: Path,
) -> None:
    # Minting loops until it draws a lease this run has not issued, which is a loop with no
    # bound: against a source that cannot produce a fresh value it spins inside the dispense
    # lock, holding the event loop, with no task, no error and no way for a harness to find out
    # why — and synchronously, so no timeout anywhere can interrupt it. A run that cannot name
    # its next task has to say so.
    #
    # The scripted source therefore runs out and falls back to the real one: an unbounded loop
    # fails this test by *succeeding* on the 65th draw rather than by hanging the suite, and what
    # the test pins is the bound.
    built: List[_TrackedEnv] = []

    def factory(_name: str) -> _TrackedEnv:
        env = _TrackedEnv(tasks=TASKS)
        built.append(env)
        return env

    tokens = _ScriptedTokens(*["d" * 32] * 64)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stream_module, "secrets", tokens)
        stream = TaskStream(
            factory,
            [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
            prov_dir=tmp_path / "prov",
            max_in_flight=2,
        )
        first = await stream.get_task()
        assert first is not None and first.lease == "d" * 32
        with pytest.raises(RuntimeError, match="could not mint a lease"):
            # Bounded only so a regression that reaches the loop below the bound fails here
            # rather than hanging the suite; generous, because opening the episode this pull
            # refuses is real work and this is not a test about how long that takes.
            await asyncio.wait_for(stream.get_task(), timeout=120)
        # The episode opened for the task that was never handed out is released with it: a
        # refusal between opening an episode and dispensing it owes the same cleanup every
        # other refusal there does. (`built` is the catalog env, then one per opened episode.)
        assert len(built) == 3, "an episode was not opened for the refused dispense"
        assert built[-1].closed, "the undispensed episode's env was left open"
        await stream.aclose()
