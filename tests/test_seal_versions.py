"""The version a seal is asked under, over every environment that captures one of its own.

A generation declares the canonicalization version its acknowledgements carry, and a port writes
its canonical submission text under the version it captures under. Those are two versions, and a
port that sealed whatever it was asked for would let them disagree: a caller who opened a
generation with a doctored descriptor would be answered with an acknowledgement naming one version
over a submission written under another, which is what the version is there to rule out. So a port
compares the two and refuses a request it cannot capture under, non-retryably and before it reads
anything at all, and this is where each of them is asked.

Before it reads anything is measured here rather than described. Every call a port could reach a
capture store through is watched, and a refused request has to leave that store untouched: a port
that read a capture and compared afterwards would be holding a record of a run it was never going
to answer, and the record is what a later call returns, so it is the refused request's own bytes
that a retry would then be answered with. The route this file hands each port raises rather than
returning a world, which is the same measurement for the world behind the store.

Each port is asked twice, over an empty store and over one already holding a capture under the
seal id. The second is the retry path, where a port that compared after reading would not fail on
the route at all: it would find the record, answer out of it, and seal a submission under a
version the request never asked for.

One case per port, and the ports are found in the package rather than typed out, so an environment
that arrives with a terminal of its own arrives with a case here rather than joining whichever
ones happened to exist when this was written.

The kernel's stand-in has no case here, because it has nothing to compare. Its canonical text is
the filing's own arguments and carries no version inside it, so there is no second version for a
request's to disagree with, and the version an acknowledgement over it names is the one the
generation declared. What the stream does with a result that came back under some other version is
the same fact from the other side, and it is asked where the results a seal cannot vouch for are.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import pytest

pytest.importorskip("temporalio")

from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.envs._grading import SEALED, CaptureStore, MemoryCaptures  # noqa: E402
from shogym.envs.wordle.protocol_v2 import SealStore  # noqa: E402
from shogym.serve.protocol_v2.gateway import CANONICALIZATION_VERSION  # noqa: E402
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    SealAttemptInput,
    SealAttemptResult,
)

from tests._fixtures.upstream_gate import gate  # noqa: E402

ATTEMPT = "b" * 32
SEAL_ID = "a" * 64

#: The version a doctored request here carries. It is the gateway's own, which is the version a
#: generation declares when the environment brings no terminal of its own and the one a descriptor
#: composed by hand carries unless it is given another. No environment captures under it, so it is
#: a version every port below has to refuse.
DOCTORED = CANONICALIZATION_VERSION


class Routed(Exception):
    """Raised where a seal asked this file's route for a world, which is as far as one gets here.

    A refused seal never reaches the route, because the version is compared before a world is
    read, and a seal that is not refused reaches nothing else over an empty store, because the
    route has no world to hand back. So a port that compared nothing, or compared after reading,
    is caught here or by the store this file watches.
    """


def route(attempt_id: str) -> Optional[Tuple[Any, str]]:
    """A route with no world behind it, asked only by a seal that got past the version."""
    raise Routed(f"attempt {attempt_id} was routed")


def entry_points(store: type) -> List[str]:
    """Every call one of these stores can be reached through, taken from the class itself.

    Reading the class rather than a list typed out here is what makes the watch below total: a
    store that grows a way in grows it here too, and a port reaching for it is seen.
    """
    return sorted(
        name for name in dir(store) if not name.startswith("_") and callable(getattr(store, name))
    )


def watched(store: type, name: str, reads: List[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Record one call into one store, and let it do what it does."""
    entry = getattr(store, name)

    def record(self: Any, *arguments: Any, **named: Any) -> Any:
        reads.append(f"{store.__name__}.{name}")
        return entry(self, *arguments, **named)

    monkeypatch.setattr(store, name, record)


def watch(reads: List[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Record every call that reaches a capture store, whichever store a port keeps.

    The two are the stores the ports here keep their captures in: the process-local one behind
    the shared grading port and the receipts port, and wordle's own directory of sealed plays.
    Both are watched for every case, so a port reaching for the other one is seen as well.
    """
    for store in (MemoryCaptures, SealStore):
        for name in entry_points(store):
            watched(store, name, reads, monkeypatch)


@dataclass(frozen=True)
class Port:
    """One environment's terminal port: the package it lives in, and the call that ends an attempt.

    ``terminal`` is the name a filing over that environment arrives under, so the request each
    case sends is the request production sends but for the version it carries.
    """

    name: str
    terminal: str


@dataclass(frozen=True)
class Terminal:
    """One port's seal, opened over no world, beside a way to put a capture under the seal id.

    ``hold`` installs the record that port's own seal writes on a first call, so the case that
    runs after it is the retry: the store has an answer, and reaching for it is what the version
    comparison has to come before.
    """

    version: str
    seal: Callable[[SealAttemptInput], Awaitable[SealAttemptResult]]
    hold: Callable[[], Dict[str, Any]]


#: Every environment here that ends its attempts its own way, with the call that ends one.
PORTS: Tuple[Port, ...] = (
    Port("automationbench", "done"),
    Port("receipts", "submit_filing"),
    Port("wordle", "terminate"),
    Port("yc_bench", "submit"),
)


def sharing(
    opened: Tuple[str, List[Any]], store: CaptureStore, capture: Dict[str, Any]
) -> Terminal:
    """A port that keeps its captures in the shared store, with the record its seal answers from.

    The record is what that port's own seal writes when it reads a world, cut down to the fields
    the seal reads back out of it: what a retry is answered with is this record and not a second
    reading, so this is the state such a port is in after one call.
    """
    version, activities = opened
    return Terminal(
        version=version,
        seal=activities[0],
        hold=lambda: store.once(SEAL_ID, SEALED, lambda: capture),
    )


def opened(port: Port, room: Path) -> Terminal:
    """The version one port declares and the seal it registers, opened over no world.

    Each is opened the way that environment's own tests open it, with the records kept where this
    test can throw them away. ``automationbench`` and ``yc_bench`` are provisioned from upstream at
    runtime, so their cases skip on a machine with neither the source nor the network to fetch it.
    """
    if port.name == "automationbench":
        gate(
            "shogym.envs.automationbench.adapter",
            package="automationbench",
            extra="automationbench",
        )
        from shogym.envs.automationbench.protocol_v2 import automationbench_terminal

        store = MemoryCaptures()
        return sharing(
            automationbench_terminal(route, store=store), store, {"world_sha256": "1" * 64}
        )
    if port.name == "receipts":
        from shogym.envs.receipts.protocol_v2 import receipts_terminal

        store = MemoryCaptures()
        return sharing(
            receipts_terminal(route, store=store), store, {"filing": "one row, filed"}
        )
    if port.name == "wordle":
        from shogym.envs.wordle.protocol_v2 import wordle_terminal

        version, activities = wordle_terminal(route, seals=room)
        played = {"target": "crane", "entries": ["slate"]}
        return Terminal(
            version=version,
            seal=activities[0],
            hold=lambda: SealStore(room).seal(SEAL_ID, lambda: played),
        )
    if port.name == "yc_bench":
        gate("shogym.envs.yc_bench.adapter", package="yc_bench", extra="yc_bench")
        from shogym.envs.yc_bench.protocol_v2 import yc_bench_terminal

        store = MemoryCaptures()
        return sharing(
            yc_bench_terminal(route, store=store), store, {"terminal_reason": "closed"}
        )
    raise AssertionError(f"{port.name} is in the table with no way to open it")


def ported() -> List[str]:
    """Every environment package here that brings a terminal port of its own."""
    root = Path(str(import_module("shogym.envs").__path__[0]))
    return sorted(module.parent.name for module in root.glob("*/protocol_v2.py"))


def test_every_environment_that_brings_a_port_has_a_case_here() -> None:
    """The table covers the package rather than the environments that existed when it was written.

    A port added later declares a version of its own and captures under it, so it has the same
    disagreement to rule out as the four here and fails this until it has a case.
    """
    assert sorted(port.name for port in PORTS) == ported()


@pytest.mark.parametrize("holding", [False, True], ids=["empty", "populated"])
@pytest.mark.parametrize("port", PORTS, ids=[port.name for port in PORTS])
async def test_a_seal_asked_under_a_version_its_port_cannot_capture_under_is_refused(
    port: Port, holding: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A doctored pairing is refused at the seal, by name, and nothing is read before it is.

    The refusal is non-retryable because nothing about it changes on a second attempt: the version
    the request carries is the generation's, and a generation does not acquire another by being
    asked again. It is also the refusal a mismatch gets rather than the one everything gets, which
    is what the control at the end reads: the same request under the version this port declares is
    past the comparison and into the store, and out to the world where the store holds nothing.
    """
    reads: List[str] = []
    watch(reads, monkeypatch)
    terminal = opened(port, tmp_path)
    assert terminal.version != DOCTORED
    if holding:
        # A capture under the seal id, installed through the store the port reads, so the case
        # below is the retry: there is a record here to be answered out of.
        assert terminal.hold()
        assert reads, "the capture was installed somewhere this does not watch"
    reads.clear()

    request = SealAttemptInput(
        attempt_id=ATTEMPT,
        seal_id=SEAL_ID,
        native_terminal_name=port.terminal,
        canonicalization_version=DOCTORED,
    )
    with pytest.raises(ApplicationError) as refused:
        await terminal.seal(request)
    assert refused.value.type == "CanonicalizationMismatch"
    assert refused.value.non_retryable is True
    # And it names both versions, because which two disagreed is the whole of what is wrong.
    assert repr(terminal.version) in str(refused.value)
    assert repr(DOCTORED) in str(refused.value)

    # Nothing was read on the way to that refusal, from a store that had an answer to give.
    assert reads == []

    # The control. The same request under the version this port declares is past the comparison,
    # so it reaches the store, and what it finds there is what a retry finds: the record where one
    # was installed, and the world behind the route where none was.
    matched = replace(request, canonicalization_version=terminal.version)
    if holding:
        sealed = await terminal.seal(matched)
        assert (sealed.seal_id, sealed.canonicalization_version) == (SEAL_ID, terminal.version)
    else:
        with pytest.raises(Routed):
            await terminal.seal(matched)
    assert reads
