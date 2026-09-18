"""What the families that report every row did before the receipt policy existed.

A change to which rows one family's receipt reports must not move another family's
artifact. These hold the cells, the serialized receipts, the gate results and the named
check results of sound change and of every gate vector against the values they produced
on the tree the policy was added to, taken there and frozen here.

Ledger is deliberately not in the fixture: its receipt reports four records of twenty
four now, so its graded cells and its gate results moved, and that is the change. What
did not move is everything else, and this is what says so.

SOUND CHANGE IS HERE AS A FAMILY THAT REPORTS EVERY ROW, WHICH IT NO LONGER IS. It now
declares two forms of twenty four, so the family as it is served produces other bytes,
and that is its own change rather than a defect in this one. What these hold is the
same genre under the full receipt, through a subclass that declares it: the tables, the
answers, the oracle and the gate arithmetic are the family's own, and the only thing the
subclass changes is the count. Every value below is the value that genre produced before
a receipt policy was declared anywhere, so a change to any of them is a change the
policy machinery made to a family that reports every row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shogym.envs.receipts import streams
from shogym.envs.receipts.admission import Thresholds, report
from shogym.envs.receipts.bank import render_fork
from shogym.envs.receipts.generators import soundchange
from shogym.envs.receipts.generators.vectors import VECTORS
from shogym.envs.receipts.observe import observe
from shogym.envs.receipts.protocol import FULL_RECEIPT, ReceiptPolicy, draw
from shogym.receipts import gate

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "_fixtures"
    / "full_receipt_before_the_policy.json"
)
MASTER = bytes(range(32))
FROZEN = json.loads(FIXTURE.read_text(encoding="utf-8"))


class FullSoundChange(soundchange.SoundChangeGenerator):
    """Sound change as it was: a verdict and a correction on every form.

    The name, the streams, the tables, the answers and the oracle are the family's, so
    what this draws is what the family drew; the count is the only thing that moves.
    """

    RECEIPT_POLICY: ReceiptPolicy = FULL_RECEIPT


def _generator(name: str):
    return FullSoundChange() if name == "soundchange" else VECTORS[name]


def _filings(generator, task) -> list[tuple[str, str]]:
    identifiers = generator.row_identifiers(task.table)
    return [
        ("canonical", "\n".join(f"{i},{v}" for i, v in zip(identifiers, task.key))),
        (
            "half",
            "\n".join(
                f"{i},{v if n % 2 else 'Nope'}"
                for n, (i, v) in enumerate(zip(identifiers, task.key))
            ),
        ),
        ("empty", ""),
    ]


@pytest.mark.parametrize("key", sorted(FROZEN["cells"]))
def test_the_cells_a_full_policy_family_serves_have_not_moved(key: str) -> None:
    """Three cells, three filings, both siblings, frozen before the policy existed.

    FAILS IF sound change's graded, placebo or oracle bytes differ from the ones it
    produced before a receipt policy was declared anywhere. The task texts are held the
    same way. A family that reports every row is not the subject of this change and
    nothing about what it serves may move with it.
    """
    name, ordinal, *rest = key.split("/")
    generator = _generator(name)
    instance = draw(generator, MASTER, int(ordinal))
    if rest == ["text"]:
        assert {
            "a": streams.digest(instance.a.text.encode()),
            "b": streams.digest(instance.b.text.encode()),
        } == FROZEN["cells"][key]
        return
    side, shape = rest
    raw = dict(_filings(generator, instance.side(side)))[shape]
    fork = render_fork(generator, instance, side, raw)
    assert {
        kind: streams.digest(fork.agent_bytes(kind))
        for kind in ("graded", "placebo", "oracle")
    } == FROZEN["cells"][key]


@pytest.mark.parametrize("key", sorted(FROZEN["printed"]))
def test_the_serialized_receipts_of_a_full_policy_family_have_not_moved(key: str) -> None:
    """Every convention's serialized receipt, hashed together, per instance.

    FAILS IF any byte of what a full-policy family prints under any convention in its
    support has moved. This is the whole artifact the gates read, not one filing's
    three cells, and the gate vectors are held by it too: every expected number in that
    module is arithmetic on a receipt that reports every row.
    """
    name, ordinal = key.split("/")
    generator = _generator(name)
    observed = observe(generator, draw(generator, MASTER, int(ordinal)), "a")
    assert streams.digest(
        *(observed.payloads[position] for position in sorted(observed.payloads))
    ) == FROZEN["printed"][key]


@pytest.mark.parametrize("key", sorted(FROZEN["gates"]))
def test_the_gate_results_of_a_full_policy_family_have_not_moved(key: str) -> None:
    """Blocks, ceiling, floor, no-receipt level, evident rows, R, S, H and the verdict.

    FAILS IF any of them differs from the value the same instance produced before the
    policy existed. The gate identity moved, which is what a named version is for, and
    the numbers it reports on a family that reports every row did not.
    """
    name, ordinal = key.split("/")
    generator = _generator(name)
    scored = gate(
        observe(generator, draw(generator, MASTER, int(ordinal)), "a"),
        min_arity=3,
        min_blocks=2,
        min_headroom=0.05,
    )
    assert {
        "blocks": {k: int(v) for k, v in sorted(scored.blocks.items())},
        "ceiling": round(scored.ceiling, 9),
        "floor": round(scored.floor, 9),
        "placebo": round(scored.placebo, 9),
        "n_evident": scored.n_evident,
        "r_pass": scored.r_pass,
        "s_pass": scored.s_pass,
        "h_pass": scored.h_pass,
        "verdict": scored.verdict,
    } == FROZEN["gates"][key]


@pytest.mark.parametrize("key", sorted(FROZEN["checks"]))
def test_the_named_checks_of_a_full_policy_family_have_not_moved(key: str) -> None:
    """Every named check, by name and by outcome, and whether the instance is admitted.

    FAILS IF a full-policy family's check list or any of its verdicts differs from
    before. The first check is dispatched on the declared policy, so this also holds
    that such a family is still asked what its receipt exercised.
    """
    name, ordinal = key.split("/")
    generator = _generator(name)
    made = report(generator, draw(generator, MASTER, int(ordinal)), MASTER, Thresholds())
    assert {
        "admitted": made.admitted,
        "checks": [[c.name, c.passed] for c in made.checks],
    } == FROZEN["checks"][key]
