"""The bank: what is frozen before launch, and the one render after a filing seals.

A bank is a generator, a key and a count. Which instances it holds is recomputed by
rerunning admission in ordinal order, so there is no list of passers to disagree
with the rule that made them.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts.generators.ledger import GENERATOR
from shogym.envs.receipts.receipt_ast import GRADED, ORACLE, PLACEBO

MASTER = bytes(range(32, 64))


@functools.lru_cache(maxsize=2)
def _bank(size: int = 2) -> bank_mod.Bank:
    """One materialized bank, cached: filling one reruns admission per ordinal."""
    return bank_mod.materialize(GENERATOR, MASTER, size)


@functools.lru_cache(maxsize=2)
def _population(size: int = 2) -> bank_mod.Population:
    return bank_mod.population(_bank(size), GENERATOR)


def test_a_bank_holds_the_instances_admission_admits_in_order() -> None:
    built = _bank(2)
    found = _population(2)
    assert built.size == 2
    assert len(found.instances) == 2
    assert found.considered >= 2
    assert found.passing_fraction == pytest.approx(2 / found.considered)
    assert list(found.ordinals) == sorted(set(found.ordinals))
    assert built.renderer == bank_mod.RENDERER_CONFIGURATION


def test_the_population_is_the_same_set_every_time_it_is_recomputed() -> None:
    built = _bank(2)
    first = bank_mod.population(built, GENERATOR)
    again = bank_mod.population(built, GENERATOR)
    assert first.ordinals == again.ordinals
    assert first.considered == again.considered
    assert [bank_mod.instance_digest(i, GENERATOR) for i in first.instances] == [
        bank_mod.instance_digest(i, GENERATOR) for i in again.instances
    ]


def test_a_bank_that_names_another_generator_does_not_fill() -> None:
    built = _bank(1)
    moved = bank_mod.Bank(
        generator="elsewhere", genre=built.genre, renderer=built.renderer,
        master=built.master, size=built.size,
    )
    with pytest.raises(ValueError, match="names elsewhere"):
        bank_mod.population(moved, GENERATOR)


def test_a_bank_frozen_under_another_renderer_does_not_fill() -> None:
    built = _bank(1)
    moved = bank_mod.Bank(
        generator=built.generator, genre=built.genre, renderer="receipts-render-v0",
        master=built.master, size=built.size,
    )
    with pytest.raises(ValueError, match="the cells would not be the cells it gated"):
        bank_mod.population(moved, GENERATOR)


def test_an_ordinal_outside_the_bank_cannot_be_drawn() -> None:
    with pytest.raises(KeyError):
        _population(2).instance(999)


def test_a_bank_survives_a_round_trip_through_a_file(tmp_path: Path) -> None:
    built = _bank(2)
    path = tmp_path / "ledger.json"
    digest = bank_mod.save_bank(built, path)
    read_back = bank_mod.load_bank(path)
    assert len(digest) == 64
    assert read_back == built


def test_a_bank_record_is_exactly_five_fields() -> None:
    built = _bank(1)
    record = bank_mod.bank_record(built)
    assert set(record) == {"generator", "genre", "renderer", "master", "size"}
    with pytest.raises(ValueError, match="carries exactly"):
        bank_mod.bank_from_record({**record, "considered": 9})
    with pytest.raises(ValueError, match="carries exactly"):
        bank_mod.bank_from_record({k: v for k, v in record.items() if k != "size"})


def test_the_commitment_binds_the_convention_without_printing_it() -> None:
    built = _bank(1)
    instance = _population(1).instances[0]
    recorded = bank_mod.commitment(built.master, instance.ordinal, instance.convention)
    for option in instance.convention.values():
        assert option not in recorded
    other = dict(instance.convention)
    other["boundary"] = "upper" if other["boundary"] == "lower" else "lower"
    assert bank_mod.commitment(built.master, instance.ordinal, other) != recorded


def test_two_renders_of_one_filing_produce_the_same_bytes() -> None:
    """Determinism, which is what makes discarding a concurrent loser's bytes safe.

    This is not a render-once test and does not claim to be one: it calls the renderer
    twice on purpose. What a run relies on is that a second render cannot disagree
    with the first, so a caller that loses the publication race can drop its own bytes
    for the winner's and every branch still reads one set.
    """
    built = _bank(1)
    instance = _population(1).instances[0]
    raw = bank_mod.review_filing(GENERATOR, instance, "a", "mixed", built.master)
    first = bank_mod.render_fork(GENERATOR, instance, "a", raw)
    second = bank_mod.render_fork(GENERATOR, instance, "a", raw)
    for kind in (GRADED, PLACEBO, ORACLE):
        assert first.agent_bytes(kind) == second.agent_bytes(kind)
        assert first.digests[kind] == second.digests[kind]


def test_a_fork_serves_only_the_three_registered_cells() -> None:
    instance = _population(1).instances[0]
    fork = bank_mod.render_fork(GENERATOR, instance, "a", "")
    with pytest.raises(ValueError, match="a fork serves"):
        fork.agent_bytes("hint")


def test_the_graded_and_placebo_cells_differ_only_inside_the_slots() -> None:
    """The judgement the fork applies compares the two cells outside their registered
    slots, so a byte the slots do not account for cannot carry the rule."""
    from shogym.envs.receipts.receipt_ast import (
        GRADED,
        PLACEBO,
        frozen_envelope,
        mask_slots,
        slot_ranges,
    )
    from shogym.envs.receipts.render import judge_cells

    instance = _population(1).instances[0]
    envelope = frozen_envelope(instance.envelope)
    canonical = GENERATOR.parse_and_canonicalize(instance.a, "")
    judged = judge_cells(GENERATOR, instance.a, canonical, instance.convention, envelope)
    assert judged.acceptable, judged.problems
    ranges = slot_ranges(judged.asts[GRADED], envelope)
    assert mask_slots(judged.payloads[GRADED], ranges) == mask_slots(
        judged.payloads[PLACEBO], ranges
    )
    nudged = bytearray(judged.payloads[PLACEBO])
    nudged[0:1] = b"x"
    assert mask_slots(judged.payloads[GRADED], ranges) != mask_slots(bytes(nudged), ranges)


def test_every_registered_filing_shape_renders() -> None:
    built = _bank(1)
    instance = _population(1).instances[0]
    for shape in bank_mod.FILING_SHAPES:
        raw = bank_mod.review_filing(GENERATOR, instance, "a", shape, built.master)
        fork = bank_mod.render_fork(GENERATOR, instance, "a", raw)
        assert 0.0 <= fork.component_score <= 1.0
        assert len(fork.graded) == instance.envelope.size
    with pytest.raises(ValueError, match="a filing shape is one of"):
        bank_mod.review_filing(GENERATOR, instance, "a", "improvised", built.master)


def test_the_mixed_shape_puts_both_verdicts_on_the_receipt() -> None:
    built = _bank(1)
    instance = _population(1).instances[0]
    raw = bank_mod.review_filing(GENERATOR, instance, "a", "mixed", built.master)
    graded = bank_mod.render_fork(GENERATOR, instance, "a", raw).graded.decode()
    assert "PASS" in graded and "FAIL" in graded
