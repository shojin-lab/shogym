"""The row-banded families: registration, construction, the perfect reader and the bank.

Each family here is a generator in `shogym.envs.receipts.generators` built on
`banded_rows`, and each test is parametrized over all of them, so a property is asked
of every family in the same words. Each test states the failure it is here to catch.

THE KEYS ARE FIXED. A bank's population is a function of its key, so a test that reads
an admitted bank reads the same one on every run.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import itertools
from collections import Counter
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import checks
from shogym.envs.receipts.admission import Thresholds, report
from shogym.envs.receipts.generators import banded_rows
from shogym.envs.receipts.protocol import (
    FULL_RECEIPT,
    SAMPLED_TWO_OF_TWENTY_FOUR,
    Instance,
    Task,
    conventions,
    draw,
    receipt_mask,
    support_of,
)
from shogym.envs.receipts.receipt_ast import frozen_envelope, serialize
from shogym.envs.receipts.receipt_law import law_for
from shogym.envs.receipts.registry import GENRES, load_generator
from shogym.envs.receipts.render import Feedback, feedback_for

#: The families under test, by registered name.
FAMILIES = ("scorecard",)

#: How many admitted families each bank holds. The brief's floor is twenty four.
BANK_SIZE = 24


def _master(name: str) -> bytes:
    return hashlib.sha256(f"banded-families:{name}:1".encode()).digest()


def _module(name: str) -> Any:
    return importlib.import_module(GENRES[name])


#: Each family's tests as one group. Under ``--dist loadgroup`` a group goes to one
#: worker, so each family's bank is filled once, on the worker that runs its tests, and
#: the families' banks fill in parallel on different workers.
PARAMS = [
    pytest.param(name, marks=pytest.mark.xdist_group(f"receipts-banded-{name}"))
    for name in FAMILIES
]


@pytest.fixture(scope="module", params=PARAMS)
def family(request: pytest.FixtureRequest) -> str:
    return str(request.param)


_BANKS: dict[str, tuple[bank_mod.Bank, bank_mod.Population]] = {}


def _bank(name: str) -> tuple[bank_mod.Bank, bank_mod.Population]:
    """The admitted bank of this family under its fixed key, filled once per process."""
    if name not in _BANKS:
        _BANKS[name] = bank_mod.materialized(load_generator(name), _master(name), BANK_SIZE)
    return _BANKS[name]


def _held(name: str, count: int = 3) -> list[Instance]:
    """The first few admitted instances of the family's bank."""
    return list(_bank(name)[1].instances[:count])


# --------------------------------------------------------------------------
# registration and the rule space
# --------------------------------------------------------------------------


def test_the_family_is_registered_and_declares_what_it_is(family: str) -> None:
    """FAILS IF the name does not reach the module's generator through the registry, if
    the family declares no registered receipt policy or copy profile, or if its rule set
    is smaller than sixty four or is not the whole axis product drawn uniformly."""
    generator = load_generator(family)
    module = _module(family)
    assert generator is module.GENERATOR
    assert generator.name == family
    assert generator.RECEIPT_POLICY == FULL_RECEIPT
    rules = support_of(generator)
    assert len(rules) >= 64
    assert len(rules) == int(np.prod([axis.arity for axis in generator.AXES]))
    assert getattr(generator, "SUPPORT", None) is None
    assert len({tuple(sorted(rule.items())) for rule in rules}) == len(rules)


def test_the_oracle_states_every_rule_and_reads_back(family: str) -> None:
    """FAILS IF some rule's oracle cannot be read back to that rule by the library's own
    parser, or two rules share one statement, which would make the oracle arm a
    statement a reader cannot carry out."""
    generator = load_generator(family)
    stated = set()
    for rule in conventions(generator.AXES):
        cell = generator.render_oracle("t" * 16, MappingProxyType(rule), 24)
        assert generator.parse_oracle(cell) == rule
        stated.add(cell.body)
    assert len(stated) == len(conventions(generator.AXES))


# --------------------------------------------------------------------------
# rule-independent construction and determinism
# --------------------------------------------------------------------------


def test_the_public_task_is_built_without_the_rule(family: str) -> None:
    """FAILS IF `build_table` can be handed a rule, if a table depends on anything but
    the master key, the ordinal and the side, or if the task text or the committed mask
    moves when the whole task is retasked under any other rule.

    Two instances whose draws differ in their rule are built from the same surface
    stream, so their tables and texts have to be byte-identical; that is asked directly
    by building the table twice and by describing every retasked public task.
    """
    generator = load_generator(family)
    parameters = inspect.signature(generator.build_table).parameters
    assert list(parameters) == ["master", "ordinal", "label"]
    master = _master(family)
    instance = draw(generator, master, 0)
    for side in ("A", "B"):
        again = generator.build_table(master, 0, side)
        task = instance.side(side)
        assert generator.table_record(again) == generator.table_record(task.table)
        texts = set()
        for rule in conventions(generator.AXES):
            retasked = Task(
                label=task.label, task_id=task.task_id, surface=task.surface,
                table=task.table, text="", key=generator.key_for(task.table, rule),
                mask=task.mask,
            )
            texts.add(generator.describe(retasked.public()))
        assert texts == {task.text}
        assert task.mask == tuple(range(task.n_rows))
    assert checks.check_invariance(generator, instance).passed
    assert checks.check_lint(generator, instance).passed


def test_a_draw_is_deterministic_and_keyed(family: str) -> None:
    """FAILS IF redrawing one ordinal under one key gives a different instance digest, or
    if two keys give the same instance, which would let two branches of one fork receive
    different bytes or two banks share their tasks."""
    generator = load_generator(family)
    master = _master(family)
    first = draw(generator, master, 1)
    second = draw(generator, master, 1)
    assert bank_mod.instance_digest(first, generator) == bank_mod.instance_digest(
        second, generator
    )
    assert checks.check_fixation(generator, first, master).passed
    other = draw(generator, hashlib.sha256(b"another key").digest(), 1)
    assert bank_mod.instance_digest(other, generator) != bank_mod.instance_digest(
        first, generator
    )


# --------------------------------------------------------------------------
# separability and materiality
# --------------------------------------------------------------------------


def test_every_pair_of_rules_is_separable_on_admitted_tasks(family: str) -> None:
    """FAILS IF two rules give one admitted task the same answer key on either side, or
    if changing any one decision moves fewer than three records on either side.

    The first would leave the perfect reader unable to tell two rules apart from a full
    receipt; the second would make a decision immaterial to held-out answers.
    """
    generator = load_generator(family)
    rules = conventions(generator.AXES)
    pairs = banded_rows.single_axis_pairs(generator.AXES)
    for instance in _held(family):
        for task in (instance.a, instance.b):
            keys = [generator.key_for(task.table, rule) for rule in rules]
            codes = banded_rows.coded(keys, generator.normalize_answer)
            found = banded_rows.separation(codes, pairs, generator.AXES)
            assert found.distinct, (instance.ordinal, task.label)
            assert min(found.fewest.values()) >= 3, (instance.ordinal, task.label, found)
        moved = checks.axis_materiality(generator, instance)
        assert all(low >= 3 for low, _ in moved.values()), moved


def test_copying_a_onto_b_does_not_score(family: str) -> None:
    """FAILS IF filing B with A's answers, positionally or through any registered
    relabel and row move, earns more than half of B on an admitted instance, or if B's
    answers under the drawn rule are identical to A's under any rule."""
    generator = load_generator(family)
    for instance in _held(family):
        scores = checks.copy_scores(generator, instance)
        assert max(scores[name] for name in checks.NO_INDUCTION_MAPS) <= 0.5
        assert scores["identity"] == 0.0


# --------------------------------------------------------------------------
# the placebo
# --------------------------------------------------------------------------


def test_the_placebo_is_byte_identical_across_rules(family: str) -> None:
    """FAILS IF the placebo cell's bytes move with the rule for any registered filing
    class on either side, or if one of its neutral tokens reads as a verdict or as a
    legal answer."""
    generator = load_generator(family)
    instance = _held(family, 1)[0]
    envelope = frozen_envelope(instance.envelope)
    for side in ("a", "b"):
        task = instance.side(side)
        for shape in checks.FILING_CLASSES:
            raw = checks.filing_of(generator, instance, side, shape)
            canonical = generator.parse_and_canonicalize(task, raw)
            seen = set()
            for rule in conventions(generator.AXES):
                retasked = Task(
                    label=task.label, task_id=task.task_id, surface=task.surface,
                    table=task.table, text=task.text,
                    key=generator.key_for(task.table, rule), mask=task.mask,
                )
                cell = generator.render_placebo(retasked.public(), canonical, envelope)
                seen.add(serialize(cell, envelope))
            assert len(seen) == 1, (side, shape)
    assert checks.check_neutral(generator, instance).passed


# --------------------------------------------------------------------------
# the perfect reader
# --------------------------------------------------------------------------


def _graded(generator, instance: Instance, filed: tuple[str, ...], feedback: Feedback):
    task = instance.a
    raw = "\n".join(
        f"{i},{v}" for i, v in zip(generator.row_identifiers(task.table), filed)
    )
    canonical = generator.parse_and_canonicalize(task, raw)
    return generator.render_receipt(task, canonical, task.key, feedback)


def test_the_perfect_reader_of_a_full_receipt_recovers_the_rule(family: str) -> None:
    """FAILS IF the ideal reader of a full receipt leaves more than the drawn rule
    standing or files B imperfectly, whatever A was filed with.

    The receipt is rendered from three filings: the right answers, the answers of a
    rule one decision away, and nothing at all. Every one reveals A's whole key, so the
    reader holds exactly one rule and scores 1.0 on B.
    """
    generator = load_generator(family)
    for instance in _held(family, 2):
        feedback = feedback_for(generator, instance.a, frozen_envelope(instance.envelope))
        near = dict(instance.convention)
        axis = generator.AXES[0]
        near[axis.name] = next(o for o in axis.options if o != near[axis.name])
        for filed in (
            tuple(instance.a.key),
            tuple(generator.key_for(instance.a.table, near)),
            ("",) * instance.a.n_rows,
        ):
            receipt = _graded(generator, instance, filed, feedback)
            reading = banded_rows.perfect_reader(generator, instance, receipt)
            assert reading.consistent == 1
            assert reading.score == 1.0 and reading.expected == 1.0
            assert reading.filing == tuple(instance.b.key)


def test_the_perfect_reader_agrees_with_the_receipt_law(family: str) -> None:
    """FAILS IF the domain's perfect reader and the library's receipt law disagree about
    what an ideal reader of a partial receipt earns.

    Under the two-of-twenty-four law the reader is asked about every mask and every rule
    as the truth. Its realized score averaged over both is exactly the law's ideal level,
    because averaging a majority filing's hits over a uniform posterior is its majority
    share. A handful of masks go through real rendered receipts, so the reading of the
    delivered cell is what is tested there and not a shortcut around it.
    """
    generator = load_generator(family)
    instance = _held(family, 1)[0]
    rules = conventions(generator.AXES)
    a_keys = [generator.key_for(instance.a.table, rule) for rule in rules]
    b_keys = [generator.key_for(instance.b.table, rule) for rule in rules]
    a_codes = banded_rows.coded(a_keys, generator.normalize_answer)
    b_codes = banded_rows.coded(b_keys, generator.normalize_answer)
    fold = generator.normalize_answer

    total = 0.0
    count = 0
    for mask in itertools.combinations(range(instance.a.n_rows), 2):
        columns = list(mask)
        classes: dict[bytes, list[int]] = {}
        for index in range(len(rules)):
            classes.setdefault(a_codes[index, columns].tobytes(), []).append(index)
        for members in classes.values():
            chosen = []
            for row in range(b_codes.shape[1]):
                tally = Counter(int(b_codes[m, row]) for m in members)
                chosen.append(tally.most_common(1)[0][1])
            total += sum(chosen) / float(b_codes.shape[1])
            count += len(members)
    mean_ideal = total / count
    law = law_for(generator, instance, "a", policy=SAMPLED_TWO_OF_TWENTY_FOUR)
    assert abs(law.ideal - mean_ideal) < 1e-9

    # The same number through the real cells, on a few masks and truths.
    envelope = frozen_envelope(instance.envelope)
    masks = [
        receipt_mask(SAMPLED_TWO_OF_TWENTY_FOUR, _master(family), family, n, "A", 24)
        for n in range(3)
    ]
    for mask in masks:
        feedback = Feedback(
            policy=SAMPLED_TWO_OF_TWENTY_FOUR, mask=mask,
            neutral=MappingProxyType(dict(envelope.neutral)),
        )
        for index in (0, len(rules) // 2, len(rules) - 1):
            truth = rules[index]
            task = instance.a
            retasked = Task(
                label=task.label, task_id=task.task_id, surface=task.surface,
                table=task.table, text=task.text, key=a_keys[index], mask=mask,
            )
            truthful = Instance(
                generator=instance.generator, genre=instance.genre,
                ordinal=instance.ordinal, convention=MappingProxyType(truth),
                a=retasked,
                b=Task(
                    label="B", task_id=instance.b.task_id, surface=instance.b.surface,
                    table=instance.b.table, text=instance.b.text, key=b_keys[index],
                    mask=instance.b.mask,
                ),
                envelope=instance.envelope,
            )
            receipt = _graded(generator, truthful, ("",) * 24, feedback)
            evidence = banded_rows.receipt_evidence(receipt)
            assert set(evidence) == set(mask)
            reading = banded_rows.perfect_reader(generator, truthful, receipt)
            members = [
                m for m in range(len(rules))
                if all(fold(a_keys[m][r]) == fold(a_keys[index][r]) for r in mask)
            ]
            assert reading.consistent == len(members)
            hits = sum(
                1 for got, want in zip(reading.filing, b_keys[index]) if fold(got) == fold(want)
            )
            assert reading.score == round(hits / 24.0, 6)


def test_the_perfect_reader_breaks_ties_by_the_published_order() -> None:
    """FAILS IF a tie between answers is broken by anything but the task's published
    answer order, which would make the perfect reader's filing depend on enumeration
    order rather than on a registered rule."""

    class Stub:
        AXES = ()

        def normalize_answer(self, value: str) -> str:
            return value.lower()

        def answer_ranks(self, table: Any) -> tuple[str, ...]:
            return ("Low", "Mid", "High")

        def key_for(self, table: Any, rule: Any) -> tuple[str, ...]:
            return tuple(rule["key"])

    rules = [
        {"key": ("High", "Mid", "Low")},
        {"key": ("Mid", "High", "Low")},
    ]
    filing, share = banded_rows.majority_filing(Stub(), None, rules)
    assert filing == ("Mid", "Mid", "Low")
    assert share == pytest.approx((0.5 + 0.5 + 1.0) / 3)


# --------------------------------------------------------------------------
# admission and the bank
# --------------------------------------------------------------------------


def test_an_admitted_bank_of_twenty_four_builds(family: str) -> None:
    """FAILS IF the family cannot fill a bank of twenty four instances under the
    registered bars and the library's bank law, or if an instance it holds is not one
    admission admits when asked again.

    The population is recomputed by rerunning admission in ordinal order, so filling it
    is the whole of the library's admission and bank law applied to this family.
    """
    generator = load_generator(family)
    bank, held = _bank(family)
    assert len(held.instances) == BANK_SIZE
    assert held.considered <= 2 * BANK_SIZE
    first = held.instances[0]
    made = report(generator, first, bank.master, Thresholds())
    assert made.admitted, made.lines()
    assert made.gates.verdict and made.gates.ceiling == 1.0
