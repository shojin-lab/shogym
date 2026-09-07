"""Attacks the gates and checks have to refuse, run end to end through admission.

Every one of these was a receipt the admission path once accepted. They are kept as
attacks rather than as unit assertions because what matters is not that one function
returns one value: it is that a wrapped, otherwise honest ledger cannot get a dead
cell, a leaking placebo, or a printed rule past the whole sequence.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
from pathlib import Path

import pytest

from shogym.envs.receipts import admission, checks, observe, protocol, streams
from shogym.envs.receipts.generators import ledger
from shogym.envs.receipts.generators.ledger import GENERATOR
from shogym.envs.receipts.registry import load_generator
from shogym.envs.receipts.protocol import Task, conventions
from shogym.envs.receipts.receipt_ast import (
    ReceiptAST,
    ReceiptRow,
    Slot,
    serialize,
)
from shogym.receipts import gate

MASTER = bytes(range(32))
# Diagnostic bars, not the registered ones: loose enough that the fixture instances
# clear them, so an attack test fails on the attack rather than on a threshold.
BARS = admission.Thresholds(max_copy_score=0.6, max_flip_score=0.95, min_leverage=0.05)


@functools.lru_cache(maxsize=1)
def _admitted():
    """One instance a bank would actually hold, cached: filling one is not cheap."""
    from shogym.envs.receipts import bank as bank_mod

    bank = bank_mod.materialize(GENERATOR, MASTER, 1)
    return bank_mod.population(bank, GENERATOR).instances[0]


def _instance(ordinal: int = 0):
    return protocol.draw(GENERATOR, MASTER, ordinal)


class _Wrapped:
    """An otherwise honest ledger with one behaviour replaced."""

    def __init__(self, inner=GENERATOR) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ----- the receipt the gates read has to be the receipt the agent reads -----


class _Ghost(_Wrapped):
    """Registered fields constant; the real verdict in an unregistered slot."""

    def render_receipt(self, task, canonical, truth):
        ast = self._inner.render_receipt(task, canonical, truth)
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
            rows=tuple(
                ReceiptRow(
                    ordinal=r.ordinal, identifier=r.identifier, observed=r.observed,
                    slots=(
                        Slot("verdict", "SAME"), Slot("correction", "SAME"),
                        Slot("ghost", r.slots[0].value + r.slots[1].value),
                    ),
                )
                for r in ast.rows
            ),
        )


class _Duplicated(_Wrapped):
    """Two slots of the same registered name, so a dict would let the last one win."""

    def render_receipt(self, task, canonical, truth):
        ast = self._inner.render_receipt(task, canonical, truth)
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
            rows=tuple(
                ReceiptRow(
                    ordinal=r.ordinal, identifier=r.identifier, observed=r.observed,
                    slots=(Slot("verdict", "PASS"), Slot("verdict", "FAIL"),
                           Slot("correction", "")),
                )
                for r in ast.rows
            ),
        )


class _Truncated(_Wrapped):
    """Two values the registered width truncates to one thing."""

    def render_receipt(self, task, canonical, truth):
        ast = self._inner.render_receipt(task, canonical, truth)
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
            rows=tuple(
                ReceiptRow(
                    ordinal=r.ordinal, identifier=r.identifier, observed=r.observed,
                    slots=(
                        # Both are longer than the registered 4-byte verdict field
                        # and share their first three characters, so both print as
                        # the same truncated token.
                        Slot("verdict", "SAMEONE" if r.slots[0].value == "PASS"
                             else "SAMETWO"),
                        Slot("correction", ""),
                    ),
                )
                for r in ast.rows
            ),
        )


@pytest.mark.parametrize("attack", [_Ghost, _Duplicated], ids=["ghost", "duplicate"])
def test_a_slot_the_bytes_never_carry_is_refused_at_the_serializer(attack) -> None:
    with pytest.raises(ValueError):
        observe.observe(attack(), _instance(), "a")


def test_values_the_width_truncates_to_one_thing_are_one_observation() -> None:
    """The gate reads the printed field, so a difference the bytes lose is not a difference."""
    instance = _instance()
    result = gate(observe.observe(_Truncated(), instance, "a"))
    assert set(result.blocks.values()) == {1}
    assert result.headroom == pytest.approx(0.0, abs=1e-12)
    assert not result.verdict


def test_a_dead_receipt_cannot_be_admitted() -> None:
    """Every convention printing the same bytes has to read as no resolution at all."""
    instance = _instance()
    observation = observe.observe(_Truncated(), instance, "a")
    assert len(set(observation.payloads.values())) == 1
    assert not admission.report(_Truncated(), instance, MASTER, BARS).admitted


# ----- the placebo cannot see the rule -----


def test_the_placebo_renderer_is_handed_no_key() -> None:
    """Not a promise the author makes: a fact about the argument's type."""
    public = _instance().a.public()
    assert not hasattr(public, "key")
    with pytest.raises(AttributeError):
        public.key  # type: ignore[attr-defined]


class _Stashing(_Wrapped):
    """No key on the signature, so it remembers one from the graded call."""

    def __init__(self) -> None:
        super().__init__()
        self._stash = ""

    def render_receipt(self, task, canonical, truth):
        self._stash = "".join(truth)
        return self._inner.render_receipt(task, canonical, truth)

    def render_placebo(self, task, canonical, envelope):
        ast = self._inner.render_placebo(task, canonical, envelope)
        code = "%04d" % (sum(ord(c) for c in self._stash) % 10000)
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
            rows=tuple(
                ReceiptRow(ordinal=r.ordinal, identifier=r.identifier,
                           observed=r.observed,
                           slots=(Slot("verdict", code), r.slots[1]))
                for r in ast.rows
            ),
        )


def test_a_placebo_that_reaches_around_for_the_key_fails_the_envelope() -> None:
    """The check retasks the renderer, so invariance is measured and not assumed."""
    instance = _instance()
    result = checks.check_envelope(_Stashing(), instance)
    assert not result.passed
    assert "retasked" in result.detail
    # and admission catches it too, on whichever of the placebo checks reaches it first
    failed = admission.report(_Stashing(), instance, MASTER, BARS).failed_checks
    assert "envelope" in failed or "placebo" in failed


def test_the_honest_placebo_is_identical_under_every_convention() -> None:
    instance = _instance()
    task = instance.a
    canonical = GENERATOR.parse_and_canonicalize(task, "")
    seen = set()
    for convention in conventions(GENERATOR.AXES):
        retasked = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text=task.text,
            key=tuple(GENERATOR.key_for(task.table, convention)),
        )
        seen.add(
            serialize(
                GENERATOR.render_placebo(retasked.public(), canonical, instance.envelope),
                instance.envelope,
            )
        )
    assert len(seen) == 1


# ----- a slot may print only what its grammar allows -----


class _Coded(_Wrapped):
    """The correction slot prints the option indices as a short numeric code."""

    def render_receipt(self, task, canonical, truth):
        ast = self._inner.render_receipt(task, canonical, truth)
        code = "0000"
        for convention in conventions(GENERATOR.AXES):
            if tuple(GENERATOR.key_for(task.table, convention)) == tuple(truth):
                code = "".join(
                    str(list(a.options).index(convention[a.name])) for a in GENERATOR.AXES
                )
                break
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
            rows=tuple(
                ReceiptRow(ordinal=r.ordinal, identifier=r.identifier,
                           observed=r.observed,
                           slots=(r.slots[0], Slot("correction", code)))
                for r in ast.rows
            ),
        )


def test_a_numeric_rule_code_fails_gate_s_though_it_spells_nothing() -> None:
    instance = _instance()
    report = admission.report(_Coded(), instance, MASTER, BARS)
    assert not report.gates.s_pass
    assert any("grammar does not allow" in leak for leak in report.gates.s_leaks)
    assert not report.admitted


def test_a_slot_that_registers_no_grammar_is_refused() -> None:
    from shogym.envs.receipts.receipt_ast import SlotSpec

    with pytest.raises(ValueError, match="registers no grammar"):
        SlotSpec("anything", 8)


def test_the_honest_ledger_prints_only_what_its_grammar_allows() -> None:
    observation = observe.observe(GENERATOR, _instance(), "a")
    assert observation.slot_realized["verdict"] <= observation.slot_grammar["verdict"]
    assert observation.slot_realized["correction"] <= observation.slot_grammar["correction"]
    assert gate(observation).s_leaks == []


# ----- the placebo's neutral tokens cannot read as a grade -----


def test_no_committed_neutral_token_reads_as_a_verdict() -> None:
    """The alphabet is digits, and the realized tokens are what is actually checked."""
    for ordinal in (0, 1, 18578):
        instance = _instance(ordinal)
        result = checks.check_neutral(GENERATOR, instance)
        assert result.passed, result.detail
        for tokens in instance.envelope.neutral.values():
            for token in tokens:
                assert token.strip() not in ("PASS", "FAIL")


def test_the_filler_alphabet_cannot_spell_a_verdict() -> None:
    assert set(ledger.FILLER_ALPHABET).isdisjoint(set("PASFIL"))


def test_a_neutral_token_that_reads_as_a_grade_is_caught() -> None:
    instance = _instance()
    envelope = instance.envelope
    tainted = type(envelope)(
        size=envelope.size, identifier_width=envelope.identifier_width,
        observed_width=envelope.observed_width, slots=envelope.slots,
        filler=envelope.filler, column_titles=envelope.column_titles,
        neutral={"verdict": ("PASS",) * 24, "correction": envelope.neutral["correction"]},
    )
    spoiled = protocol.Instance(
        generator=instance.generator, genre=instance.genre, ordinal=instance.ordinal,
        convention=instance.convention, a=instance.a, b=instance.b, envelope=tainted,
    )
    result = checks.check_neutral(GENERATOR, spoiled)
    assert not result.passed
    assert "read as a grade" in result.detail


# ----- the copy screen sees the map a reader would actually reach for -----


def test_the_public_band_rank_map_is_enumerated() -> None:
    """Both band tables are printed in the two task texts; mapping first to first
    needs no induction, and the screen has to price it."""
    instance = _instance(2)
    rank = dict(
        zip(instance.a.table.dom["bands"], instance.b.table.dom["bands"])
    )
    values = [rank.get(v, v) for v in instance.a.key][: len(instance.b.key)]
    identifiers = GENERATOR.row_identifiers(instance.b.table)
    raw = "\n".join(f"{i},{v}" for i, v in zip(identifiers, values))
    by_hand = GENERATOR.score(
        instance.b, GENERATOR.parse_and_canonicalize(instance.b, raw)
    )[0]
    assert checks.copy_scores(GENERATOR, instance)["relabel"] >= by_hand


def test_ordinal_two_is_rejected_at_the_bar_that_once_admitted_it() -> None:
    instance = _instance(2)
    assert not checks.check_copy(GENERATOR, instance, 0.35, 0.95, 0.05).passed


def test_the_rank_map_never_beats_the_registered_family() -> None:
    for ordinal in range(16):
        instance = _instance(ordinal)
        rank = dict(zip(instance.a.table.dom["bands"], instance.b.table.dom["bands"]))
        values = [rank.get(v, v) for v in instance.a.key][: len(instance.b.key)]
        identifiers = GENERATOR.row_identifiers(instance.b.table)
        raw = "\n".join(f"{i},{v}" for i, v in zip(identifiers, values))
        by_hand = GENERATOR.score(
            instance.b, GENERATOR.parse_and_canonicalize(instance.b, raw)
        )[0]
        assert checks.copy_scores(GENERATOR, instance)["relabel"] >= by_hand - 1e-12


# ----- only the objective the gate prices may be gated -----


def test_a_scorer_the_gate_does_not_price_is_refused() -> None:
    class Weighted(_Wrapped):
        SCORING = "weighted_rows"

    with pytest.raises(ValueError, match="prices only"):
        observe.observe(Weighted(), _instance(), "a")


# ----- a licensed value chosen for what it encodes is still a channel -----


class _EncodedCorrections(_Wrapped):
    """Corrections spelled as ordinary band names, carrying the option indices."""

    def render_receipt(self, task, canonical, truth):
        ast = self._inner.render_receipt(task, canonical, truth)
        bands = list(task.table.dom["bands"])
        indices = [0, 0, 0, 0]
        for convention in conventions(GENERATOR.AXES):
            if tuple(GENERATOR.key_for(task.table, convention)) == tuple(truth):
                indices = [
                    list(a.options).index(convention[a.name]) for a in GENERATOR.AXES
                ]
                break
        rows = []
        for n, row in enumerate(ast.rows):
            slots = row.slots
            if n < len(indices):
                slots = (row.slots[0], Slot("correction", bands[indices[n]]))
            rows.append(
                ReceiptRow(ordinal=row.ordinal, identifier=row.identifier,
                           observed=row.observed, slots=slots)
            )
        return ReceiptAST(kind=ast.kind, task_id=ast.task_id,
                          row_count=ast.row_count, rows=tuple(rows))


def test_corrections_that_encode_the_rule_are_refused_though_every_value_is_legal() -> None:
    """No unknown slot, no duplicate, no forbidden word, no code outside the grammar.

    Four corrections spelled as ordinary band names hand over every drawn option. The
    gates see legal answers in a legal slot, so the defence cannot be another lint: a
    graded row has to say what the scorer said about that row and nothing else.
    """
    instance = _instance()
    report = admission.report(_EncodedCorrections(), instance, MASTER, BARS)
    assert "graded" in report.failed_checks
    assert not report.admitted
    detail = next(c.detail for c in report.checks if c.name == "graded")
    assert "scorer's own outcomes" in detail


def test_the_honest_ledger_renders_exactly_what_the_scorer_says() -> None:
    result = checks.check_graded(GENERATOR, _instance())
    assert result.passed, result.detail
    assert "the three cells the fork would commit are the ones it commits" in result.detail


# ----- the placebo is checked on what it printed, not on what it was handed -----


class _AllPass(_Wrapped):
    """Every placebo verdict is the literal PASS."""

    def render_placebo(self, task, canonical, envelope):
        ast = self._inner.render_placebo(task, canonical, envelope)
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
            rows=tuple(
                ReceiptRow(ordinal=r.ordinal, identifier=r.identifier,
                           observed=r.observed,
                           slots=(Slot("verdict", "PASS"), r.slots[1]))
                for r in ast.rows
            ),
        )


class _Conditional(_Wrapped):
    """Honest on the perfect filing; a convention code on every other."""

    def __init__(self) -> None:
        super().__init__()
        self._stash = ""

    def render_receipt(self, task, canonical, truth):
        self._stash = "".join(truth)
        return self._inner.render_receipt(task, canonical, truth)

    def render_placebo(self, task, canonical, envelope):
        ast = self._inner.render_placebo(task, canonical, envelope)
        if getattr(canonical, "is_filing", False) and any(canonical.values):
            return ast
        code = "%04d" % (sum(ord(c) for c in self._stash) % 10000)
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
            rows=tuple(
                ReceiptRow(ordinal=r.ordinal, identifier=r.identifier,
                           observed=r.observed,
                           slots=(Slot("verdict", code), r.slots[1]))
                for r in ast.rows
            ),
        )


@pytest.mark.parametrize("attack", [_AllPass, _Conditional], ids=["all-pass", "conditional"])
def test_a_placebo_that_prints_anything_but_its_committed_tokens_is_refused(attack) -> None:
    """One perfect filing is not a sample: the second attack is honest on exactly that
    one and carries the rule on every other."""
    instance = _instance()
    report = admission.report(attack(), instance, MASTER, BARS)
    assert "placebo" in report.failed_checks
    assert not report.admitted


def test_the_placebo_check_covers_every_registered_filing_class() -> None:
    result = checks.check_placebo(GENERATOR, _instance())
    assert result.passed, result.detail
    assert len(checks.FILING_CLASSES) >= 6
    assert "filing classes" in result.detail


# ----- eligibility is validated, not asserted -----


def test_a_bank_cannot_be_filled_by_a_caller_s_predicate() -> None:
    """There is no predicate to pass: filling a bank runs the settled rule or nothing.

    A predicate a caller could supply is a predicate that could admit anything, and
    which rule filled a bank is the whole of what the bank means.
    """
    import inspect

    from shogym.envs.receipts import bank as bank_mod

    taken = set(inspect.signature(bank_mod.materialize).parameters)
    assert taken == {"generator", "master", "size"}
    assert not hasattr(bank_mod, "admit_all")
    assert not hasattr(bank_mod, "build_bank")


def test_the_screen_refuses_evidence_no_run_could_have_produced() -> None:
    from shogym.receipts import Outcomes, screen

    with pytest.raises(ValueError, match="between 0 and 1"):
        Outcomes(placebo=(0.0,), graded=(2.0,), oracle=(1.0,))
    with pytest.raises(ValueError, match="not a bar"):
        screen("f", Outcomes(placebo=(0.0,), graded=(0.0,), oracle=(0.0,)),
               min_room=float("-inf"), min_ratio=0.1, min_pairs=2)
    with pytest.raises(ValueError, match="fewer than two pairs"):
        screen("f", Outcomes(placebo=(0.4,), graded=(0.6,), oracle=(0.9,)),
               min_room=0.1, min_ratio=0.3, min_pairs=1)


def test_a_screen_record_missing_its_shape_is_not_a_screen() -> None:
    """A verdict is not a screen. The rows and the bars are, and both are required."""
    from shogym.receipts import ScreenRecord

    with pytest.raises(ValueError, match="names exactly"):
        ScreenRecord.from_payload({"verdict": True})
    with pytest.raises(ValueError, match="names exactly"):
        ScreenRecord.from_payload({**_screen_payload(), "verdict": True})


# ----- the frozen code hash is enforced where serving happens -----


def test_verification_computes_the_code_hash_itself(bundle_room) -> None:
    """A caller that could supply the value could supply the one that matches."""
    import inspect

    from shogym.envs.receipts import bundle as bundle_mod

    taken = set(inspect.signature(bundle_mod.verify).parameters)
    assert taken == {"bundle", "generator"}
    root = _tamper(bundle_room, bundle_mod.CODE, {"digest": "0" * 64})
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert any("code pin does not hold" in p for p in checked.problems)


# ----- one fork, keyed by the filing that produced it -----


def test_a_fork_filed_under_another_filing_s_name_is_refused(tmp_path) -> None:
    import shutil

    from shogym.envs.receipts import bank as bank_mod

    instance = _admitted()
    one = checks.filing_of(GENERATOR, instance, "a", "canonical")
    two = checks.filing_of(GENERATOR, instance, "a", "none")
    held = "b" * 64
    bank_mod.fork_for(GENERATOR, instance, "a", one, tmp_path, held)
    shutil.copy(
        bank_mod.fork_path(
            tmp_path, instance.a.task_id, bank_mod.filing_digest(one), held
        ),
        bank_mod.fork_path(
            tmp_path, instance.a.task_id, bank_mod.filing_digest(two), held
        ),
    )
    with pytest.raises(ValueError, match="is for filing"):
        bank_mod.fork_for(GENERATOR, instance, "a", two, tmp_path, held)


def test_a_fork_must_name_the_source_it_belongs_to(tmp_path) -> None:
    """Two frozen bundles can hold the same task and the same filing, so a record that
    named neither would answer for both."""
    from shogym.envs.receipts import bank as bank_mod

    instance = _admitted()
    raw = checks.filing_of(GENERATOR, instance, "a", "canonical")
    fork = bank_mod.render_fork(GENERATOR, instance, "a", raw)
    with pytest.raises(ValueError, match="name the frozen source"):
        bank_mod.save_fork(fork, tmp_path, "")
    bank_mod.save_fork(fork, tmp_path, "b" * 64)
    with pytest.raises(ValueError, match="needs the source"):
        bank_mod.load_fork(tmp_path, instance.a.task_id, fork.filing_digest, "")
    assert (
        bank_mod.load_fork(tmp_path, instance.a.task_id, fork.filing_digest, "c" * 64)
        is None
    )


def test_a_fork_is_rendered_once_and_replayed_after(tmp_path) -> None:
    from shogym.envs.receipts import bank as bank_mod

    instance = _admitted()
    raw = checks.filing_of(GENERATOR, instance, "a", "canonical")
    held = "d" * 64
    first = bank_mod.fork_for(GENERATOR, instance, "a", raw, tmp_path, held)
    written = bank_mod.fork_path(
        tmp_path, instance.a.task_id, bank_mod.filing_digest(raw), held
    )
    stamp = written.stat().st_mtime_ns
    second = bank_mod.fork_for(GENERATOR, instance, "a", raw, tmp_path, held)
    # Both calls return the COMMITTED bytes, including the one that rendered them:
    # a caller that kept what it rendered while a concurrent winner wrote something
    # else would be the one branch reading bytes no other branch will see.
    assert first.replayed and second.replayed
    assert written.stat().st_mtime_ns == stamp
    assert first.graded == second.graded
    assert first.placebo == second.placebo
    assert first.oracle == second.oracle
    assert first.digests == second.digests


# ----- admission evidence is verified, not read off a label -----


def test_there_is_no_field_that_asserts_admission() -> None:
    """A label, a verdict and a records digest were three fields whoever wrote the file
    chose. None of them exists now: what a bundle holds is raw material, and what it is
    worth is recomputed from that material every time it is opened."""
    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts import bundle as bundle_mod

    fields = set(bank_mod.bank_record(bank_mod.Bank(
        generator="g", genre="j", renderer=bank_mod.RENDERER_CONFIGURATION,
        master=MASTER, size=1,
    )))
    assert fields == {"generator", "genre", "renderer", "master", "size"}
    for gone in ("admission", "considered", "thresholds", "screen", "review",
                 "module_digest", "records", "envelope_size", "axes"):
        assert gone not in fields
    for gone in ("records_digest", "gated", "screened", "reviewed", "claims_complete",
                 "missing_claims", "passing_fraction"):
        assert not hasattr(bank_mod.Bank, gone), gone
    assert not hasattr(bank_mod, "verify_admission")
    assert bundle_mod.verify is not None


def test_the_production_environment_has_no_override() -> None:
    """A flag that turned the refusal off would make the development name decorative."""
    import inspect

    from shogym.envs.receipts.env_v1 import ReceiptsDevEnv, ReceiptsV1Env

    assert "allow_incomplete" not in inspect.signature(ReceiptsV1Env.__init__).parameters
    assert "allow_incomplete" not in inspect.signature(ReceiptsDevEnv.__init__).parameters


# ----- the sealed filing is where semantics are enforced -----


class _SampleHonest(_Wrapped):
    """Honest on every filing admission samples, and only on those."""

    def render_receipt(self, task, canonical, truth):
        ast = self._inner.render_receipt(task, canonical, truth)
        # Honest on every filing admission samples, on BOTH sides and in every
        # registered class, since admission judges the whole set. Compared as parsed
        # values rather than as filing text: a partial filing canonicalizes to one
        # value per printed row, so comparing reconstructed text would make this
        # dishonest on filings admission does sample, which is a different attack.
        # What is left is the rest of the parser's legal space.
        sampled = {
            tuple(getattr(
                GENERATOR.parse_and_canonicalize(
                    self.instance.side(side),
                    checks.filing_of(GENERATOR, self.instance, side, shape),
                ),
                "values",
                (),
            ))
            for side in ("a", "b")
            for shape in checks.FILING_CLASSES
        }
        values = tuple(getattr(canonical, "values", ()))
        if not values or values in sampled:
            return ast
        options = []
        for convention in conventions(GENERATOR.AXES):
            if tuple(GENERATOR.key_for(task.table, convention)) == tuple(truth):
                options = [convention[a.name] for a in GENERATOR.AXES]
                break
        rows = [
            ReceiptRow(
                ordinal=r.ordinal, identifier=r.identifier, observed=r.observed,
                slots=(r.slots[0], Slot("correction",
                                        options[n] if n < len(options) else r.slots[1].value)),
            )
            for n, r in enumerate(ast.rows)
        ]
        return ReceiptAST(kind=ast.kind, task_id=ast.task_id,
                          row_count=ast.row_count, rows=tuple(rows))


def test_a_renderer_honest_only_on_the_samples_is_caught_when_the_filing_seals() -> None:
    """The parser's legal filing space is open-ended, so sampling cannot close this.

    What a fork commits is checked against that filing's own outcomes at the moment
    it is committed, and a fork that fails is never serialized or persisted.
    """
    from shogym.envs.receipts import bank as bank_mod

    instance = _instance()
    attack = _SampleHonest()
    attack.instance = instance
    assert admission.report(attack, instance, MASTER, BARS).admitted
    identifiers = GENERATOR.row_identifiers(instance.a.table)
    unsampled = f"{identifiers[0]},Standard"
    with pytest.raises(ValueError, match="not what the scorer's own outcomes say"):
        bank_mod.render_fork(attack, instance, "a", unsampled)


# ----- the oracle states the rule that was drawn -----


class _RotatedOracle(_Wrapped):
    """States a rule one anchor option away from the one drawn."""

    def render_oracle(self, task_id, convention, row_count=0):
        options = list(GENERATOR.AXES[0].options)
        moved = dict(convention)
        moved["anchor"] = options[
            (options.index(convention["anchor"]) + 1) % len(options)
        ]
        return self._inner.render_oracle(task_id, moved, row_count)


class _EmptyOracle(_Wrapped):
    def render_oracle(self, task_id, convention, row_count=0):
        return ReceiptAST(kind="oracle", task_id=task_id, row_count=row_count, body=())


@pytest.mark.parametrize("attack", [_RotatedOracle, _EmptyOracle], ids=["rotated", "empty"])
def test_an_oracle_that_states_the_wrong_rule_is_refused(attack) -> None:
    """The oracle arm is the denominator of the room the screen measures, so a false
    one moves the measurement rather than merely failing to teach."""
    instance = _instance()
    report = admission.report(attack(), instance, MASTER, BARS)
    assert "oracle" in report.failed_checks
    assert not report.admitted


def test_the_honest_oracle_reads_back_over_the_whole_support() -> None:
    result = checks.check_oracle(GENERATOR, _instance())
    assert result.passed, result.detail
    assert "all 72 conventions" in result.detail


def test_a_fork_whose_oracle_states_another_rule_is_refused() -> None:
    from shogym.envs.receipts import bank as bank_mod

    instance = _instance()
    raw = checks.filing_of(GENERATOR, instance, "a", "canonical")
    with pytest.raises(ValueError, match="not the one drawn"):
        bank_mod.render_fork(_RotatedOracle(), instance, "a", raw)


# ----- the instance the bank commits to is the whole instance -----


class _HiddenSerial(_Wrapped):
    """Builds a table whose rows carry a serial that changes on every rebuild.

    Everything the commitment used to record stays equal: task ids come from the
    HMAC, surfaces from the ordinal, texts from the same body, keys from the same
    dates. Only the row identifiers move, which is what the agent reads and what a
    receipt is aligned on.
    """

    def __init__(self) -> None:
        super().__init__()
        self._built = 0

    def build_table(self, master, ordinal, label):
        table = self._inner.build_table(master, ordinal, label)
        self._built += 1
        serial = self._built
        rows = tuple(
            ledger.LedgerRow(row_id=f"{row.row_id}-{serial}", dates=row.dates)
            for row in table.rows
        )
        return ledger.LedgerTable(
            domain=table.domain, rows=rows, holidays=table.holidays, body=table.body
        )


def test_a_table_that_does_not_rebuild_fails_fixation() -> None:
    """The commitment is the WHOLE instance, tables included.

    A table is opaque to this package, so a commitment that recorded only the fields
    it could read left the rows out, and two rebuilds could hand two branches of one
    fork different identifiers, different receipts and different bytes while the
    digest fixation compares stayed equal. The family says what its table is and the
    bank hashes what it said.
    """
    from shogym.envs.receipts import bank as bank_mod

    generator = _HiddenSerial()
    one = protocol.draw(generator, MASTER, 0)
    two = protocol.draw(generator, MASTER, 0)
    identifiers_moved = generator.row_identifiers(one.a.table) != generator.row_identifiers(
        two.a.table
    )
    assert identifiers_moved
    # Every field the record used to carry still agrees.
    for side in ("a", "b"):
        first, second = getattr(one, side), getattr(two, side)
        assert (first.task_id, first.surface, first.text, first.key) == (
            second.task_id, second.surface, second.text, second.key
        )
    assert bank_mod.instance_digest(one, generator) != bank_mod.instance_digest(
        two, generator
    )
    result = checks.check_fixation(generator, one, MASTER)
    assert not result.passed
    assert "does not rebuild identically" in result.detail
    report = admission.report(generator, one, MASTER, BARS)
    assert "fixation" in report.failed_checks and not report.admitted


# ----- one convention builds both siblings -----


class _MutatesTheConvention(_Wrapped):
    """Computes A's key, then edits the convention before B's is computed."""

    def __init__(self) -> None:
        super().__init__()
        self._seen = 0

    def key_for(self, table, convention):
        key = self._inner.key_for(table, convention)
        self._seen += 1
        if self._seen == 1:
            options = list(self._inner.AXES[0].options)
            here = options.index(convention[self._inner.AXES[0].name])
            convention[self._inner.AXES[0].name] = options[(here + 1) % len(options)]
        return key


def test_a_generator_cannot_edit_the_convention_between_a_and_b() -> None:
    """The family relation holds by construction, so nothing may build it twice.

    One convention scoring both sides is the whole of the relation. A plain dictionary
    passed to A's callback and then to B's is one either callback can edit in between,
    and the pair that comes out is not a family: each side is internally consistent, so
    every named check passes on it. The draw is frozen before either side is built, so
    the edit raises where the mistake is.
    """
    with pytest.raises(TypeError):
        protocol.draw(_MutatesTheConvention(), MASTER, 1)
    # And the stored convention is not something a later caller can edit either.
    instance = protocol.draw(GENERATOR, MASTER, 1)
    with pytest.raises(TypeError):
        instance.convention["anchor"] = "event"  # type: ignore[index]


# ----- the declared support is what is drawn -----


def test_the_correlated_exhibit_draws_only_inside_its_declared_support() -> None:
    """Gating one distribution and materializing another would demonstrate nothing."""
    from shogym.envs.receipts.protocol import support_of

    generator = load_generator("affine")
    reachable = {tuple(sorted(c.items())) for c in support_of(generator)}
    assert len(reachable) == 9
    for ordinal in range(24):
        drawn = protocol.draw(generator, MASTER, ordinal).convention
        assert tuple(sorted(drawn.items())) in reachable


def test_an_ordinary_family_still_draws_the_whole_product() -> None:
    assert not hasattr(GENERATOR, "SUPPORT")
    drawn = {
        tuple(sorted(protocol.draw(GENERATOR, MASTER, o).convention.items()))
        for o in range(400)
    }
    assert len(drawn) > 40


# ----- the generator module is untrusted, and so is what it hands back -----


def _unsampled(instance) -> str:
    """A parser-valid filing that is not one of admission's registered classes."""
    return f"{GENERATOR.row_identifiers(instance.a.table)[0]},Standard"


def _option_indices(truth, table) -> list[int]:
    for convention in conventions(GENERATOR.AXES):
        if tuple(GENERATOR.key_for(table, convention)) == tuple(truth):
            return [list(a.options).index(convention[a.name]) for a in GENERATOR.AXES]
    return [0, 0, 0, 0]


class _RewritesCommitment(_Wrapped):
    """Rewrites the committed neutral tokens before the comparison is built from them."""

    def __init__(self) -> None:
        super().__init__()
        self._code = 0

    def render_receipt(self, task, canonical, truth):
        self._code = sum(_option_indices(truth, task.table))
        return self._inner.render_receipt(task, canonical, truth)

    def render_placebo(self, task, canonical, envelope):
        envelope.neutral["verdict"] = ("%04d" % self._code,) * 24
        return self._inner.render_placebo(task, canonical, envelope)


def test_a_renderer_cannot_rewrite_the_placebo_commitment() -> None:
    """The committed tokens are the placebo arm's whole meaning, so what the renderer
    is handed is a read-only view and the comparison is built before it is called."""
    from shogym.envs.receipts import bank as bank_mod

    instance = _instance()
    with pytest.raises(TypeError, match="does not support item assignment"):
        bank_mod.render_fork(_RewritesCommitment(), instance, "a", _unsampled(instance))
    assert not admission.report(_RewritesCommitment(), instance, MASTER, BARS).admitted


def test_the_frozen_envelope_is_a_deep_copy() -> None:
    from shogym.envs.receipts.receipt_ast import frozen_envelope

    envelope = _instance().envelope
    frozen = frozen_envelope(envelope)
    with pytest.raises(TypeError):
        frozen.neutral["verdict"] = ("x",)  # type: ignore[index]
    assert frozen.neutral["verdict"] == tuple(envelope.neutral["verdict"])


class _BendsOrdinals(_Wrapped):
    """Writes the convention into the printed row ordinal, in both cells."""

    def __init__(self) -> None:
        super().__init__()
        self._indices = [0, 0, 0, 0]

    def _bend(self, ast):
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
            rows=tuple(
                ReceiptRow(
                    ordinal=9000 + self._indices[n] if n < 4 else row.ordinal,
                    identifier=row.identifier, observed=row.observed, slots=row.slots,
                )
                for n, row in enumerate(ast.rows)
            ),
        )

    def render_receipt(self, task, canonical, truth):
        self._indices = _option_indices(truth, task.table)
        return self._bend(self._inner.render_receipt(task, canonical, truth))

    def render_placebo(self, task, canonical, envelope):
        return self._bend(self._inner.render_placebo(task, canonical, envelope))


def test_the_printed_row_ordinal_is_compared_like_any_other_field() -> None:
    """A visible wrapper column carried the convention while every registered slot
    stayed honest, so the comparison is exact row equality rather than a list of
    fields someone thought to name."""
    from shogym.envs.receipts import bank as bank_mod

    instance = _instance()
    with pytest.raises(ValueError, match="ordinal"):
        bank_mod.render_fork(_BendsOrdinals(), instance, "a", _unsampled(instance))


class _MutatesConvention(_Wrapped):
    """Rotates the convention in place before rendering the oracle from it."""

    def render_oracle(self, task_id, convention, row_count=0):
        options = list(GENERATOR.AXES[0].options)
        convention["anchor"] = options[
            (options.index(convention["anchor"]) + 1) % len(options)
        ]
        return self._inner.render_oracle(task_id, convention, row_count)


def test_a_renderer_cannot_move_the_oracle_comparison_target() -> None:
    """The oracle child would otherwise be taught a rule different from the one that
    scored the task, and the room denominator would measure the wrong intervention."""
    from shogym.envs.receipts import bank as bank_mod

    instance = _instance()
    before = dict(instance.convention)
    with pytest.raises(TypeError, match="does not support item assignment"):
        bank_mod.render_fork(
            _MutatesConvention(), instance, "a",
            checks.filing_of(GENERATOR, instance, "a", "canonical"),
        )
    assert dict(instance.convention) == before
    assert not admission.report(_MutatesConvention(), instance, MASTER, BARS).admitted


class _Raises(_Wrapped):
    def render_placebo(self, task, canonical, envelope):
        raise RuntimeError("no")


def test_a_generator_that_raises_has_not_passed_the_check() -> None:
    """An exception escaping admission would stop the report rather than record a
    refusal, which is the wrong shape of outcome for something a bank decides about."""
    report = admission.report(_Raises(), _instance(), MASTER, BARS)
    assert not report.admitted
    assert "placebo" in report.failed_checks


# ----- the oracle is read by this package, not by the family that wrote it -----


class _CollusiveOracle(_Wrapped):
    """Renders one fixed false sentence and reads back whatever it was handed."""

    def __init__(self) -> None:
        super().__init__()
        self._stash: dict = {}

    def render_oracle(self, task_id, convention, row_count=0):
        self._stash = dict(convention)
        return ReceiptAST(
            kind="oracle", task_id=task_id, row_count=row_count,
            body=("Always use the first date and count every day.",),
        )

    def parse_oracle(self, ast):
        return dict(self._stash)


def test_a_family_cannot_certify_its_own_oracle() -> None:
    """A renderer and a reader supplied together have no fixed point outside
    themselves, so a round trip through both is not evidence about either."""
    from shogym.envs.receipts import bank as bank_mod

    instance = _instance()
    report = admission.report(_CollusiveOracle(), instance, MASTER, BARS)
    assert "oracle" in report.failed_checks
    assert not report.admitted
    with pytest.raises(ValueError, match="oracle cannot be read"):
        bank_mod.render_fork(
            _CollusiveOracle(), instance, "a",
            checks.filing_of(GENERATOR, instance, "a", "canonical"),
        )


class _CoachingOracle(_Wrapped):
    """States the drawn rule, and then says a little more than the rule."""

    def render_oracle(self, task_id, convention, row_count=0):
        base = self._inner.render_oracle(task_id, convention, row_count)
        return ReceiptAST(
            kind=base.kind, task_id="f" * 16, row_count=9999,
            body=tuple(base.body) + ("  Tip: work the undated records first.",),
        )


def test_an_oracle_that_states_the_rule_and_coaches_is_refused() -> None:
    """Parsing one option per axis back out leaves everything else free.

    The wrapper, the record count and the rest of the body all ride into the arm that
    is the denominator of the room the whole screen measures, so the oracle has to BE
    the cell the registered template renders and not merely say the same rule.
    """
    from shogym.envs.receipts import bank as bank_mod

    instance = _instance()
    generator = _CoachingOracle()
    result = checks.check_oracle(generator, instance)
    assert not result.passed
    assert "not the cell the registered template renders" in result.detail
    report = admission.report(generator, instance, MASTER, BARS)
    assert "oracle" in report.failed_checks and not report.admitted
    with pytest.raises(ValueError, match="not the cell the registered template renders"):
        bank_mod.render_fork(
            generator, instance, "a",
            checks.filing_of(GENERATOR, instance, "a", "canonical"),
        )


def test_a_family_with_no_declared_oracle_table_cannot_have_one_read() -> None:
    class Undeclared(_Wrapped):
        ORACLE = None

    result = checks.check_oracle(Undeclared(), _instance())
    assert not result.passed
    assert "declares no oracle phrase table" in result.detail


def test_the_oracle_table_refuses_a_phrase_contained_in_another() -> None:
    from shogym.envs.receipts.oracle import OracleTemplate

    with pytest.raises(ValueError, match="contained in another"):
        OracleTemplate(
            head=(), sentences={"k": "it is {}."},
            phrases={"k": {"a": "the first date", "b": "the first date of the month"}},
        )


# ----- the screen's sample and the review pack's contents -----


def test_a_screen_states_the_sample_it_was_required_to_have() -> None:
    from shogym.receipts import Outcomes, screen

    with pytest.raises(ValueError, match="fewer than two pairs"):
        screen("f", Outcomes(placebo=(0.4,), graded=(0.6,), oracle=(0.9,)),
               min_room=0.1, min_ratio=0.3, min_pairs=1)
    thin = screen("f", Outcomes(placebo=(0.4,), graded=(0.6,), oracle=(0.9,)),
                  min_room=0.1, min_ratio=0.3, min_pairs=8)
    assert not thin.verdict


def test_a_review_pack_has_to_be_a_manifest(bundle_room) -> None:
    """A file is not a review, and neither is a manifest with nothing in it."""
    from shogym.envs.receipts import bundle as bundle_mod

    root = _tamper(bundle_room, bundle_mod.REVIEW, {"reviewer": "x"})
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert any("names exactly" in problem for problem in checked.problems)


# ----- the pin covers the code that decides what a run means -----


def test_the_pin_is_enumerated_from_the_import_graph() -> None:
    """A pin that has to be remembered is a pin somebody forgets.

    The closure is walked from the roots, so a module that starts deciding something
    joins the pin by being imported rather than by being added to a list. What stays
    written down is the short set that decides nothing, each with its reason, and this
    walks the graph independently to check that nothing else escaped.
    """
    from shogym.envs.receipts import bank as bank_mod

    pinned = set(bank_mod.pinned_modules(GENERATOR))
    reachable: set[str] = set()
    stack = [type(GENERATOR).__module__, *bank_mod.PIN_ROOTS]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(bank_mod._imported(current) - reachable)
    escaped = reachable - pinned - set(bank_mod.NOT_DECIDING)
    assert not escaped, f"these decide something and are not pinned: {sorted(escaped)}"
    for name, reason in bank_mod.NOT_DECIDING.items():
        assert bank_mod._is_module(name), f"{name} is declared out of the pin and is gone"
        assert reason
    # the modules the redesign added to the decision path are in it
    for name in (
        "shogym.envs.receipts.bundle",
        "shogym.envs.receipts.review",
        "shogym.envs.receipts.streams",
        "shogym.envs.receipts.env_v1",
        "shogym.receipts.screen",
        "shogym.receipts.resolution",
    ):
        assert name in pinned


def test_every_pinned_module_moves_the_pin_when_it_drifts() -> None:
    import importlib.util
    from pathlib import Path as _Path

    from shogym.envs.receipts import bank as bank_mod

    base = bank_mod.current_code_digest(GENERATOR)
    for name in bank_mod.pinned_modules(GENERATOR):
        spec = importlib.util.find_spec(name)
        assert spec and spec.origin
        path = _Path(spec.origin)
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# drift\n")
            assert bank_mod.current_code_digest(GENERATOR) != base, name
        finally:
            path.write_bytes(original)
    assert bank_mod.current_code_digest(GENERATOR) == base


# ----- eligibility means one thing -----


def test_eligibility_is_one_operation_over_one_bundle() -> None:
    """Two eligibility tests composed beside each other is one of them being wrong.

    Production, the roster and development all reach the same function, and it takes
    a bundle and a generator: there is no claim to read first and no second answer to
    reconcile with the first.
    """
    import inspect

    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.envs.receipts.env_v1 import _open_bundle

    assert set(inspect.signature(bundle_mod.verify).parameters) == {
        "bundle", "generator"
    }
    source = inspect.getsource(_open_bundle)
    assert "bundle_mod.verify(" in source
    assert "missing_claims" not in source
    assert "claims_complete" not in source


# ----- the empirical and human admission stages -----


def _screen_payload(
    model: str = "a scripted policy", pairs: int = 40, family: str = "ledger", **changes
) -> dict:
    """A screen artifact: the rows, the family they were taken on, and the bars."""
    payload = {
        "family": family,
        "model": model,
        "task_seeds": [str(i) for i in range(pairs)],
        "pairs": [
            {"instance": f"task-{i:02d}", "filing": f"filing-{i:02d}",
             "placebo": 0.4, "graded": 0.6, "oracle": 0.9}
            for i in range(pairs)
        ],
        "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36,
        "floor": 0.0, "floor_rule": "drop",
        "candidates_screened": 1, "selection_note": "",
    }
    payload.update(changes)
    return payload


def _run_payload(model: str = "a scripted policy", pairs: int = 40) -> dict:
    """Just the run half, for the reader that only takes rows."""
    whole = _screen_payload(model, pairs)
    return {name: whole[name] for name in ("family", "model", "task_seeds", "pairs")}


def _pack(
    room,
    coverage,
    envelope_size: int,
    reviewer: str = "andrew",
    family: str = "ledger",
    bank_identity: str = "",
) -> Path:
    """A pack covering what the family declares, with plausibly sized artifacts."""
    folder = room / "renders"
    folder.mkdir(exist_ok=True)
    renders = []
    for index, (category, key) in enumerate(coverage.required):
        kind = "task" if category == "surface" else "cell"
        floor = 400 if kind == "task" else envelope_size
        artifact = folder / f"{index:03d}.txt"
        artifact.write_text("R" * (floor + 8), encoding="utf-8")
        renders.append({
            "category": category, "key": key, "kind": kind,
            "path": f"renders/{artifact.name}",
        })
    pack = room / "pack.json"
    pack.write_text(
        json.dumps({"reviewer": reviewer, "checklist": ["options", "filings"],
                    "seeds": [0], "family": family, "bank": bank_identity,
                    "renders": renders}),
        encoding="utf-8",
    )
    return pack


def _materials(room, generator=None, master=MASTER, size: int = 1):
    """A bank, a screen artifact and a review pack: the three things a bundle holds."""
    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts.review import required_coverage

    generator = generator or GENERATOR
    bank, held = bank_mod.materialized(generator, master, size)
    screen = room / "screen.json"
    screen.write_text(
        json.dumps(_screen_payload(family=generator.name)), encoding="utf-8"
    )
    counts = [i.a.n_rows for i in held.instances] + [i.b.n_rows for i in held.instances]
    coverage = required_coverage(generator, checks.FILING_CLASSES, counts)
    pack = _pack(
        room,
        coverage,
        min(i.envelope.size for i in held.instances),
        family=generator.name,
        bank_identity=bank_mod.bank_identity(bank),
    )
    return bank, screen, pack, held


@pytest.fixture(scope="session")
def built_bundle(tmp_path_factory):
    """One honest bundle, built once. Every attack works on a copy of it."""
    from shogym.envs.receipts import bundle as bundle_mod

    room = tmp_path_factory.mktemp("bundle-source")
    bank, screen, pack, _ = _materials(room)
    return bundle_mod.build(room / "bundles", GENERATOR, bank, screen, pack)


@pytest.fixture()
def bundle_room(built_bundle, tmp_path):
    """A private copy of that bundle, for a test that is about to damage it."""
    room = tmp_path / "bundles"
    room.mkdir()
    shutil.copytree(built_bundle.root, room / built_bundle.root.name)
    return room / built_bundle.root.name


def _reseal(root: Path) -> Path:
    """Rewrite the manifest for whatever the directory now holds, and re-address it.

    This is the attacker who repairs what they broke. It works, and it is supposed to:
    what it cannot do is produce the SAME address, so nothing that opens a bundle by
    digest is reachable this way, and the repaired bundle still has to verify.
    """
    from shogym.envs.receipts import bundle as bundle_mod

    entries = [
        {"path": name, "size": (root / name).stat().st_size,
         "digest": streams.file_digest(str(root / name))}
        for name in sorted(
            item.relative_to(root).as_posix()
            for item in root.rglob("*")
            if item.is_file() and item.name != bundle_mod.MANIFEST
        )
    ]
    text = bundle_mod.canonical_manifest(entries)
    (root / bundle_mod.MANIFEST).write_text(text, encoding="utf-8")
    moved = root.parent / streams.digest(text.encode())
    root.rename(moved)
    return moved


def _tamper(root: Path, name: str, payload) -> Path:
    """Edit one file inside a bundle and reseal, which changes its address.

    Written in canonical form, because a bundle refuses a file that is not in it and
    every one of these is probing what happens AFTER the file is read.
    """
    from shogym.envs.receipts import bundle as bundle_mod

    (root / name).write_text(bundle_mod.canonical_json(payload), encoding="utf-8")
    return _reseal(root)


def _read(root: Path, name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))


# ----- the bundle is one address over everything admission rests on -----


def test_a_verified_bundle_holds_every_named_file_and_opens(built_bundle) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    for name in bundle_mod.CONTENTS:
        assert name in built_bundle.files
    assert any(f.startswith("renders/") for f in built_bundle.files)
    assert built_bundle.root.name == built_bundle.digest
    checked = bundle_mod.verify(built_bundle, GENERATOR)
    assert checked.problems == ()
    assert len(checked.instances) == 1
    assert checked.considered >= 1


def test_a_bundle_names_nothing_outside_itself(built_bundle) -> None:
    """A path leading out of the bundle is evidence that could be replaced without
    changing the bundle."""
    from shogym.envs.receipts import bundle as bundle_mod

    for name in built_bundle.files:
        assert not name.startswith("/") and ".." not in name.split("/")
    text = (built_bundle.root / bundle_mod.REVIEW).read_text(encoding="utf-8")
    assert str(built_bundle.root) not in text
    for entry in _read(built_bundle.root, bundle_mod.REVIEW)["renders"]:
        assert entry["path"] in built_bundle.files


def test_an_edited_file_no_longer_matches_the_manifest(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    payload = _read(bundle_room, bundle_mod.BANK)
    (bundle_room / bundle_mod.BANK).write_text(
        json.dumps({**payload, "size": 9}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not the file the manifest hashed|bytes and the manifest says"):
        bundle_mod.load(bundle_room)


def test_a_repaired_manifest_is_a_different_address(bundle_room) -> None:
    """Fixing the manifest to match the edit gives the bundle a new name, and the name
    is what production asks for."""
    from shogym.envs.receipts import bundle as bundle_mod

    before = bundle_room.name
    payload = _read(bundle_room, bundle_mod.BANK)
    (bundle_room / bundle_mod.BANK).write_text(
        json.dumps({**payload, "size": 9}), encoding="utf-8"
    )
    entries = [
        {"path": name, "size": (bundle_room / name).stat().st_size,
         "digest": streams.file_digest(str(bundle_room / name))}
        for name in sorted(
            item.relative_to(bundle_room).as_posix()
            for item in bundle_room.rglob("*")
            if item.is_file() and item.name != bundle_mod.MANIFEST
        )
    ]
    (bundle_room / bundle_mod.MANIFEST).write_text(
        bundle_mod.canonical_manifest(entries), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="addressed by its own contents"):
        bundle_mod.load(bundle_room)
    moved = bundle_room.rename(
        bundle_room.parent / streams.digest(bundle_mod.canonical_manifest(entries).encode())
    )
    assert moved.name != before
    # It loads under its new name, and then has to survive verification like any other.
    assert bundle_mod.load(moved).digest == moved.name
    assert bundle_mod.verify_at(moved, GENERATOR).problems


def test_an_unlisted_file_is_not_part_of_a_bundle(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    (bundle_room / "renders" / "extra.txt").write_text("R" * 800, encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted"):
        bundle_mod.load(bundle_room)


def test_a_missing_file_is_not_a_bundle(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    (bundle_room / bundle_mod.SCREEN).unlink()
    with pytest.raises(ValueError, match="missing"):
        bundle_mod.load(bundle_room)


def test_a_render_that_is_a_link_out_of_the_bundle_is_refused(bundle_room, tmp_path) -> None:
    """A link is a name for somebody else's bytes, and they can change afterwards."""
    from shogym.envs.receipts import bundle as bundle_mod

    outside = tmp_path / "outside.txt"
    outside.write_text("R" * 4000, encoding="utf-8")
    target = next(iter((bundle_room / "renders").iterdir()))
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="is a link"):
        bundle_mod.load(bundle_room)


def test_a_manifest_that_is_not_canonical_is_refused(bundle_room) -> None:
    """Whitespace and ordering are not free: the manifest's bytes are the address."""
    from shogym.envs.receipts import bundle as bundle_mod

    path = bundle_room / bundle_mod.MANIFEST
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical form"):
        bundle_mod.load(bundle_room)


# ----- the population is recomputed, so it cannot be chosen -----


def test_a_duplicated_instance_cannot_survive_recomputation(bundle_room) -> None:
    """A persistence merge that doubles an entry, or a hand-written second one: the
    recomputed population holds each ordinal once, so the sequences differ."""
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.INSTANCES)
    root = _tamper(bundle_room, bundle_mod.INSTANCES, stored + [dict(stored[0])])
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert any("its bank admits" in problem for problem in checked.problems)


def test_an_instance_this_bank_does_not_admit_cannot_be_listed(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.INSTANCES)
    root = _tamper(
        bundle_room, bundle_mod.INSTANCES,
        [{**stored[0], "ordinal": stored[0]["ordinal"] + 1}],
    )
    assert any(
        "its bank admits" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_an_instance_digest_that_is_not_the_rebuilt_one_is_caught(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.INSTANCES)
    root = _tamper(
        bundle_room, bundle_mod.INSTANCES, [{**stored[0], "digest": "0" * 64}]
    )
    assert any(
        "not the ones its bank rebuilds to" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_a_commitment_to_another_convention_is_caught(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.INSTANCES)
    root = _tamper(
        bundle_room, bundle_mod.INSTANCES, [{**stored[0], "commitment": "0" * 64}]
    )
    assert any(
        "not the ones its bank rebuilds to" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_the_passing_fraction_is_computed_and_never_stored(built_bundle) -> None:
    """It was a third trusted summary. There is nowhere to write it now: the fraction
    is what recomputing the population produced, and it comes back on the result."""
    from shogym.envs.receipts import bundle as bundle_mod

    for name in bundle_mod.CONTENTS:
        text = (built_bundle.root / name).read_text(encoding="utf-8")
        assert "considered" not in text
        assert "passing_fraction" not in text
    checked = bundle_mod.verify(built_bundle, GENERATOR)
    assert checked.passing_fraction == pytest.approx(
        len(checked.instances) / checked.considered
    )


# ----- the thresholds, the code pin and the bank identity -----


@pytest.mark.parametrize(
    "change",
    [{"min_arity": 99.0}, {"min_blocks": 99.0}, {"max_copy_score": 0.9}],
)
def test_the_persisted_bars_are_the_registered_ones(bundle_room, change) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.THRESHOLDS)
    root = _tamper(bundle_room, bundle_mod.THRESHOLDS, {**stored, **change})
    assert any(
        "registered bars" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_a_threshold_field_cannot_be_left_out(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.THRESHOLDS)
    root = _tamper(
        bundle_room, bundle_mod.THRESHOLDS,
        {k: v for k, v in stored.items() if k != "min_material_rows"},
    )
    assert any(
        "names exactly" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


@pytest.mark.parametrize("pin", [{"digest": ""}, {}, {"digest": "0" * 64}])
def test_a_bundle_without_the_running_code_s_pin_is_refused(bundle_room, pin) -> None:
    """An absent pin was silently skipped. There is no absent pin now: the file is
    named in the manifest, and both of its fields, the aggregate digest and the
    module-to-digest map, are compared with the running code."""
    from shogym.envs.receipts import bundle as bundle_mod

    root = _tamper(bundle_room, bundle_mod.CODE, pin)
    assert any(
        "code pin does not hold" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


@pytest.mark.parametrize(
    "change", [{"generator": "elsewhere"}, {"genre": "something else"}]
)
def test_a_bundle_that_names_another_family_is_refused(bundle_room, change) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.BANK)
    root = _tamper(bundle_room, bundle_mod.BANK, {**stored, **change})
    assert any(
        "population does not recompute" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_a_bank_record_with_a_field_nobody_reads_is_refused(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.BANK)
    root = _tamper(bundle_room, bundle_mod.BANK, {**stored, "admission": "gated"})
    assert any(
        "bank does not read" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


# ----- the review travels inside the bundle -----


def test_a_review_pack_read_of_one_bank_is_refused_over_another(tmp_path) -> None:
    """A human read renders under ONE draw. Two banks under two master keys draw two
    different conventions, so a pack read for the first attests to nothing about the
    second, and being inside a hashed directory does not make it about that bank. The
    pack names the bank it was read from and verification refuses a mismatch.
    """
    from shogym.envs.receipts import bundle as bundle_mod

    one = tmp_path / "one"
    one.mkdir()
    bank, screen, pack, _ = _materials(one)
    first = bundle_mod.build(one / "bundles", GENERATOR, bank, screen, pack)

    two = tmp_path / "two"
    two.mkdir()
    other, screen_two, _, _ = _materials(two, master=bytes(range(1, 33)))
    assert other.master != bank.master
    with pytest.raises(ValueError, match="was read from bank"):
        bundle_mod.build(two / "bundles", GENERATOR, other, screen_two, pack)

    # Nothing in the honest bundle names another, and it holds no records digest that
    # could be edited to make one pack claim another's instances.
    for name in bundle_mod.CONTENTS:
        text = (first.root / name).read_text(encoding="utf-8")
        assert "records_digest" not in text


def test_one_bank_has_one_spelling_and_therefore_one_bundle_address(
    bundle_room,
) -> None:
    """A bundle is addressed by the hash of its files, so one value has one spelling.

    `bytes.fromhex` reads an uppercased or spaced master into the same bytes, so the
    same bank could be frozen twice under two addresses and both would verify. A genre
    holding two bundles with no digest named is a genre production refuses to serve.
    """
    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts import bundle as bundle_mod

    stored = json.loads((bundle_room / bundle_mod.BANK).read_text(encoding="utf-8"))
    assert bank_mod.bank_from_record(stored).master.hex() == stored["master"]
    shouted = {**stored, "master": str(stored["master"]).upper()}
    # The bytes are the same bank; the spelling is not the canonical one.
    assert bytes.fromhex(shouted["master"]) == bytes.fromhex(stored["master"])
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        bank_mod.bank_from_record(shouted)
    root = _tamper(bundle_room, bundle_mod.BANK, shouted)
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert not checked.verified
    assert any("does not read" in problem for problem in checked.problems)


def test_a_screen_selected_from_many_candidates_is_not_deal_evidence(
    bundle_room,
) -> None:
    """Disclosure is not an adjustment, so a selected winner is not dealable.

    The best of several clears a bar more easily than one does, and nothing corrects
    for it: the interval, the bars and the screen's own verdict are identical for one
    candidate and for a million. A note said so and the bundle verified anyway, which
    made the disclosure the whole remedy. Until an adjustment is registered a selected
    record is scored, printed, and refused as evidence a bundle may be frozen on.
    """
    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.receipts import ScreenRecord

    selected = _screen_payload(candidates_screened=1000000,
                               selection_note="best of a million")
    # It is a readable record and its arithmetic still passes, which is the point.
    record = ScreenRecord.from_payload(selected)
    assert record.result("ledger").verdict
    assert not record.dealable_selection

    root = _tamper(bundle_room, bundle_mod.SCREEN, selected)
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert not checked.verified
    assert any("1000000 candidates" in problem for problem in checked.problems)


def test_a_screen_taken_on_another_family_is_refused(bundle_room) -> None:
    """Three numbers say what was measured, never what it was measured ON.

    The family label used to arrive from the caller at verification time, so a pilot
    run on anything at all froze into any bundle and read as its room. The run names
    the family it was taken on and verification refuses a mismatch.
    """
    from shogym.envs.receipts import bundle as bundle_mod

    root = _tamper(bundle_room, bundle_mod.SCREEN, _screen_payload(family="wordle"))
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert not checked.verified
    assert any("taken on 'wordle'" in problem for problem in checked.problems)


def test_a_review_pack_that_names_another_family_is_refused(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = json.loads((bundle_room / bundle_mod.REVIEW).read_text(encoding="utf-8"))
    root = _tamper(bundle_room, bundle_mod.REVIEW, {**stored, "family": "wordle"})
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert not checked.verified
    assert any("read of 'wordle'" in problem for problem in checked.problems)


def test_the_code_pin_records_the_modules_it_covers_and_names_a_moved_one(
    bundle_room, monkeypatch
) -> None:
    """One opaque digest says a bundle is stale and never says where.

    The pin records the module list with a hash each, so a reader can see what it
    covers, see where it stopped, and be told which module moved.
    """
    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts import bundle as bundle_mod

    stored = json.loads((bundle_room / bundle_mod.CODE).read_text(encoding="utf-8"))
    assert set(stored) == {"digest", "modules"}
    assert "shogym.envs.receipts.generators.ledger" in stored["modules"]
    # The stated boundary is readable: what is absent is outside the closure.
    assert "shogym.serve.lifecycle" not in stored["modules"]
    assert "shogym.core" not in stored["modules"]

    moved = dict(stored["modules"])
    moved["shogym.envs.receipts.generators.ledger"] = "0" * 64
    root = _tamper(bundle_room, bundle_mod.CODE, {**stored, "modules": moved})
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert not checked.verified
    assert any(
        "shogym.envs.receipts.generators.ledger changed" in problem
        for problem in checked.problems
    )
    assert bank_mod.code_pin(GENERATOR)["digest"] == stored["digest"]


def test_bundle_takes_no_argument_pointing_at_another_bank() -> None:
    """The command freezes the genre's own bank and takes no alternative path.

    What this pins is the command line, and nothing more. It is NOT a proof that
    nobody chose the bank: `materialize --force` rerolls the master as often as an
    operator likes, the bank directory is redirectable, and a bank record is five
    fields anyone can write. Resistance to operator selection is process here.
    """
    from shogym.envs.receipts import cli as receipts_cli

    parser = argparse.ArgumentParser()
    receipts_cli.add_parser(parser.add_subparsers(dest="command", required=True))
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["receipts", "bundle", "ledger", "--screen", "s.json", "--review", "p.json",
             "--bank", "mine.json"]
        )


def test_a_gate_exhibit_is_never_served(tmp_path) -> None:
    """Vectors wear the protocol so the gates can be exercised through the shipped
    path. They have no surface and no admission, and refusing them in the env is what
    makes "never dealt" a property of the environment rather than of whichever
    command happened to check."""
    from shogym.envs.receipts.env_v1 import ReceiptsDevEnv

    with pytest.raises(ValueError, match="gate exhibit"):
        ReceiptsDevEnv(genre="slots-c3", bank=str(tmp_path / "nothing.json"))


@pytest.mark.parametrize("reviewer", ["", "   "])
def test_a_blank_reviewer_is_not_a_reviewer(bundle_room, reviewer) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.REVIEW)
    root = _tamper(bundle_room, bundle_mod.REVIEW, {**stored, "reviewer": reviewer})
    assert any(
        "reviewer is blank" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_a_render_the_bundle_does_not_hold_covers_nothing(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.REVIEW)
    renders = [dict(entry) for entry in stored["renders"]]
    renders[0]["path"] = "renders/never-copied.txt"
    root = _tamper(bundle_room, bundle_mod.REVIEW, {**stored, "renders": renders})
    problems = bundle_mod.verify_at(root, GENERATOR).problems
    assert any("not a file this bundle holds" in problem for problem in problems)
    assert any("covers no" in problem for problem in problems)


def test_a_one_byte_render_is_not_a_review(tmp_path) -> None:
    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.envs.receipts.review import required_coverage

    bank, screen, pack, held = _materials(tmp_path)
    stored = json.loads(pack.read_text(encoding="utf-8"))
    tiny = tmp_path / "renders" / "tiny.txt"
    tiny.write_text("x", encoding="utf-8")
    renders = [dict(entry) for entry in stored["renders"]]
    renders[0]["path"] = "renders/tiny.txt"
    pack.write_text(json.dumps({**stored, "renders": renders}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not verify"):
        bundle_mod.build(tmp_path / "bundles", GENERATOR, bank, screen, pack)
    coverage = required_coverage(GENERATOR, checks.FILING_CLASSES, [24])
    assert ("counterfactual", "alternative convention") in coverage.required


def test_a_pack_missing_a_category_fails(tmp_path) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    bank, screen, pack, _ = _materials(tmp_path)
    stored = json.loads(pack.read_text(encoding="utf-8"))
    kept = [e for e in stored["renders"] if e["category"] != "option"]
    pack.write_text(json.dumps({**stored, "renders": kept}), encoding="utf-8")
    with pytest.raises(ValueError, match="covers no option:"):
        bundle_mod.build(tmp_path / "bundles", GENERATOR, bank, screen, pack)


def test_the_coverage_is_enumerated_from_the_family(tmp_path) -> None:
    """Every surface, every option, every filing shape, every row count, and a
    counterfactual, each present and each hashed by the manifest."""
    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.envs.receipts.review import required_coverage

    bank, screen, pack, held = _materials(tmp_path)
    counts = [i.a.n_rows for i in held.instances] + [i.b.n_rows for i in held.instances]
    coverage = required_coverage(GENERATOR, checks.FILING_CLASSES, counts)
    assert ("option", "anchor=event") in coverage.required
    assert ("surface", "claims") in coverage.required
    built = bundle_mod.build(tmp_path / "bundles", GENERATOR, bank, screen, pack)
    renders = _read(built.root, bundle_mod.REVIEW)["renders"]
    assert len(renders) == len(coverage.required)
    assert bundle_mod.verify(built, GENERATOR).problems == ()


def test_review_row_coverage_comes_from_both_siblings() -> None:
    """The protocol builds A and B independently, so a bank can hold two row counts."""
    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts.generators.ledger import LedgerTable
    from shogym.envs.receipts.review import required_coverage

    class ShortB(_Wrapped):
        def build_table(self, master, ordinal, label):
            table = self._inner.build_table(master, ordinal, label)
            if label.upper() == "B":
                return LedgerTable(
                    domain=table.domain, rows=table.rows[:-1],
                    holidays=table.holidays, body=table.body,
                )
            return table

    generator = ShortB()
    bank = bank_mod.materialize(generator, MASTER, 1)
    held = bank_mod.population(bank, generator)
    instance = held.instances[0]
    assert instance.a.n_rows != instance.b.n_rows
    coverage = required_coverage(
        generator, checks.FILING_CLASSES, [instance.a.n_rows, instance.b.n_rows]
    )
    counts = sorted(key for kind, key in coverage.required if kind == "rows")
    assert counts == ["23", "24"]


def test_the_review_pack_requires_both_pools_of_surfaces() -> None:
    from shogym.envs.receipts.generators import ledger
    from shogym.envs.receipts.review import required_coverage

    coverage = required_coverage(GENERATOR, checks.FILING_CLASSES, [24])
    surfaces = {key for kind, key in coverage.required if kind == "surface"}
    assert set(ledger.POOL_A) <= surfaces
    assert set(ledger.POOL_B) <= surfaces


# ----- the screen names what it was taken on, and the bars it was judged by -----


def test_a_screen_is_rerun_on_its_own_rows(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    thin = _screen_payload()
    thin["pairs"] = [dict(row, graded=0.4) for row in thin["pairs"]]
    root = _tamper(bundle_room, bundle_mod.SCREEN, thin)
    assert any(
        "took 0.0000 of the room its oracle had" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


@pytest.mark.parametrize(
    "field", ["min_room", "min_ratio", "min_pairs", "floor", "floor_rule",
              "candidates_screened", "selection_note"]
)
def test_every_decision_input_is_required(bundle_room, field) -> None:
    """A bar the reader supplies is a bar the record does not state."""
    from shogym.envs.receipts import bundle as bundle_mod

    payload = {k: v for k, v in _screen_payload().items() if k != field}
    root = _tamper(bundle_room, bundle_mod.SCREEN, payload)
    assert any(
        "names exactly" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_a_nonfinite_identity_is_not_a_name() -> None:
    """`str(float("nan"))` is the perfectly ordinary-looking name "nan", and Python's
    JSON reader accepts the token by default."""
    from shogym.receipts import ScreenRecord, ScreenRun, read_payload

    with pytest.raises(ValueError, match="not a JSON number"):
        read_payload('{"model": NaN}')
    with pytest.raises(ValueError, match="which names nothing"):
        ScreenRun.from_payload({**_run_payload(), "model": float("nan")})
    with pytest.raises(ValueError, match="which names nothing"):
        ScreenRun.from_payload(
            {**_run_payload(), "task_seeds": [float("inf")] * 40}
        )
    with pytest.raises(ValueError, match="is a number"):
        ScreenRecord.from_payload(_screen_payload(min_room="0.1"))


def test_a_bundle_carrying_a_nonfinite_screen_is_refused(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    (bundle_room / bundle_mod.SCREEN).write_text(
        json.dumps(_screen_payload()).replace('"a scripted policy"', "NaN"),
        encoding="utf-8",
    )
    root = _reseal(bundle_room)
    assert any(
        "not a readable record" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_screening_several_candidates_has_to_be_declared(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.receipts import Outcomes, ScreenRecord, screen

    rows = [{"placebo": 0.4, "graded": 0.6, "oracle": 0.9} for _ in range(40)]
    quiet = screen(
        "f", Outcomes.from_rows(rows), min_room=0.1, min_ratio=0.3,
        candidates_screened=6,
    )
    assert not quiet.verdict
    assert any("selection" in r for r in quiet.reasons)

    with pytest.raises(ValueError, match="says nothing about the selection"):
        ScreenRecord.from_payload(_screen_payload(candidates_screened=6))
    with pytest.raises(ValueError, match="says nothing about the selection"):
        ScreenRecord.from_payload(
            _screen_payload(candidates_screened=6, selection_note="   ")
        )
    declared = ScreenRecord.from_payload(
        _screen_payload(candidates_screened=6, selection_note="six screened, one kept")
    )
    assert declared.result("ledger").verdict
    root = _tamper(
        bundle_room, bundle_mod.SCREEN, _screen_payload(candidates_screened=6)
    )
    assert any(
        "not a readable record" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_a_screen_below_the_registered_sample_is_not_deal_evidence(tmp_path) -> None:
    """Both routes: lowering the bar, and keeping the bar with too few tasks."""
    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.receipts.screen import REGISTERED_MIN_PAIRS

    room = tmp_path / "one"
    room.mkdir()
    bank, screen, pack, _ = _materials(room)
    for payload, expected in (
        (_screen_payload(pairs=4, min_pairs=REGISTERED_MIN_PAIRS - 2),
         "not the registered ones"),
        (_screen_payload(pairs=4), "distinct tasks where 36 is registered"),
    ):
        screen.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            bundle_mod.build(tmp_path / "bundles", GENERATOR, bank, screen, pack)


def test_a_bare_list_of_scores_is_not_a_screen_artifact() -> None:
    """Three numbers say what was measured, not what it was measured on."""
    from shogym.receipts import ScreenRun

    with pytest.raises(ValueError, match="not a bare list of scores"):
        ScreenRun.from_payload([{"placebo": 0.4, "graded": 0.6, "oracle": 0.9}])


def test_a_run_binds_every_pair_to_an_identity() -> None:
    from shogym.receipts import ScreenRun

    payload = _run_payload(pairs=3)
    assert ScreenRun.from_payload(payload).outcomes().n_pairs == 3
    short = dict(payload, task_seeds=["0"])
    with pytest.raises(ValueError, match="one ordered set"):
        ScreenRun.from_payload(short)
    doubled = dict(payload)
    doubled["pairs"] = [dict(payload["pairs"][0]) for _ in range(3)]
    with pytest.raises(ValueError, match="repeats instances"):
        ScreenRun.from_payload(doubled)
    nameless = dict(payload)
    nameless["pairs"] = [dict(row, instance="") for row in payload["pairs"]]
    with pytest.raises(ValueError, match="instance is blank"):
        ScreenRun.from_payload(nameless)


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"model": None}, "names no model"),
        ({"model": "   "}, "model is blank"),
        ({"task_seeds": [None] * 40}, "task seed 0"),
        ({"task_seeds": [""] * 40}, "task seed 0 is blank"),
        ({"model": {"name": "m"}}, "model is not a name"),
    ],
)
def test_missing_run_provenance_is_refused_not_defaulted(payload, expected) -> None:
    """`str(None)` is the nonempty string "None", so converting before validating turns
    a pilot export that lost its model field into a run that names one."""
    from shogym.receipts import ScreenRun

    with pytest.raises(ValueError, match=expected):
        ScreenRun.from_payload({**_run_payload(), **payload})


def test_a_pair_with_no_identity_or_no_score_is_refused() -> None:
    from shogym.receipts import ScreenRun

    base = _run_payload(pairs=3)
    for change, expected in (
        ({"instance": None}, "names no instance"),
        ({"filing": " "}, "filing is blank"),
        ({"placebo": None}, "placebo is not a score"),
        ({"graded": "0.6"}, "graded is not a score"),
    ):
        payload = dict(base)
        payload["pairs"] = [dict(row, **change) for row in base["pairs"]]
        with pytest.raises(ValueError, match=expected):
            ScreenRun.from_payload(payload)


def test_the_screen_reports_an_interval_and_refuses_room_it_cannot_establish() -> None:
    from shogym.receipts import Outcomes, screen

    steady = [{"placebo": 0.4, "graded": 0.6, "oracle": 0.9} for _ in range(40)]
    good = screen("f", Outcomes.from_rows(steady), min_room=0.1, min_ratio=0.3)
    assert good.verdict
    assert good.room_low > 0.0 and good.room_high >= good.room_low
    # a sample whose room straddles zero has not established that there is any
    noisy = [
        {"placebo": 0.5, "graded": 0.5, "oracle": 1.0 if i % 2 else 0.0}
        for i in range(40)
    ]
    unsure = screen("f", Outcomes.from_rows(noisy), min_room=0.0, min_ratio=0.0)
    assert unsure.room_low <= 0.0
    assert not unsure.verdict
    assert any("does not establish" in r for r in unsure.reasons)


def test_the_sample_the_screen_needs_is_registered_not_chosen() -> None:
    """A caller free to pick the sample can pick the one that passes."""
    from shogym.receipts import Outcomes, screen
    from shogym.receipts.screen import REGISTERED_MIN_PAIRS

    assert REGISTERED_MIN_PAIRS == 36
    rows = [{"placebo": 0.4, "graded": 0.6, "oracle": 0.9} for _ in range(4)]
    assert not screen(
        "f", Outcomes.from_rows(rows), min_room=0.1, min_ratio=0.3
    ).verdict


def test_the_screen_bars_are_registered_and_an_override_is_declared() -> None:
    """The bars are the maintainer's call, and the call has been made.

    A diagnostic run may still ask what a family does against another bar, which is
    why they are defaults rather than constants. What a bundle may carry is a
    different question, and the answer is in the verifier.
    """
    from shogym.receipts import (
        REGISTERED_MIN_PAIRS,
        REGISTERED_MIN_RATIO,
        REGISTERED_MIN_ROOM,
        Outcomes,
        ScreenRecord,
        screen,
    )

    assert (REGISTERED_MIN_ROOM, REGISTERED_MIN_RATIO) == (0.05, 0.25)
    rows = [{"placebo": 0.4, "graded": 0.6, "oracle": 0.9} for _ in range(40)]
    default = screen("f", Outcomes.from_rows(rows))
    assert default.min_room == REGISTERED_MIN_ROOM
    assert default.min_ratio == REGISTERED_MIN_RATIO
    assert default.min_pairs == REGISTERED_MIN_PAIRS
    assert default.registered and default.verdict
    assert "BARS                   registered" in "\n".join(default.lines())

    moved = screen("f", Outcomes.from_rows(rows), min_room=0.1, min_ratio=0.3)
    assert not moved.registered
    assert "OVERRIDDEN" in "\n".join(moved.lines())

    registered = ScreenRecord.from_payload(_screen_payload())
    assert registered.registered and registered.overrides() == []
    overridden = ScreenRecord.from_payload(_screen_payload(min_ratio=0.3))
    assert not overridden.registered
    assert overridden.overrides() == ["min_ratio=0.3 against the registered 0.25"]


@pytest.mark.parametrize(
    "change", [{"min_ratio": 0.3}, {"min_room": 0.0}, {"min_pairs": 40}]
)
def test_a_bundle_judged_against_other_bars_is_not_dealable(bundle_room, change) -> None:
    """Recording the bar is not enforcing it. A family admitted under an easier rule
    was not admitted under the rule the measurement is registered under, and a harder
    one was not either: a dealable bundle carries the registered bars exactly, the way
    it carries the registered gate thresholds."""
    from shogym.envs.receipts import bundle as bundle_mod

    root = _tamper(bundle_room, bundle_mod.SCREEN, _screen_payload(**change))
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert any("not the registered ones" in problem for problem in checked.problems)


def test_a_screen_under_the_registered_bars_is_refused(bundle_room) -> None:
    """The recomputed statistics are compared with the registered bars where they are
    required, not inferred from a verdict composed elsewhere."""
    from shogym.envs.receipts import bundle as bundle_mod

    thin = _screen_payload()
    # room 0.04 against the registered 0.05, and the ratio still healthy
    thin["pairs"] = [
        dict(row, placebo=0.30, graded=0.335, oracle=0.34) for row in thin["pairs"]
    ]
    root = _tamper(bundle_room, bundle_mod.SCREEN, thin)
    problems = bundle_mod.verify_at(root, GENERATOR).problems
    assert any("under the registered 0.05" in problem for problem in problems)


# ----- a pair is a task, not an observation of one -----


def test_forty_filings_on_one_table_are_not_a_forty_pair_pilot() -> None:
    """The pair bootstrap prices every row as an independent draw. Forty rows on one
    clerical table clear the sample floor while the pilot has one sampled unit."""
    from shogym.receipts import ScreenRun

    one_task = _screen_payload()
    one_task["pairs"] = [
        {"instance": "one-table", "filing": f"f{n:02d}",
         "placebo": 0.4, "graded": 0.6, "oracle": 0.9}
        for n in range(40)
    ]
    with pytest.raises(ValueError, match="repeats instances"):
        ScreenRun.from_payload(
            {k: one_task[k] for k in ("family", "model", "task_seeds", "pairs")}
        )
    one_seed = {k: _screen_payload()[k] for k in ("family", "model", "task_seeds", "pairs")}
    one_seed["task_seeds"] = ["7"] * 40
    with pytest.raises(ValueError, match="repeats task seeds"):
        ScreenRun.from_payload(one_seed)
    honest = ScreenRun.from_payload(
        {k: _screen_payload()[k] for k in ("family", "model", "task_seeds", "pairs")}
    )
    assert honest.distinct_instances == 40


def test_a_bundle_screened_on_one_repeated_task_is_refused(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    repeated = _screen_payload()
    repeated["task_seeds"] = ["7"] * 40
    repeated["pairs"] = [
        {"instance": "one-table", "filing": f"f{n:02d}",
         "placebo": 0.4, "graded": 0.6, "oracle": 0.9}
        for n in range(40)
    ]
    root = _tamper(bundle_room, bundle_mod.SCREEN, repeated)
    assert any(
        "not a readable record" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


# ----- a missing identity is missing, in the pack as in the run -----


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"reviewer": None}, "names no reviewer"),
        ({"reviewer": "   "}, "reviewer is blank"),
        ({"reviewer": {"name": "a"}}, "reviewer is not a name"),
        ({"checklist": [None]}, "names no checklist item 0"),
        ({"checklist": []}, "lists no checklist item"),
        ({"seeds": [""]}, "seed 0 is blank"),
    ],
)
def test_a_missing_pack_identity_is_refused_not_defaulted(tmp_path, change, expected):
    """`str(None)` is the nonempty string "None", so a pack export that lost the
    attesting person would be written into the bundle as a reviewer named None."""
    from shogym.envs.receipts import bundle as bundle_mod

    bank, screen, pack, _ = _materials(tmp_path)
    stored = json.loads(pack.read_text(encoding="utf-8"))
    pack.write_text(json.dumps({**stored, **change}), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        bundle_mod.build(tmp_path / "bundles", GENERATOR, bank, screen, pack)


def test_a_bundled_pack_identity_is_checked_again_at_verification(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.REVIEW)
    root = _tamper(bundle_room, bundle_mod.REVIEW, {**stored, "reviewer": None})
    assert any(
        "names no reviewer" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


# ----- a field the verifier ignores is a field a conclusion travels in -----


def test_a_stored_conclusion_in_the_review_pack_is_refused(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.REVIEW)
    root = _tamper(
        bundle_room, bundle_mod.REVIEW,
        {**stored, "passing_fraction": 1.0, "reviewed": True},
    )
    assert any(
        "names exactly" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


def test_a_stored_conclusion_in_a_pair_row_is_refused(bundle_room) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _screen_payload()
    stored["pairs"] = [dict(row, passed=True) for row in stored["pairs"]]
    root = _tamper(bundle_room, bundle_mod.SCREEN, stored)
    assert any(
        "a pair names exactly" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


# ----- a bundle is bytes at a path, not a name for somebody else's -----


def test_a_bundle_that_is_itself_a_link_is_refused(built_bundle, tmp_path) -> None:
    """The digest-named entry would be a replaceable pointer, and the evidence could
    move through the outside name with no operation at the bundle's own path."""
    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.envs.receipts.registry import bundles

    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(built_bundle.root, elsewhere)
    home = tmp_path / "bundles" / "ledger"
    home.mkdir(parents=True)
    link = home / built_bundle.digest
    link.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="a name that can be repointed"):
        bundle_mod.load(link)
    assert bundle_mod.verify_at(link, GENERATOR).problems
    monkey = os.environ.get("SHOGYM_RECEIPTS_BANKS")
    os.environ["SHOGYM_RECEIPTS_BANKS"] = str(tmp_path)
    try:
        assert bundles("ledger") == []
    finally:
        if monkey is None:
            del os.environ["SHOGYM_RECEIPTS_BANKS"]
        else:
            os.environ["SHOGYM_RECEIPTS_BANKS"] = monkey


def test_a_hard_linked_render_is_refused(bundle_room, tmp_path) -> None:
    """A second hard link is a second name for the same inode, and the bytes under it
    can be replaced from outside without touching the bundle."""
    from shogym.envs.receipts import bundle as bundle_mod

    outside = tmp_path / "outside.txt"
    target = next(iter((bundle_room / "renders").iterdir()))
    shutil.copyfile(target, outside)
    target.unlink()
    os.link(outside, target)
    assert target.stat().st_nlink == 2
    with pytest.raises(ValueError, match="another name outside this bundle"):
        bundle_mod.load(bundle_room)


# ----- admission and the fork run the same acceptance predicate -----


class _WrongWrapper(_Wrapped):
    """Every row honest; the wrapper's row count is a constant 9999."""

    def _bend(self, ast):
        return ReceiptAST(kind=ast.kind, task_id=ast.task_id, row_count=9999,
                          rows=ast.rows, body=ast.body)

    def render_receipt(self, task, canonical, truth):
        return self._bend(self._inner.render_receipt(task, canonical, truth))

    def render_placebo(self, public, canonical, envelope):
        return self._bend(self._inner.render_placebo(public, canonical, envelope))


class _ShortPartialPlacebo(_Wrapped):
    """Honest everywhere except a partial filing, where the placebo is truncated."""

    def render_placebo(self, public, canonical, envelope):
        ast = self._inner.render_placebo(public, canonical, envelope)
        filed = getattr(canonical, "filed", None)
        if filed is not None and any(filed) and not all(filed):
            return ReceiptAST(kind=ast.kind, task_id=ast.task_id,
                              row_count=ast.row_count, rows=ast.rows[:12], body=ast.body)
        return ast


class _EmptyPartialPlacebo(_Wrapped):
    """Honest everywhere except a partial filing, where the placebo has no rows."""

    def render_placebo(self, public, canonical, envelope):
        ast = self._inner.render_placebo(public, canonical, envelope)
        filed = getattr(canonical, "filed", None)
        if filed is not None and any(filed) and not all(filed):
            return ReceiptAST(kind=ast.kind, task_id=ast.task_id,
                              row_count=ast.row_count, rows=(), body=ast.body)
        return ast


@pytest.mark.parametrize(
    "attack,failing",
    [
        (_WrongWrapper(), "graded"),
        (_ShortPartialPlacebo(), "placebo"),
        (_EmptyPartialPlacebo(), "placebo"),
    ],
)
def test_a_cell_the_fork_refuses_is_not_admitted(attack, failing) -> None:
    """A renderer with a wrapper bug, or one that drops rows on a partial filing, used
    to pass admission and then fail at every seal, or at a filing-dependent subset of
    them, which reaches an experiment as branch-specific missing outcomes on a family
    the instrument said was usable."""
    from shogym.envs.receipts import bank as bank_mod

    instance = protocol.draw(attack, MASTER, 1)
    report = admission.report(attack, instance, MASTER, admission.Thresholds())
    assert not report.admitted
    assert failing in report.failed_checks
    refusals = 0
    for shape in checks.FILING_CLASSES:
        raw = checks.filing_of(attack, instance, "a", shape)
        try:
            bank_mod.render_fork(attack, instance, "a", raw)
        except ValueError:
            refusals += 1
    assert refusals, "the fork accepted every filing, so there was nothing to catch"


def test_every_cell_admission_accepts_the_fork_accepts() -> None:
    """The two are one function, and this walks the registered filing classes on both
    siblings to say so on the honest family."""
    from shogym.envs.receipts import bank as bank_mod

    instance = _admitted()
    report = admission.report(GENERATOR, instance, MASTER, admission.Thresholds())
    assert report.admitted
    seen = 0
    for side in ("a", "b"):
        for shape in checks.FILING_CLASSES:
            raw = checks.filing_of(GENERATOR, instance, side, shape)
            fork = bank_mod.render_fork(GENERATOR, instance, side, raw)
            assert len(fork.graded) == instance.envelope.size
            seen += 1
    assert seen == 2 * len(checks.FILING_CLASSES)


def test_the_fork_and_admission_reach_one_judgement() -> None:
    """Not two implementations that agree: one function, called by both."""
    import inspect

    from shogym.envs.receipts import bank as bank_mod
    from shogym.envs.receipts import checks as checks_mod

    assert "judge_cells(" in inspect.getsource(bank_mod.render_fork)
    for name in ("check_graded", "check_placebo"):
        assert "judge_cells(" in inspect.getsource(getattr(checks_mod, name))


def test_the_judgement_is_built_before_any_renderer_runs() -> None:
    """A comparison built after a callback compares against whatever the callback
    left behind."""
    import inspect

    from shogym.envs.receipts.render import judge_cells

    source = inspect.getsource(judge_cells)
    assert source.index("expected = {") < source.index("asts = {")
    assert source.index("frozen_envelope(envelope)") < source.index("asts = {")


# ----- one representation per value, and one type per field -----


def test_a_name_that_appears_twice_has_no_single_value() -> None:
    """Python keeps the last member; another reader or a person auditing the file can
    take the first, so the artifact does not identify one record."""
    from shogym.receipts import read_payload

    with pytest.raises(ValueError, match="appears twice"):
        read_payload('{"size": 999, "size": 1}')
    with pytest.raises(ValueError, match="appears twice"):
        read_payload('{"a": {"graded": 0.0, "graded": 0.6}}')


@pytest.mark.parametrize("name", ["bank.json", "thresholds.json", "screen.json",
                                  "review.json", "instances.json"])
def test_a_repeated_name_anywhere_in_a_bundle_is_refused(bundle_room, name) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    path = bundle_room / name
    text = path.read_text(encoding="utf-8")
    doubled = (
        text[:-1] + ', "extra": 1, "extra": 2}' if text.startswith("{")
        else text[:-1] + ', {"extra": 1, "extra": 2}]'
    )
    path.write_text(doubled, encoding="utf-8")
    root = _reseal(bundle_room)
    assert bundle_mod.verify_at(root, GENERATOR).problems


@pytest.mark.parametrize("size", [1.9, "1", True, None])
def test_a_count_is_a_whole_number_and_never_coerced(size) -> None:
    """`int(1.9)` is 1 and `int(True)` is 1, so a serializer that emitted a fractional
    or boolean count would be read as a different bank than the one written."""
    from shogym.envs.receipts import bank as bank_mod

    record = {
        "generator": "ledger", "genre": "g",
        "renderer": bank_mod.RENDERER_CONFIGURATION, "master": "00" * 32, "size": size,
    }
    with pytest.raises(ValueError, match="a bank's size is a whole number"):
        bank_mod.bank_from_record(record)


def test_a_malformed_scalar_is_a_bundle_that_does_not_verify(bundle_room) -> None:
    """Never an exception out of a roster: `list` has to be able to say a bundle is
    not dealable, and this is exactly the case where saying so matters."""
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.BANK)
    root = _tamper(bundle_room, bundle_mod.BANK, {**stored, "size": None})
    checked = bundle_mod.verify_at(root, GENERATOR)
    assert not checked.verified
    assert any("whole number" in problem for problem in checked.problems)


@pytest.mark.parametrize("value", ["1", True, float("inf")])
def test_a_threshold_is_a_finite_number_and_never_coerced(bundle_room, value) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    stored = _read(bundle_room, bundle_mod.THRESHOLDS)
    path = bundle_room / bundle_mod.THRESHOLDS
    path.write_text(
        json.dumps({**stored, "min_material_rows": value}), encoding="utf-8"
    )
    root = _reseal(bundle_room)
    assert bundle_mod.verify_at(root, GENERATOR).problems


def test_every_bundle_file_is_in_canonical_form(built_bundle) -> None:
    from shogym.envs.receipts import bundle as bundle_mod

    for name in built_bundle.files:
        if not name.endswith(".json"):
            continue
        text = (built_bundle.root / name).read_text(encoding="utf-8")
        assert bundle_mod.canonical_json(json.loads(text)) == text
    text = (built_bundle.root / bundle_mod.MANIFEST).read_text(encoding="utf-8")
    assert bundle_mod.canonical_json(json.loads(text)) == text


@pytest.mark.parametrize("name", ["bank.json", "review.json"])
def test_a_file_that_is_not_canonical_is_refused(bundle_room, name) -> None:
    """A reordering or a space would change the bytes the manifest hashed without
    changing what any reader sees."""
    from shogym.envs.receipts import bundle as bundle_mod

    path = bundle_room / name
    path.write_text(
        json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2),
        encoding="utf-8",
    )
    root = _reseal(bundle_room)
    assert any(
        "canonical form" in problem
        for problem in bundle_mod.verify_at(root, GENERATOR).problems
    )


# ----- the pin covers what Python actually executes -----


def test_the_pin_covers_package_initializers() -> None:
    """Importing `a.b.c` runs `a/__init__.py` and `a/b/__init__.py` first, so whatever
    they do is part of what the run does."""
    from shogym.envs.receipts import bank as bank_mod

    pinned = bank_mod.pinned_modules(GENERATOR)
    assert "shogym.envs.receipts.generators" in pinned
    assert "shogym.envs.receipts" in pinned
    assert "shogym.receipts" in pinned


def test_a_package_initializer_change_moves_the_pin() -> None:
    import importlib.util
    from pathlib import Path as _Path

    from shogym.envs.receipts import bank as bank_mod

    base = bank_mod.current_code_digest(GENERATOR)
    spec = importlib.util.find_spec("shogym.envs.receipts.generators")
    assert spec and spec.origin
    path = _Path(spec.origin)
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# drift\n")
        assert bank_mod.current_code_digest(GENERATOR) != base
    finally:
        path.write_bytes(original)
    assert bank_mod.current_code_digest(GENERATOR) == base


def test_a_package_initializer_resolves_its_relative_imports_as_a_package(tmp_path):
    """Counting dots from the module name is one level too high inside every
    `__init__.py`, so a child a package initializer imports relatively went unseen."""
    import shutil as _shutil
    from pathlib import Path as _Path

    from shogym.envs.receipts import bank as bank_mod

    probe = _Path(str(_Path(bank_mod.__file__).parent)) / "_pin_probe"
    probe.mkdir()
    try:
        (probe / "__init__.py").write_text(
            "from . import decider\n\n__all__ = [\"decider\"]\n", encoding="utf-8"
        )
        (probe / "decider.py").write_text("VALUE = 1\n", encoding="utf-8")
        seen = bank_mod._imported("shogym.envs.receipts._pin_probe")
        assert "shogym.envs.receipts._pin_probe.decider" in seen
        assert bank_mod._ancestors("shogym.envs.receipts._pin_probe.decider") == {
            "shogym.envs.receipts", "shogym.envs.receipts._pin_probe"
        }
    finally:
        _shutil.rmtree(probe)
