"""The sampled receipt: which rows it reports, what it says on the rest, and who decides.

A receipt that reports four records of twenty four is a different instrument from one
that reports all of them, and the difference has to be a fact about the artifact rather
than an intention of the renderer. These hold the four things that makes true: the rows
are drawn before any filing and independently of the drawn rule, a rebuild draws the
same ones, the cell says nothing at all on the rows it did not draw, and a cell that
says otherwise is refused.

Every test names the condition it fails under, because a test whose failure nobody can
state is a test nobody can act on.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from shogym.envs.receipts import checks
from shogym.envs.receipts.bank import instance_record, render_fork
from shogym.envs.receipts.generators import ledger, soundchange, soundchange_audit
from shogym.envs.receipts.generators.vectors import VECTORS
from shogym.envs.receipts.protocol import (
    FULL_RECEIPT,
    SAMPLED_FOUR_OF_TWENTY_FOUR,
    SAMPLED_TWO_OF_TWENTY_FOUR,
    ReceiptPolicy,
    Task,
    conventions,
    draw,
    policy_of,
    receipt_mask,
)
from shogym.envs.receipts.receipt_ast import (
    GRADED,
    PLACEBO,
    frozen_envelope,
    row_lines,
    serialize,
)
from shogym.envs.receipts.receipt_law import law_for
from shogym.envs.receipts.registry import load_generator
from shogym.envs.receipts.render import (
    FAIL_TOKEN,
    PASS_TOKEN,
    feedback_for,
    judge_cells,
)

MASTER = bytes(range(32))
ENVELOPE_BYTES = 2657
#: Sound change's own count, named here so the law can be priced at it before and
#: after the family declares it.
TWO_OF_TWENTY_FOUR = SAMPLED_TWO_OF_TWENTY_FOUR


def _sampled():
    """The ledger family under the sampled policy.

    A subclass while the family itself still declares the full receipt, and the family
    once it declares the sampled one. The mask stream is keyed by the generator's NAME,
    which does not change, so the rows this draws are the rows the family draws.
    """
    if ledger.GENERATOR.RECEIPT_POLICY.samples:
        return ledger.GENERATOR

    class SampledLedger(ledger.LedgerGenerator):
        RECEIPT_POLICY: ReceiptPolicy = SAMPLED_FOUR_OF_TWENTY_FOUR

    return SampledLedger()


def _canonical(generator, task, raw=None):
    if raw is None:
        raw = "\n".join(
            f"{i},{v}" for i, v in zip(generator.row_identifiers(task.table), task.key)
        )
    return generator.parse_and_canonicalize(task, raw)


def _judge(generator, instance, side="a", raw=None, convention=None):
    task = instance.side(side)
    return judge_cells(
        generator,
        task,
        _canonical(generator, task, raw),
        convention or instance.convention,
        instance.envelope,
    )


# ----- which rows, and who chose them -----


def test_the_mask_does_not_depend_on_the_drawn_convention() -> None:
    """The rows a receipt reports are drawn from a stream of their own.

    FAILS IF the same generator at the same ordinal draws different rows when the
    convention differs, which would make the selection a channel the rule can be read
    off and would make the gate's average over a uniform mask law an average over a
    law nobody registered.
    """
    generator = _sampled()
    first, second = ledger.ALL_CONVENTIONS[0], ledger.ALL_CONVENTIONS[-1]
    assert first != second

    class OneRule(type(generator)):  # type: ignore[misc]
        SUPPORT = (MappingProxyType(dict(first)),)

    class OtherRule(type(generator)):  # type: ignore[misc]
        SUPPORT = (MappingProxyType(dict(second)),)

    one = draw(OneRule(), MASTER, 3)
    other = draw(OtherRule(), MASTER, 3)
    assert dict(one.convention) != dict(other.convention)
    assert one.a.mask == other.a.mask
    assert one.b.mask == other.b.mask


def test_the_mask_is_the_registered_draw_and_is_taken_before_the_task_text() -> None:
    """The mask is a function of the key, the generator, the ordinal and the side.

    FAILS IF the instance's mask is not the one the registered law draws from those
    four coordinates, which would mean the rows a receipt reports came from somewhere
    the gate cannot reproduce.
    """
    generator = _sampled()
    instance = draw(generator, MASTER, 1)
    for side, label in (("a", "A"), ("b", "B")):
        task = instance.side(side)
        assert task.mask == receipt_mask(
            policy_of(generator), MASTER, generator.name, 1, label, task.n_rows
        )
        assert len(task.mask) == SAMPLED_FOUR_OF_TWENTY_FOUR.reported
        assert len(set(task.mask)) == len(task.mask)
        assert all(0 <= row < task.n_rows for row in task.mask)
    assert instance.a.mask != instance.b.mask


def test_the_mask_does_not_depend_on_the_filing() -> None:
    """The mask exists before the agent has filed anything.

    FAILS IF two filings of one task produce receipts that report different rows,
    which would let an agent choose what it is told by choosing what it files.
    """
    generator = _sampled()
    instance = draw(generator, MASTER, 2)
    task = instance.a
    identifiers = generator.row_identifiers(task.table)
    reported: list[tuple[int, ...]] = []
    for raw in (
        "\n".join(f"{i},{v}" for i, v in zip(identifiers, task.key)),
        "\n".join(f"{i}," for i in identifiers),
        "",
        "\n".join(f"{i},Routine" for i in identifiers),
    ):
        judged = _judge(generator, instance, "a", raw)
        assert not judged.problems, judged.problems
        rows = judged.asts[GRADED].rows
        reported.append(
            tuple(
                row.ordinal
                for row in rows
                if {s.name: s.value for s in row.slots}["verdict"]
                in (PASS_TOKEN, FAIL_TOKEN)
            )
        )
    assert len(set(reported)) == 1
    assert reported[0] == tuple(row + 1 for row in task.mask)


def test_a_rebuild_reproduces_the_same_mask() -> None:
    """Re-rendering after a crash serves the cell the instance was committed with.

    FAILS IF a second draw at one ordinal selects other rows, or if the bank's
    instance record leaves the mask out, either of which would let two branches of one
    fork receive graded cells that report different records.
    """
    generator = _sampled()
    once = draw(generator, MASTER, 5)
    again = draw(generator, MASTER, 5)
    assert once.a.mask == again.a.mask
    assert once.b.mask == again.b.mask
    record = instance_record(once, generator)
    assert record["a"]["mask"] == list(once.a.mask)
    assert record["b"]["mask"] == list(once.b.mask)
    assert record["receipt_policy"] == policy_of(generator).as_record()
    assert checks.check_fixation(generator, once, MASTER).passed


def test_the_task_says_what_the_receipt_will_report_on() -> None:
    """One sentence, after the scope sentence and before the schedule, in both arms.

    FAILS IF the sentence is absent, moves, differs between the two siblings, differs
    under two conventions, or names an option of the hidden rule. A reader not told
    that twenty of the twenty four lines say nothing has been handed twenty rows of
    apparent evidence that those records were filed correctly, and a reader told which
    four were selected could file the rest at random and lose nothing.
    """
    from shogym.envs.receipts.protocol import option_mentions

    generator = _sampled()
    expected = (
        "The receipt for this schedule reports the verdict and the correct band for "
        "four\nselected records only, and the lines for the other records say nothing "
        "about\nwhether they were right."
    )
    assert "\n".join(ledger.RECEIPT_SENTENCE) == expected
    assert ledger.receipt_sentence(FULL_RECEIPT) == ""
    assert option_mentions(generator.AXES, expected) == []
    for ordinal in (0, 1):
        instance = draw(generator, MASTER, ordinal)
        for side, label in (("a", "A"), ("b", "B")):
            text = instance.side(side).text
            assert text.count(expected) == 1
            assert text.index(ledger.scope_sentence(label)) < text.index(expected)
            assert text.index(expected) < text.index("\nSCHEDULE (")
    # the same bytes in every arm and under every convention: it is a fact about the
    # policy, not about the draw, so the invariance check has to keep holding.
    assert checks.check_invariance(generator, draw(generator, MASTER, 0)).passed
    assert checks.check_lint(generator, draw(generator, MASTER, 0)).passed


# ----- what the cells say -----


def test_graded_and_placebo_are_identical_outside_the_selected_rows() -> None:
    """Outside the four rows the receipt reports, the two arms are the same bytes.

    FAILS IF any row the mask did not select differs between the graded and the
    placebo cell, INSIDE the registered slots as well as outside them. A difference
    there is feedback the policy does not declare and the gate did not price, on
    twenty of the twenty four rows.
    """
    generator = _sampled()
    for ordinal in (0, 1, 2):
        instance = draw(generator, MASTER, ordinal)
        for side in ("a", "b"):
            task = instance.side(side)
            envelope = frozen_envelope(instance.envelope)
            judged = _judge(generator, instance, side)
            assert not judged.problems, judged.problems
            graded_lines = row_lines(
                judged.payloads[GRADED], judged.asts[GRADED], envelope
            )
            placebo_lines = row_lines(
                judged.payloads[PLACEBO], judged.asts[PLACEBO], envelope
            )
            assert len(graded_lines) == task.n_rows
            for position, (one, other) in enumerate(zip(graded_lines, placebo_lines)):
                if position in task.mask:
                    assert one != other
                else:
                    assert one == other


def test_a_suppressed_row_prints_its_own_committed_neutral_tokens() -> None:
    """Silence is the placebo's token for that position, not a token chosen here.

    FAILS IF a row outside the mask prints anything but the committed neutral tokens
    for its own position: another position's token would carry the offset, and any
    other string would be a value the receipt is free to give a meaning.
    """
    generator = _sampled()
    instance = draw(generator, MASTER, 4)
    task = instance.a
    envelope = frozen_envelope(instance.envelope)
    judged = _judge(generator, instance, "a")
    assert not judged.problems, judged.problems
    for row in judged.asts[GRADED].rows:
        printed = {slot.name: slot.value for slot in row.slots}
        if row.ordinal - 1 in task.mask:
            assert printed["verdict"] in (PASS_TOKEN, FAIL_TOKEN)
            continue
        for name in ("verdict", "correction"):
            assert printed[name] == envelope.neutral[name][row.ordinal - 1]


def test_the_envelope_holds_for_every_filing_class_and_every_convention() -> None:
    """All three cells are 2657 bytes whatever was filed and whatever was drawn.

    FAILS IF any cell is any other size. The envelope is what makes the three arms
    indistinguishable by length, and a sampled receipt prints shorter slot values than
    a full one on twenty of its rows.
    """
    generator = _sampled()
    instance = draw(generator, MASTER, 0)
    envelope = frozen_envelope(instance.envelope)
    assert envelope.size == ENVELOPE_BYTES
    seen = 0
    for side in ("a", "b"):
        task = instance.side(side)
        for shape in checks.FILING_CLASSES:
            raw = checks.filing_of(generator, instance, side, shape)
            for convention in conventions(generator.AXES)[::7]:
                truth = tuple(generator.key_for(task.table, convention))
                retasked = Task(
                    label=task.label, task_id=task.task_id, surface=task.surface,
                    table=task.table, text=task.text, key=truth, mask=task.mask,
                )
                judged = judge_cells(
                    generator,
                    retasked,
                    generator.parse_and_canonicalize(retasked, raw),
                    convention,
                    instance.envelope,
                )
                assert not judged.problems, (side, shape, judged.problems)
                for payload in judged.payloads.values():
                    assert len(payload) == ENVELOPE_BYTES
                seen += 1
    assert seen == 2 * len(checks.FILING_CLASSES) * len(conventions(generator.AXES)[::7])


# ----- what the judge refuses -----


class _ReportsAnUnselectedRow:
    """A renderer that reports the rows it was given and one more besides."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def render_receipt(self, task, canonical, truth, feedback):
        from shogym.envs.receipts.receipt_ast import ReceiptAST, ReceiptRow, Slot

        ast = self._inner.render_receipt(task, canonical, truth, feedback)
        spare = next(
            row.ordinal for row in ast.rows if not feedback.reports(row.ordinal)
        )
        rows = tuple(
            ReceiptRow(
                ordinal=row.ordinal,
                identifier=row.identifier,
                observed=row.observed,
                slots=(Slot("verdict", PASS_TOKEN), Slot("correction", "")),
            )
            if row.ordinal == spare
            else row
            for row in ast.rows
        )
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count, rows=rows
        )


class _SuppressesASelectedRow:
    """A renderer that goes quiet on a row the mask drew."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def render_receipt(self, task, canonical, truth, feedback):
        from shogym.envs.receipts.receipt_ast import ReceiptAST, ReceiptRow, Slot

        ast = self._inner.render_receipt(task, canonical, truth, feedback)
        quiet = next(row.ordinal for row in ast.rows if feedback.reports(row.ordinal))
        rows = tuple(
            ReceiptRow(
                ordinal=row.ordinal,
                identifier=row.identifier,
                observed=row.observed,
                slots=tuple(
                    Slot(name, value)
                    for name, value in feedback.suppressed(row.ordinal).items()
                ),
            )
            if row.ordinal == quiet
            else row
            for row in ast.rows
        )
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count, rows=rows
        )


class _WritesItsOwnTokenWhereItIsQuiet:
    """A renderer that prints a legal-looking number where it owes silence."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def render_receipt(self, task, canonical, truth, feedback):
        from shogym.envs.receipts.receipt_ast import ReceiptAST, ReceiptRow, Slot

        ast = self._inner.render_receipt(task, canonical, truth, feedback)
        code = "".join(
            str(list(axis.options).index(convention[axis.name]))
            for axis in ledger.AXES
            for convention in [self._rule(task, truth)]
        )
        rows = tuple(
            ReceiptRow(
                ordinal=row.ordinal,
                identifier=row.identifier,
                observed=row.observed,
                slots=(
                    Slot("verdict", code),
                    Slot("correction", feedback.suppressed(row.ordinal)["correction"]),
                ),
            )
            if not feedback.reports(row.ordinal)
            else row
            for row in ast.rows
        )
        return ReceiptAST(
            kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count, rows=rows
        )

    @staticmethod
    def _rule(task, truth):
        for convention in ledger.ALL_CONVENTIONS:
            if tuple(ledger.key_for(task.table, convention)) == tuple(truth):
                return convention
        return ledger.ALL_CONVENTIONS[0]


@pytest.mark.parametrize(
    "attack, says",
    [
        (_ReportsAnUnselectedRow, "not one the committed mask selected"),
        (_SuppressesASelectedRow, "prints its neutral tokens"),
        (_WritesItsOwnTokenWhereItIsQuiet, "not one the committed mask selected"),
    ],
)
def test_the_judge_refuses_a_cell_that_reports_other_rows(attack, says) -> None:
    """The cell has to report the rows the mask drew, and only those.

    FAILS IF a renderer can add a verdict on a row the mask did not draw, drop one it
    did, or write its own token where it owes the committed neutral one. The first is
    a receipt that says more than the gate priced, the second is a receipt that says
    less than the task text promised, and the third is a four-digit statement of the
    whole rule in a slot the grammar was supposed to close.
    """
    generator = _sampled()
    instance = draw(generator, MASTER, 0)
    judged = _judge(attack(generator), instance, "a")
    assert judged.problems
    assert any(says in problem for problem in judged.problems), judged.problems


def test_the_slot_grammar_licenses_only_this_position_s_committed_token() -> None:
    """At a suppressed position the grammar is one token wide.

    FAILS IF gate S accepts any other value at a row the mask did not draw, or if it
    refuses the honest cell. A slot that licensed every neutral token everywhere would
    license a short numeric code stating the whole rule, which is the reading the
    closed grammar exists to refuse, and one that licensed none would refuse silence
    itself.
    """
    from shogym.envs.receipts.observe import observe
    from shogym.receipts.resolution import grammar_violations

    generator = _sampled()
    instance = draw(generator, MASTER, 0)
    honest = observe(generator, instance, "a")
    assert grammar_violations(honest) == []
    quiet = {
        name: set(tokens) for name, tokens in instance.envelope.neutral.items()
    }
    for name, licensed in honest.slot_row_grammar.items():
        for position, allowed in enumerate(licensed):
            if position in instance.a.mask:
                assert not (set(allowed) & quiet[name])
            else:
                assert allowed == frozenset(
                    {instance.envelope.neutral[name][position]}
                )
    forged = observe(_WritesItsOwnTokenWhereItIsQuiet(generator), instance, "a")
    assert grammar_violations(forged)


def test_an_undeclared_policy_is_refused_at_registration() -> None:
    """A module that has not said what its receipt reports is not a generator yet.

    FAILS IF a generator without a declared policy, or with one nobody registered, can
    be loaded: it would be gated under whatever this package tried first and served
    under whatever it rendered.
    """

    class Undeclared(ledger.LedgerGenerator):
        RECEIPT_POLICY = None  # type: ignore[assignment]

    class Invented(ledger.LedgerGenerator):
        RECEIPT_POLICY = ReceiptPolicy(
            name="sampled-8-of-24", shape="sampled-rows", reported=8, rows=24
        )

    with pytest.raises(ValueError, match="declares no RECEIPT_POLICY"):
        policy_of(Undeclared())
    with pytest.raises(ValueError, match="the registered policies are"):
        policy_of(Invented())
    for name in ("ledger", "soundchange", *sorted(VECTORS)):
        assert policy_of(load_generator(name)) in (
            FULL_RECEIPT,
            SAMPLED_TWO_OF_TWENTY_FOUR,
            SAMPLED_FOUR_OF_TWENTY_FOUR,
        )


# ----- the law the gate reads instead of the exercise check -----


def test_the_law_reproduces_the_gate_s_own_ceiling_and_floor() -> None:
    """The law's arithmetic is the gate's, asked about the whole mask instead of one.

    FAILS IF the ideal level, the lookup floor or the no-receipt level computed over
    masks disagrees with what the gate computes from the rendered receipt when the
    policy reports every row and the floor is taken at the drawn reference. There is
    one mask then, and the two calculations are the same question; a difference means
    one of them is pricing something the other is not.
    """
    from shogym.envs.receipts.observe import observe
    from shogym.envs.receipts.receipt_law import DRAWN_REFERENCE, law_for
    from shogym.receipts import gate

    class FullLedger(ledger.LedgerGenerator):
        RECEIPT_POLICY: ReceiptPolicy = FULL_RECEIPT

    generator = FullLedger()
    for ordinal in (0, 1):
        instance = draw(generator, MASTER, ordinal)
        law = law_for(generator, instance, "a", references=DRAWN_REFERENCE)
        scored = gate(observe(generator, instance, "a"))
        assert law.masks == 1
        assert abs(law.ideal - scored.ceiling) < 1e-12
        assert abs(law.floor - scored.floor) < 1e-12
        assert abs(law.no_receipt - scored.placebo) < 1e-12


def test_the_law_refuses_an_alternative_the_receipt_rarely_speaks_to() -> None:
    """Every single-axis alternative has to be distinguished often enough to teach.

    FAILS IF the check passes an instance whose weakest alternative sits under the
    registered 0.30, or fails one whose alternatives all clear it. An alternative the
    receipt almost never shows a row for is a choice the link cannot teach, whatever
    the realized mask happened to draw.
    """
    from shogym.envs.receipts.receipt_law import REGISTERED_MIN_DISTINGUISHING

    generator = _sampled()
    instance = draw(generator, MASTER, 0)
    passed = checks.check_receipt_law(generator, instance)
    assert passed.name == "law"
    assert passed.passed, passed.detail
    refused = checks.check_receipt_law(generator, instance, min_distinguishing=0.99)
    assert not refused.passed
    assert "under the registered" in refused.detail
    assert REGISTERED_MIN_DISTINGUISHING == 0.30
    # And it is the check the sampled family gets, in place of the exercise check.
    names = [
        result.name
        for result in checks.run_checks(
            generator,
            instance,
            MASTER,
            max_copy_score=1.0,
            max_flip_score=1.0,
            min_leverage=0.0,
        )
    ]
    assert names[0] == "law" and "exercise" not in names


def test_a_full_policy_family_still_gets_the_exercise_check() -> None:
    """The families that report every row are asked the question they can answer.

    FAILS IF the exercise check is dropped for a family whose receipt reports every
    row. The law replaced it for one instrument and not for the roster.
    """
    instance = draw(soundchange.GENERATOR, MASTER, 0)
    names = [
        result.name
        for result in checks.run_checks(
            soundchange.GENERATOR,
            instance,
            MASTER,
            max_copy_score=1.0,
            max_flip_score=1.0,
            min_leverage=0.0,
        )
    ]
    assert names[0] == "exercise" and "law" not in names


# ----- who decides admission, for a receipt that reports some rows -----


def test_a_mask_that_leaves_no_realized_headroom_is_admitted_under_the_law() -> None:
    """The realized mask does not select the instances the bank holds.

    FAILS IF admission refuses a sampled instance whose registered law clears every
    bar because the mask it happened to draw left too little headroom on the receipt
    in front of it. Ordinal 1 is that instance: R and S pass, its realized headroom is
    0.0394 against the 0.05 bar, every named check passes, and its law leaves an ideal
    reader 0.8398 against a lookup floor of 0.7704 with its weakest single-axis
    alternative distinguished with probability 0.5440. A mask that happens to omit an
    axis is not redrawn, and dropping the instance instead filters the same masks.
    """
    from shogym.envs.receipts.admission import Thresholds, report

    generator = _sampled()
    made = report(generator, draw(generator, MASTER, 1), MASTER, Thresholds())
    assert made.gates.r_pass and made.gates.s_pass
    assert not made.gates.h_pass and not made.gates.verdict
    assert made.gates.headroom < Thresholds().min_headroom
    assert made.failed_checks == ()
    assert made.admitted
    # And what replaces the realized verdict is reported rather than assumed.
    assert made.law is not None
    assert abs(made.law.ideal - 0.839846664) < 1e-9
    assert abs(made.law.weakest[1] - 0.544042914) < 1e-9


def test_a_full_policy_instance_that_fails_realized_headroom_is_refused() -> None:
    """The realized gates still decide for a family that reports every row.

    FAILS IF a full-policy instance is admitted while the gates refuse it. There is one
    receipt then and no law over masks behind it, so R, S and H are the whole question
    and nothing about the sampled policy loosens them. The same bar moved past the
    sampled family's realized headroom leaves it admitted, which is the difference.
    """
    from shogym.envs.receipts.admission import Thresholds, report

    full = soundchange.GENERATOR
    instance = draw(full, MASTER, 0)
    at_the_bars = report(full, instance, MASTER, Thresholds())
    assert at_the_bars.admitted and at_the_bars.gates.h_pass
    above = Thresholds(min_headroom=at_the_bars.gates.headroom + 0.01)
    refused = report(full, instance, MASTER, above)
    assert not refused.gates.h_pass and not refused.gates.verdict
    assert refused.failed_checks == ()
    assert not refused.admitted
    # The same move leaves the sampled family where it was.
    sampled = _sampled()
    drawn = draw(sampled, MASTER, 1)
    held = report(sampled, drawn, MASTER, Thresholds(min_headroom=0.99))
    assert not held.gates.h_pass
    assert held.admitted


def test_the_admitted_sampled_set_is_the_set_the_law_admits() -> None:
    """Nothing is excluded for what its own mask drew, over a run of ordinals.

    FAILS IF a sampled instance is excluded while every named check passes and the
    receipt's printed form is sound, which is the only way the realized mask could
    still be selecting the bank's population. It also fails if no drawn instance has a
    realized gate verdict the law overrides, because then the run proves nothing.
    """
    from shogym.envs.receipts.admission import Thresholds, decides, report

    generator = _sampled()
    registered = Thresholds()
    overridden = 0
    for ordinal in range(12):
        made = report(generator, draw(generator, MASTER, ordinal), MASTER, registered)
        assert made.samples
        assert made.gate_prerequisite == made.gates.s_form_pass
        assert made.admitted == (made.gates.s_form_pass and not made.failed_checks)
        if made.admitted and not made.gates.verdict:
            overridden += 1
        # A full-policy report over the same gates would have required all three.
        assert decides(made.gates, False) == made.gates.verdict
    assert overridden, "no drawn mask left a realized verdict for the law to override"


def _run_five_pairs():
    """The six ledger table pairs run 5 served, rebuilt from its own bank.

    Read-only, and skipped where that run is not on the machine. The bank names its
    master key and its size and nothing else, so the pairs are a function of the key
    and the ordinals its forks name; the tables do not depend on which gate set
    admitted them, which is what makes the rebuild exact.
    """
    import json
    import re
    from pathlib import Path

    from shogym.envs.receipts import streams

    root = Path("/Users/andrew/deutero-runs/claude-code-sonnet-5/bank")
    if not root.is_dir():
        return None, ()
    banks = sorted(root.glob("*/ledger-*.json"))
    if not banks:
        return None, ()
    record = json.loads(banks[0].read_text(encoding="utf-8"))
    master = bytes.fromhex(record["master"])
    served: set[str] = set()
    for path in banks[0].parent.rglob("fork-*.json"):
        found = re.match(r"fork-([0-9a-f]{16})-", path.name)
        if found:
            served.add(found.group(1))
    ordinals = [
        ordinal
        for ordinal in range(64)
        if streams.task_identifier(master, "ledger", ordinal, "A") in served
    ]
    return master, tuple(ordinals)


def test_the_law_reproduces_the_consultation_on_the_run_five_tables() -> None:
    """The arithmetic the policy was chosen on, recomputed by the code that gates it.

    FAILS IF the ideal level, the lookup floor or the expected number of consistent
    conventions over the six table pairs run 5 served differs from the consultation's
    exact calculation. A policy registered on numbers the implementation does not
    reproduce is a policy nobody has priced.

    The floor is held per pair and to six places, not to a tolerance. The two
    calculations are one construction: the evident rows the reduced receipt shows, plus
    one all-matched bit per dependence class among the rest, the class of empty local
    dependence included, averaged over every canonical reference. A row no single-axis
    alternative moves can still move under a joint one, so its class is a column a
    reader has and the floor concedes it.
    """
    from shogym.envs.receipts.receipt_law import bank_law, law_for

    master, ordinals = _run_five_pairs()
    if master is None or len(ordinals) != 6:
        pytest.skip("the run 5 bank is not on this machine")
    generator = _sampled()
    laws = [law_for(generator, draw(generator, master, o), "a") for o in ordinals]
    band = bank_law(laws)
    assert abs(band.ideal - 0.817285062) < 5e-10
    assert abs(sum(law.compatible for law in laws) / len(laws) - 7.206182) < 0.001
    assert [round(law.floor, 9) for law in laws] == [
        0.753568219,
        0.733884414,
        0.751611431,
        0.734439942,
        0.708373223,
        0.767741008,
    ]
    assert abs(band.floor - 0.741603040) < 5e-10
    assert band.passed


def test_eight_reported_rows_are_refused_by_the_band() -> None:
    """Twice the rows is a different instrument, and the band says so.

    FAILS IF a policy reporting eight rows of twenty four clears the registered band.
    Its ideal level over the six pairs run 5 served is 0.922879, above the 0.90 the
    band tops out at: a receipt that leaves that little has recreated the saturation
    this policy exists to undo, and switching to it later would raise the measured
    level by a tenth with no procedural learning at all.

    ONE PAIR, because eight reported rows of twenty four is 735471 masks against 10626
    and the walk is exact. The six-pair mean is in the build evidence; what this holds
    is that the arithmetic runs at another count and that the band refuses what it
    returns.
    """
    from shogym.envs.receipts.receipt_law import bank_law, law_for

    master, ordinals = _run_five_pairs()
    if master is None or len(ordinals) != 6:
        pytest.skip("the run 5 bank is not on this machine")
    generator = _sampled()
    eight = ReceiptPolicy(
        name="sampled-8-of-24", shape="sampled-rows", reported=8, rows=24
    )
    law = law_for(
        generator, draw(generator, master, ordinals[0]), "a", policy=eight
    )
    assert law.masks == 735471
    assert law.ideal > 0.90
    assert abs(law.ideal - 0.936706) < 1e-5
    band = bank_law([law])
    assert not band.passed
    assert any("outside the registered" in reason for reason in band.reasons)


def test_the_bank_band_is_read_over_the_bank_and_not_the_instance() -> None:
    """A fresh bank's mean ideal level sits inside the band with room above lookup.

    FAILS IF the mean over freshly drawn ledger instances leaves the registered 0.75 to
    0.90, or if the mean room over the recomputed lookup floor is not above 0.05. This
    is the claim the policy was registered on, taken on tables the consultation never
    saw, and it runs wherever the suite runs.
    """
    from shogym.envs.receipts.receipt_law import bank_law, law_for

    generator = _sampled()
    laws = [law_for(generator, draw(generator, MASTER, o), "a") for o in range(4)]
    band = bank_law(laws)
    assert band.passed, band.reasons
    assert 0.75 <= band.ideal <= 0.90
    assert band.room > 0.05
    # And a bank whose mean sits outside the band is refused, whichever side it is on.
    outside = bank_law(laws, min_ideal=0.95, max_ideal=0.99)
    assert not outside.passed
    tight = bank_law(laws, min_room=0.5)
    assert not tight.passed


def test_the_review_pack_covers_the_selected_and_suppressed_cases() -> None:
    """A person has to have seen what this receipt does before it is released.

    FAILS IF a pack for a family whose receipt reports only some rows can be verified
    without a reported pass, a reported failure, a reported omission, a reported legal
    empty answer, a suppressed failure, a receipt that reports one of the records with
    no dates and one that reports none, a posterior of one rule, a posterior of several
    with different held-out keys, the mask commitment beside the three cells, or a pair
    of conventions whose reduced receipts are identical. It also fails if a family that
    reports every row is asked for any of them.
    """
    from shogym.envs.receipts.review import SAMPLED_CASES, required_coverage, verify

    counts = [24, 24]
    sampled = required_coverage(_sampled(), checks.FILING_CLASSES, counts)
    full = required_coverage(soundchange.GENERATOR, checks.FILING_CLASSES, counts)
    added = [key for kind, key in sampled.required if kind == "sampled"]
    assert added == list(SAMPLED_CASES)
    assert len(SAMPLED_CASES) == 11
    assert not [key for kind, key in full.required if kind == "sampled"]
    # A pack that shows everything but one of them does not establish the family.
    seen = [pair for pair in sampled.required if pair != ("sampled", SAMPLED_CASES[4])]
    missing = sampled.missing(seen)
    assert missing == ["sampled:" + SAMPLED_CASES[4]]
    files = {f"renders/{n:03d}.txt": 4000 for n in range(len(sampled.required))}
    manifest = {
        "reviewer": "a person",
        "checklist": ["the suppressed rows say nothing"],
        "seeds": [0],
        "family": "ledger",
        "bank": "a-bank",
        "renders": [
            {
                "category": category,
                "key": key,
                "kind": "cell",
                "path": f"renders/{n:03d}.txt",
            }
            for n, (category, key) in enumerate(sampled.required)
        ],
    }
    assert verify(manifest, sampled, 2657, files, "ledger", "a-bank") == []
    short = dict(manifest, renders=manifest["renders"][:-1])
    assert verify(short, sampled, 2657, files, "ledger", "a-bank")


# ----- the families that keep the full receipt -----


def _reports_every_row(generator, instance, side: str) -> None:
    task = instance.side(side)
    envelope = frozen_envelope(instance.envelope)
    canonical = _canonical(generator, task)
    feedback = feedback_for(generator, task, envelope)
    ast = generator.render_receipt(task, canonical, task.key, feedback)
    _, outcomes = generator.score(task, canonical)
    assert len(ast.rows) == task.n_rows
    for row, outcome in zip(ast.rows, outcomes):
        printed = {slot.name: slot.value for slot in row.slots}
        assert printed["verdict"] == (PASS_TOKEN if outcome.matched else FAIL_TOKEN)
        assert printed["correction"] == (
            "" if outcome.matched else (outcome.correct or generator.BLANK_TOKEN)
        )
    assert len(serialize(ast, envelope)) == envelope.size


def test_a_full_policy_receipt_reports_a_verdict_and_a_correction_on_every_row() -> None:
    """The full receipt is the receipt it was: every row, verdict and same-row answer.

    FAILS IF a family that declares the full policy suppresses any row, which would
    move the bytes of every family that has not done its own arithmetic under the
    sampled policy and would silently change what their gate results were taken on.
    """

    class FullLedger(ledger.LedgerGenerator):
        RECEIPT_POLICY: ReceiptPolicy = FULL_RECEIPT

    for generator in (FullLedger(), soundchange.GENERATOR):
        instance = draw(generator, MASTER, 0)
        assert instance.a.mask == tuple(range(instance.a.n_rows))
        for side in ("a", "b"):
            _reports_every_row(generator, instance, side)


def test_sound_change_and_every_fixture_keep_the_full_receipt() -> None:
    """They keep it until their own arithmetic under the sampled policy is done.

    FAILS IF any of them declares a policy that samples, which the consultation's
    fixture arithmetic says would leave sound change at an ideal level of 0.96 and
    components at 0.98, or if the gate vectors move: every expected number in that
    module was computed on a receipt that reports every row.
    """
    assert policy_of(soundchange.GENERATOR) is FULL_RECEIPT
    for name, vector in sorted(VECTORS.items()):
        assert policy_of(vector) is FULL_RECEIPT, name


def test_a_full_policy_cell_is_the_cell_the_fork_commits() -> None:
    """Sound change's three cells are still the three cells the judge accepts.

    FAILS IF the policy machinery refuses, or changes, a cell a family that reports
    every row renders. Those families are not the subject of this change and nothing
    about their bytes may move with it.
    """
    instance = draw(soundchange.GENERATOR, MASTER, 0)
    for side in ("a", "b"):
        task = instance.side(side)
        fork = render_fork(
            soundchange.GENERATOR,
            instance,
            side,
            "\n".join(
                f"{i},{v}"
                for i, v in zip(
                    soundchange.GENERATOR.row_identifiers(task.table), task.key
                )
            ),
        )
        assert fork.component_score == 1.0
        for kind in (GRADED, PLACEBO):
            assert len(fork.agent_bytes(kind)) == instance.envelope.size


def test_a_conceded_axis_moves_the_floor_and_never_the_ceiling() -> None:
    """A comparator handed two axes reads a higher floor under the same receipt.

    FAILS IF handing the reader the reflex and the deleted vowel leaves the lookup
    floor where it was, which would mean the concession bought nothing and the
    comparator was not comparing anything; or if it moves the ideal level, which is
    what an ideal reader of the receipt reaches and is not a comparator's business; or
    if the law record does not say which axes were conceded, which would leave a number
    in the record nobody can say the meaning of.
    """
    instance = draw(soundchange.GENERATOR, MASTER, 0)
    plain = law_for(soundchange.GENERATOR, instance, "a", policy=TWO_OF_TWENTY_FOUR)
    given = law_for(
        soundchange.GENERATOR,
        instance,
        "a",
        policy=TWO_OF_TWENTY_FOUR,
        given=soundchange_audit.READABLE_AXES,
    )
    assert plain.ideal == given.ideal
    assert plain.augmented == plain.floor
    assert given.augmented > given.floor
    assert given.augmented_room == given.ideal - given.augmented
    assert given.as_record()["given"] == ["reflex", "loss"]
    assert given.as_record()["augmented"] == given.augmented
    assert "reflex and loss given away" in given.detail()
    assert "given away" not in plain.detail()


def test_an_axis_no_family_has_cannot_be_conceded() -> None:
    """A comparator hands over choices the family makes, not choices it does not.

    FAILS IF a misspelled axis name is quietly conceded as nothing, which would report
    a floor computed with no concession under a name that says otherwise.
    """
    instance = draw(soundchange.GENERATOR, MASTER, 0)
    with pytest.raises(ValueError, match="no axis"):
        law_for(
            soundchange.GENERATOR,
            instance,
            "a",
            policy=TWO_OF_TWENTY_FOUR,
            given=("reflexes",),
        )
