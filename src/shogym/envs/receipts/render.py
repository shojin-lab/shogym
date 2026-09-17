"""The one way a graded row is built, and the one way a placebo row is built.

A family renderer does not get to decide what a graded row says. It says what the
scorer says happened on that row, and nothing else: the verdict is the row's
`matched` bit, and the correction is empty on a pass and the row's own correct
answer on a fail.

WHY THE RENDERER DOES NOT OWN THE VALUE. Closing the slot schema stopped a receipt
carrying a field the bytes never show. Closing the grammar stopped it printing a
numeric code. Neither stops the value that IS licensed from being chosen for what
it encodes rather than for what it means: four corrections spelled as ordinary band
names, one per axis, hand over the whole convention while every gate sees a legal
answer in a legal slot. The defence is not a longer list of forbidden values. It is
that the value is not a choice.

So the graded row is a function of the row's `RowOutcome` and the family's committed
feedback policy alone, and the receipt a generator actually rendered has to equal the
receipt this function would have rendered. A generator is free to render its own; it
is not free to render a different one.

WHICH ROWS IT REPORTS IS ALSO NOT A CHOICE. A family declares a receipt policy, and
under a sampled policy the rows that carry a verdict and a correction are the ones a
committed stream drew before any filing existed. Every other row prints that
position's committed neutral tokens in both slots, so the graded and placebo cells
are identical there even inside the slots. The policy and the mask reach this module
as one immutable `Feedback`, built from the frozen envelope before any renderer runs,
and the same object builds the expected cell and is handed to the family.

The placebo is the same idea from the other side: its slot values are the committed
neutral tokens for that row, fixed before launch.

`judge_cells` at the end of this module is where both of those are decided, along
with the wrapper, the envelope and the oracle's parse-back. It is the ONE acceptance
predicate: the post-filing fork runs it on the filing that sealed, and admission runs
it at every sample it takes, over every convention and every registered filing class.
A cell admission accepts is a cell the fork accepts, because one function decided
both.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.oracle import parse_body as parse_oracle_body
from shogym.envs.receipts.oracle import render as oracle_render
from shogym.envs.receipts.protocol import ReceiptPolicy, RowOutcome, policy_of
from shogym.envs.receipts.receipt_ast import (
    GRADED,
    PLACEBO,
    Envelope,
    ReceiptAST,
    ReceiptRow,
    Slot,
)

VERDICT_SLOT = "verdict"
CORRECTION_SLOT = "correction"
PASS_TOKEN = "PASS"
FAIL_TOKEN = "FAIL"


@dataclass(frozen=True)
class Feedback:
    """What the graded cell may report, fixed before any renderer runs.

    Three things, and they travel together because they are one commitment: the
    registered policy, the mask that policy drew for this task, and the committed
    neutral tokens a suppressed position prints. A renderer is handed this and cannot
    widen it, because the cell it produces is compared against the cell this same
    object builds: reporting a row the mask did not select is a refusal rather than a
    receipt that says more than the gate priced.

    Under the full policy the mask is every row and nothing is suppressed, which is
    what keeps a full-policy family's bytes the bytes it had.
    """

    policy: ReceiptPolicy
    #: The printed positions the receipt reports on, zero-based and in order.
    mask: tuple[int, ...]
    #: Slot name to one committed neutral token per printed position.
    neutral: Mapping[str, tuple[str, ...]]

    def reports(self, ordinal: int) -> bool:
        """Whether the row at this printed ordinal carries a verdict and a correction."""
        return (int(ordinal) - 1) in self.mask

    @property
    def positions(self) -> int:
        """How many printed positions this commitment covers."""
        return min((len(tokens) for tokens in self.neutral.values()), default=0)

    def covers(self, ordinal: int) -> bool:
        """Whether this commitment has a position at this printed ordinal at all."""
        return 1 <= int(ordinal) <= self.positions

    def suppressed(self, ordinal: int) -> dict[str, str]:
        """What every slot prints at a position the mask did not select.

        The caller asks `covers` first. A row printed at an ordinal the instance has no
        position for is already a row nothing can be compared against, and answering it
        with some position's tokens would put a policy complaint on a fault that is not
        about the policy.
        """
        return {name: tokens[int(ordinal) - 1] for name, tokens in self.neutral.items()}


def feedback_for(generator, task, envelope: Envelope) -> Feedback:
    """The feedback commitment for one task, read off the declaration and the task.

    Every caller in this package builds it from the FROZEN envelope, so the neutral
    tokens a suppressed position prints are the committed ones rather than whatever a
    renderer left behind.
    """
    return Feedback(
        policy=policy_of(generator),
        mask=tuple(task.mask),
        neutral=MappingProxyType(
            {name: tuple(values) for name, values in envelope.neutral.items()}
        ),
    )


def observed_cell(outcome: RowOutcome, blank_token: str, unfiled_token: str) -> str:
    """What the filed column echoes: the value, or which kind of nothing it was.

    Filing an empty value and filing nothing at all are different acts, so they are
    different tokens.
    """
    if not outcome.was_filed:
        return unfiled_token
    return outcome.filed if outcome.filed else blank_token


def canonical_correction(outcome: RowOutcome, blank_token: str) -> str:
    """What a failed row's correction says: that row's own correct answer.

    Nothing else is available to it. Not a code, not another row's answer, not a
    value chosen because of what it would let a reader deduce.
    """
    if outcome.matched:
        return ""
    return outcome.correct if outcome.correct else blank_token


def graded_rows(
    outcomes: Sequence[RowOutcome],
    blank_token: str,
    unfiled_token: str,
    feedback: Feedback,
) -> tuple[ReceiptRow, ...]:
    """The graded rows, as a function of the scorer's outcomes and the commitment.

    On a REPORTED row, the verdict is the row's matched bit and the correction is that
    row's own answer, which is what every row of a full receipt carries. On a row the
    mask did not select, both slots carry that position's committed neutral tokens,
    which is exactly what the placebo carries there: outside the selected rows the two
    cells are identical even inside the slots, so a reader learns nothing about those
    rows from the graded arm that the placebo arm does not also say.

    The echoed filing and the identifier are untouched by the policy. Suppressing the
    echo as well would move bytes the registered slots do not cover, and the row would
    stop being one the envelope check can hold against its placebo.
    """
    rows: list[ReceiptRow] = []
    for outcome in outcomes:
        if feedback.reports(outcome.ordinal):
            slots = (
                Slot(VERDICT_SLOT, PASS_TOKEN if outcome.matched else FAIL_TOKEN),
                Slot(CORRECTION_SLOT, canonical_correction(outcome, blank_token)),
            )
        else:
            quiet = feedback.suppressed(outcome.ordinal)
            slots = tuple(
                Slot(name, quiet[name]) for name in (VERDICT_SLOT, CORRECTION_SLOT)
            )
        rows.append(
            ReceiptRow(
                ordinal=outcome.ordinal,
                identifier=outcome.identifier,
                observed=observed_cell(outcome, blank_token, unfiled_token),
                slots=slots,
            )
        )
    return tuple(rows)


def graded_receipt(
    task_id: str,
    outcomes: Sequence[RowOutcome],
    blank_token: str,
    unfiled_token: str,
    feedback: Feedback,
) -> ReceiptAST:
    """The whole graded cell, built from the outcomes and the committed policy."""
    rows = graded_rows(outcomes, blank_token, unfiled_token, feedback)
    return ReceiptAST(
        kind=GRADED, task_id=task_id, row_count=len(rows), rows=rows
    )


def mask_disagreements(rendered: ReceiptAST, feedback: Feedback) -> list[str]:
    """Where a rendered graded cell reports rows other than the ones the mask drew.

    `row_disagreements` already refuses any row that is not the one the shared builder
    would have made, so this adds no strictness. What it adds is the diagnosis: a cell
    that reports an unselected row and a cell that prints the wrong band on a selected
    one are the same byte difference to an equality test and are two entirely different
    faults, and only one of them says more than the gate priced.
    """
    out: list[str] = []
    for row in rendered.rows:
        if not feedback.covers(row.ordinal):
            # A row at an impossible ordinal is refused by the row comparison, which
            # names the ordinal it printed. It is not a policy fault and this does not
            # claim it as one.
            continue
        quiet = feedback.suppressed(row.ordinal)
        printed = {slot.name: slot.value for slot in row.slots}
        silent = all(printed.get(name) == value for name, value in quiet.items())
        if feedback.reports(row.ordinal) and silent:
            out.append(
                f"row {row.ordinal} is one the committed mask selected and this cell "
                "prints its neutral tokens, so a reported row was suppressed"
            )
        elif not feedback.reports(row.ordinal) and not silent:
            out.append(
                f"row {row.ordinal} is not one the committed mask selected and this "
                "cell reports on it, so the receipt says more than the policy registers"
            )
    return out


def placebo_rows(
    outcomes: Sequence[RowOutcome],
    envelope: Envelope,
    blank_token: str,
    unfiled_token: str,
) -> tuple[ReceiptRow, ...]:
    """The placebo rows: the same shape, filled with this instance's committed tokens."""
    return tuple(
        ReceiptRow(
            ordinal=outcome.ordinal,
            identifier=outcome.identifier,
            observed=observed_cell(outcome, blank_token, unfiled_token),
            slots=tuple(
                Slot(spec.name, envelope.neutral[spec.name][outcome.ordinal - 1])
                for spec in envelope.slots
            ),
        )
        for outcome in outcomes
    )


def placebo_receipt(
    task_id: str,
    outcomes: Sequence[RowOutcome],
    envelope: Envelope,
    blank_token: str,
    unfiled_token: str,
) -> ReceiptAST:
    """The whole placebo cell."""
    rows = placebo_rows(outcomes, envelope, blank_token, unfiled_token)
    return ReceiptAST(
        kind=PLACEBO, task_id=task_id, row_count=len(rows), rows=rows
    )


def row_disagreements(rendered: ReceiptAST, expected: ReceiptAST) -> list[str]:
    """Where a rendered cell differs from the one this module would have built.

    EXACT equality on the whole row, field by field, not a list of the fields anyone
    thought to name. The printed ordinal is a visible column like any other, and a
    comparison that skipped it let a renderer write the drawn convention into it while
    every registered slot stayed honest. A future visible field must not be able to
    recreate that gap, so the default is equality and the named fields exist only to
    say which one differed.
    """
    out: list[str] = []
    if len(rendered.rows) != len(expected.rows):
        return [f"{len(rendered.rows)} rows against the expected {len(expected.rows)}"]
    for got, want in zip(rendered.rows, expected.rows):
        if got == want:
            continue
        if got.ordinal != want.ordinal:
            out.append(f"a row prints ordinal {got.ordinal!r}, not {want.ordinal!r}")
        if got.identifier != want.identifier:
            out.append(f"row {want.ordinal} names {got.identifier!r}, not {want.identifier!r}")
        if got.observed != want.observed:
            out.append(f"row {want.ordinal} echoes {got.observed!r}, not {want.observed!r}")
        if tuple(got.slots) != tuple(want.slots):
            by_name = {slot.name: slot.value for slot in got.slots}
            for slot in want.slots:
                if by_name.get(slot.name) != slot.value:
                    out.append(
                        f"row {want.ordinal} slot {slot.name} prints "
                        f"{by_name.get(slot.name)!r}, not {slot.value!r}"
                    )
        if not out:
            out.append(f"row {want.ordinal} differs from the row the grader would build")
    return out


def wrapper_disagreements(rendered: ReceiptAST, expected: ReceiptAST) -> list[str]:
    """Where a rendered cell's wrapper differs from the expected one."""
    out: list[str] = []
    for field in ("kind", "task_id", "row_count", "body"):
        got, want = getattr(rendered, field), getattr(expected, field)
        if got != want:
            out.append(f"{field} is {got!r}, not {want!r}")
    return out


def oracle_difference(rendered: ReceiptAST, expected: ReceiptAST) -> str:
    """The first place a rendered oracle stops being the registered one."""
    for field in ("kind", "task_id", "row_count"):
        got, want = getattr(rendered, field), getattr(expected, field)
        if got != want:
            return f"its {field} is {got!r} and the template's is {want!r}"
    if rendered.rows:
        return "it prints rows, and the oracle has none to align"
    extra = len(rendered.body) - len(expected.body)
    if extra > 0:
        return f"it prints {extra} lines the template does not"
    if extra < 0:
        return f"it is {-extra} lines short of the template"
    for n, (got, want) in enumerate(zip(rendered.body, expected.body)):
        if got != want:
            return f"body line {n + 1} is {got!r} and the template's is {want!r}"
    return "its body differs from the template's"


def frozen_template(template: OracleTemplate) -> OracleTemplate:
    """A snapshot of a declared oracle template, deep enough to compare against.

    The phrase tables are ordinary dictionaries hanging off family code, so a family
    that edited them inside its own renderer would be editing what it is about to be
    compared with. This is taken before any family callback runs.
    """
    return OracleTemplate(
        head=tuple(template.head),
        sentences=MappingProxyType(dict(template.sentences)),
        phrases=MappingProxyType(
            {axis: MappingProxyType(dict(options))
             for axis, options in template.phrases.items()}
        ),
    )


# --------------------------------------------------------------------------
# the one acceptance predicate: what a set of three cells has to be
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cells:
    """The three cells a fork would commit, and everything wrong with them.

    `problems` empty means these cells are servable. A caller that is about to
    persist them raises on the first entry; a caller that is deciding whether a
    family is admissible reports them.
    """

    asts: Mapping[str, ReceiptAST]
    payloads: Mapping[str, bytes]
    score: float
    outcomes: tuple[RowOutcome, ...]
    problems: tuple[str, ...]

    @property
    def acceptable(self) -> bool:
        return not self.problems


def judge_cells(
    generator,
    task,
    canonical,
    convention: Mapping[str, str],
    envelope: Envelope,
) -> Cells:
    """Render the three cells for one filing and say whether they may be served.

    THIS IS THE WHOLE CONTRACT, IN ONE PLACE. The post-filing fork calls it on the
    filing that actually sealed, and admission calls it at every sample it takes,
    over the whole convention space and every registered filing class. There is no
    second, weaker version: a cell admission accepts is a cell the fork accepts,
    because the same function decided both. A renderer with a wrapper bug, or one
    that drops rows on a partial filing, otherwise passes admission and then fails at
    every seal, which lands in an experiment as branch-specific missing outcomes on a
    family the instrument said was usable.

    EVERYTHING COMPARED AGAINST IS BUILT BEFORE ANY RENDERER RUNS. Family code can
    mutate a structure it is handed, including the committed neutral tokens and the
    drawn convention, so a comparison built afterwards would be a comparison against
    whatever the renderer left behind. Snapshots first, renderers second, equality
    last.

    `task.key` is the truth the graded cell is rendered against and `convention` is
    the rule the oracle has to state; a caller walking the support passes a retasked
    task and that convention together.
    """
    from shogym.envs.receipts.receipt_ast import (
        ORACLE,
        frozen_envelope,
        mask_slots,
        serialize,
        slot_ranges,
    )

    committed = frozen_envelope(envelope)
    drawn = MappingProxyType(dict(convention))
    blank = str(getattr(generator, "BLANK_TOKEN", "(empty)"))
    unfiled = str(getattr(generator, "UNFILED_TOKEN", "(none)"))
    score, outcomes = generator.score(task, canonical)
    # The commitment, taken from the frozen envelope and the task's own committed mask
    # BEFORE the renderer runs. The same object builds the expected cell and is handed
    # to the family, so what it may report and what it is held to are one thing.
    feedback = feedback_for(generator, task, committed)
    expected = {
        GRADED: graded_receipt(task.task_id, outcomes, blank, unfiled, feedback),
        PLACEBO: placebo_receipt(task.task_id, outcomes, committed, blank, unfiled),
    }
    # The oracle this package would render, built from the declared phrase table
    # BEFORE the family's own renderer runs, so the thing being compared against is
    # not something the renderer left behind. The declared table is snapshotted with
    # it, for the same reason.
    template = getattr(generator, "ORACLE", None)
    if isinstance(template, OracleTemplate):
        expected[ORACLE] = oracle_render(
            frozen_template(template), task.task_id, drawn, task.n_rows
        )

    asts = {
        GRADED: generator.render_receipt(task, canonical, task.key, feedback),
        PLACEBO: generator.render_placebo(task.public(), canonical, committed),
        ORACLE: generator.render_oracle(task.task_id, drawn, task.n_rows),
    }

    problems: list[str] = []
    # WHICH ROWS, then WHAT THEY SAY. A cell that reports a row the mask did not draw
    # and a cell that prints the wrong answer on a row it did are the same inequality
    # and two different faults, so the policy question is asked first and answered in
    # its own words.
    off_mask = mask_disagreements(asts[GRADED], feedback)
    if off_mask:
        problems.append(
            "the graded cell does not report the rows the committed mask drew: "
            + "; ".join(off_mask[:3])
        )
    wrong = row_disagreements(asts[GRADED], expected[GRADED])
    if wrong:
        problems.append(
            "the graded cell is not what the scorer's own outcomes say: "
            + "; ".join(wrong[:3])
        )
    wrong = row_disagreements(asts[PLACEBO], expected[PLACEBO])
    if wrong:
        problems.append(
            "the placebo cell does not print its committed tokens: " + "; ".join(wrong[:3])
        )
    for kind in (GRADED, PLACEBO):
        off = wrapper_disagreements(asts[kind], expected[kind])
        if off:
            problems.append(
                f"the {kind} cell's wrapper is not the one this fork commits: "
                + "; ".join(off[:3])
            )

    # THE ORACLE IS THE PACKAGE'S ORACLE, EXACTLY. Parsing one option per axis back out
    # of whatever a family printed leaves the wrapper and the rest of the body free: a
    # renderer can put another task's reference over it, print a different record
    # count, and append coaching under the rule, and every one of those rides into the
    # arm that is the denominator of the room the whole screen measures. So the family
    # renders the oracle and the comparison is equality with the one this package
    # renders from the declared phrases, wrapper and body together.
    if not isinstance(template, OracleTemplate):
        problems.append(
            "this family declares no oracle phrase table, so what its oracle says "
            "cannot be established"
        )
    else:
        # WHAT IT SAYS, then WHAT IT IS. The rule the family printed is read first,
        # because "it states another rule" is the diagnosis a reader wants and a byte
        # difference is not; then the whole cell has to be the registered one.
        try:
            stated = parse_oracle_body(template, asts[ORACLE].body)
        except ValueError as exc:
            problems.append(f"this fork's oracle cannot be read: {exc}")
        else:
            if dict(stated) != dict(drawn):
                problems.append(
                    "this fork's oracle states a rule that was not the one drawn, so "
                    "the room it measures is not this family's room"
                )
        try:
            declared = parse_oracle_body(template, expected[ORACLE].body)
        except ValueError as exc:
            problems.append(f"this family's oracle template cannot be read back: {exc}")
        else:
            if dict(declared) != dict(drawn):
                problems.append(
                    "this family's oracle template does not read back to the rule it "
                    "was given, so the room it measures is not this family's room"
                )
        if asts[ORACLE] != expected[ORACLE]:
            problems.append(
                "this fork's oracle is not the cell the registered template renders, "
                + oracle_difference(asts[ORACLE], expected[ORACLE])
            )

    # Serialization is inside the predicate rather than around it. The serializer
    # refuses a field it cannot fit and a cell that overruns the envelope, and a
    # refusal thrown from here would leave the caller with an exception where it was
    # promised a verdict: `problems` empty is what says these cells are servable, so
    # a cell that cannot be made is a problem with it and not a crash in the judging.
    payloads: dict[str, bytes] = {}
    for kind, ast in asts.items():
        try:
            payloads[kind] = serialize(ast, committed)
        except ValueError as exc:
            problems.append(
                f"the {kind} cell cannot be serialized into the registered envelope: {exc}"
            )
    for kind, payload in payloads.items():
        if len(payload) != committed.size:
            problems.append(
                f"the {kind} cell is {len(payload)} bytes against an envelope of "
                f"{committed.size}"
            )
    if not problems:
        ranges = slot_ranges(asts[GRADED], committed)
        if mask_slots(payloads[GRADED], ranges) != mask_slots(payloads[PLACEBO], ranges):
            problems.append(
                "the graded and placebo cells differ outside the registered slots, so "
                "the placebo carries something the slots do not account for"
            )
    return Cells(
        asts=asts,
        payloads=payloads,
        score=score,
        outcomes=tuple(outcomes),
        problems=tuple(problems),
    )


__all__ = [
    "CORRECTION_SLOT",
    "FAIL_TOKEN",
    "PASS_TOKEN",
    "VERDICT_SLOT",
    "Cells",
    "Feedback",
    "canonical_correction",
    "feedback_for",
    "frozen_template",
    "mask_disagreements",
    "oracle_difference",
    "graded_receipt",
    "graded_rows",
    "judge_cells",
    "observed_cell",
    "placebo_receipt",
    "placebo_rows",
    "row_disagreements",
    "wrapper_disagreements",
]
