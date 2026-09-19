"""Export what a human has to have read before the retail refund genre is released.

A person reading rendered instances is the boundary against a family that is wrong in a
way no mechanical check can see. That only holds if the person saw the family, so the
pack is enumerated from the generator's own declarations rather than chosen by whoever
exported it: every surface template, every option of every axis, every registered filing
class, the row count the bank holds, and counterfactual renders under conventions that
were not the ones drawn.

WHAT THIS ADDS BEYOND THE SHARED COVERAGE. All 27 oracle cells on one pair, so the
reader sees every sentence the oracle arm can state rather than the one this draw
produced; and a counterfactual render for EVERY option of every axis on every surface,
each beside a worksheet whose destinations were computed by the independent selector, so
the reader can confirm that a greatest balance means a greatest current balance on a
pipe-separated schedule and on a JSON one alike.

AND WORKSHEETS, WHICH ARE NOT TASKS. Beside the renders it writes private worksheets: a
row-by-row trace of each counterfactual with the class the rule selected and the member
it took, the bank rows that exhibit each phenomenon a reader has to see, the equal-amount
fixtures the admitted distribution deliberately never produces, the separating
eight-case answer matrix, and the validation and copying numbers for the pair. Class
names and selected members belong there and nowhere else, and no cell prints them. The
worksheets are deliberately NOT in the review manifest and not a bundle field: they are
explanatory material for the roster release, carried with their own hashes, and
labelling them as renders would put a document nobody serves inside the evidence that
says what was served.

THE ATTESTATION IS LEFT UNSET. The manifest carries every field a pack carries and names
no reviewer, so `review.verify` refuses it until a person puts their name to it. An
exporter that filled that in would be a machine attesting that a human read something.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from shogym.envs.receipts import checks
from shogym.envs.receipts.bank import Bank, Population, bank_identity, render_fork
from shogym.envs.receipts.generators import retail_refund as retail
from shogym.envs.receipts.generators import retail_validation as validation
from shogym.envs.receipts.protocol import Instance, Task
from shogym.envs.receipts.receipt_ast import (
    GRADED,
    ORACLE,
    PLACEBO,
    frozen_envelope,
    mask_slots,
    serialize,
    slot_ranges,
)
from shogym.envs.receipts.render import feedback_for
from shogym.envs.receipts.review import required_coverage
from shogym.envs.receipts.streams import digest

#: Where the exported files live under the directory the caller names.
RENDERS = "renders"
WORKSHEETS = "worksheets"
PACK = "review.json"
WORKSHEET_INDEX = "worksheets.json"

#: What the reader is being asked to confirm. Enumerated here because a checklist the
#: exporter invents per run is a checklist nobody agreed to, and because these are the
#: judgements no mechanical check makes.
CHECKLIST = (
    "the same words mean the same operation on all four surfaces: a greatest balance "
    "is a greatest current balance, including on a gift card that paid nothing",
    "the public rules settle eligibility, class membership, the empty-class fallback "
    "and ties, and leave which class is preferred and how it is picked undetermined",
    "the receipt names cases and destinations and never a class, an axis or a "
    "preference, and a code alone is not a selector label because its binding is "
    "different in every case",
    "a failed row's correction is that same row's own destination code",
    "the oracle states the drawn convention and adds no coaching, no worked case and "
    "no scope a reader could map onto a name printed on the schedule",
    "the placebo carries no grade, no destination and nothing that reads as either",
    "the counterfactual worksheets show the stated rule producing the printed answers",
    "the oracle is a statement of a rule and not a list of answers",
    "the two schedules share no case and neither repeats an identifier",
)

#: The phenomena the worksheets have to exhibit from the bank itself, each a question
#: about one printed case.
WORKSHEET_CASES = (
    "a case whose original class is empty, so the public fallback sends it to the "
    "gift-card class",
    "a case whose gift-card class is empty, so the public fallback sends it to the "
    "original class",
    "a case of both classes where a gift card paid for the purchase",
    "a case of both classes where no gift card paid for the purchase",
    "a gift card whose current balance is zero, which is still a destination",
    "a sibling case whose first printed candidate holds the middle amount in its class",
)


@dataclass(frozen=True)
class Render:
    """One exported artifact and what it is evidence of."""

    category: str
    key: str
    kind: str
    path: str


def _oracle_sentences(convention: Mapping[str, str]) -> tuple[str, ...]:
    """The oracle's own wording for one convention, for a worksheet's right-hand side."""
    return retail.GENERATOR.render_oracle("0" * 16, convention, retail.ROWS).body


def _retasked(task: Task, convention: Mapping[str, str]) -> Task:
    """The same task scored under another convention, for a counterfactual render."""
    return Task(
        label=task.label,
        task_id=task.task_id,
        surface=task.surface,
        table=task.table,
        text=task.text,
        key=tuple(retail.key_for(task.table, convention)),
        mask=task.mask,
    )


def _cells(
    instance: Instance, task: Task, convention: Mapping[str, str], raw: object
) -> dict[str, bytes]:
    """The three cells one filing would produce, through the shared judge."""
    from shogym.envs.receipts.render import judge_cells

    generator = retail.GENERATOR
    canonical = generator.parse_and_canonicalize(task, raw)
    judged = judge_cells(generator, task, canonical, convention, instance.envelope)
    if judged.problems:
        raise ValueError(judged.problems[0])
    return dict(judged.payloads)


def _mixed_filing(task: Task, every: int = 3) -> str:
    """A filing that passes most cases and fails some, deterministically.

    Deterministic in the task's own printed order rather than drawn from a stream,
    because two readers of one pack have to be looking at one artifact and because the
    worksheet's verdict column is only worth reading if some rows failed.
    """
    lines = []
    identifiers = retail.GENERATOR.row_identifiers(task.table)
    for position, (identifier, correct, case) in enumerate(
        zip(identifiers, task.key, task.table.rows)
    ):
        if position % every:
            lines.append("%s,%s" % (identifier, correct))
            continue
        other = next(
            (i.code for i in case.instruments if i.code != correct), correct
        )
        lines.append("%s,%s" % (identifier, other))
    return "\n".join(lines)


def _omission_filing(task: Task) -> str:
    """A filing that answers the first half and says nothing about the rest."""
    identifiers = list(retail.GENERATOR.row_identifiers(task.table))
    half = len(identifiers) // 2
    return "\n".join(
        "%s,%s" % kv for kv in zip(identifiers[:half], list(task.key)[:half])
    )


def _write(root: Path, name: str, payload: bytes | str) -> str:
    """One exported file, returned as the path the manifest names it by."""
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        target.write_bytes(payload)
    else:
        target.write_text(payload, encoding="utf-8")
    return name


def _printed(case: validation.ParsedCase) -> list[dict[str, Any]]:
    """One parsed case's instruments, as the worksheet prints them."""
    return [
        {
            "code": i["code"],
            "type": i["type"],
            "balance": None if i["balance"] is None else "%d.%02d" % divmod(
                i["balance"], retail.CENTS
            ),
            "contribution": "%d.%02d" % divmod(i["contribution"], retail.CENTS),
        }
        for i in case.instruments
    ]


def _selected_by(case: validation.ParsedCase, convention: Mapping[str, str]) -> dict[str, Any]:
    """Which class the rule took and which member of it, by the independent selector."""
    gifts = [i for i in case.instruments if i["type"] == retail.GIFT_CARD]
    originals = [
        i for i in case.instruments
        if i["type"] in validation.NON_GIFT and i["contribution"] > 0
    ]
    used = any(i["contribution"] > 0 for i in gifts)
    gift_side = validation.prefers_gift(convention["route_policy"], used)
    if not (gifts if gift_side else originals):
        gift_side = not gift_side
    return {
        "class_selected": "gift cards" if gift_side else "original instruments",
        "fallback_applied": gift_side
        != validation.prefers_gift(convention["route_policy"], used),
        "destination": validation.audit_destination(case.instruments, convention),
    }


def _worksheet_rows(
    task: Task, convention: Mapping[str, str], raw: str
) -> list[dict[str, Any]]:
    """One worksheet row per printed case, with the class and member spelled out."""
    generator = retail.GENERATOR
    retasked = _retasked(task, convention)
    canonical = generator.parse_and_canonicalize(retasked, raw)
    _, outcomes = generator.score(retasked, canonical)
    out: list[dict[str, Any]] = []
    for case, outcome in zip(validation.parsed_cases(task.table), outcomes):
        entry: dict[str, Any] = {
            "case_id": case.case_id,
            "kind": validation.parsed_kind(case),
            "gift_used": case.gift_used,
            "instruments": _printed(case),
        }
        entry.update(_selected_by(case, convention))
        entry.update(
            {
                "filed": outcome.filed if outcome.was_filed else "(none)",
                "verdict": "PASS" if outcome.matched else "FAIL",
                "correction": "" if outcome.matched else outcome.correct,
            }
        )
        out.append(entry)
    return out


def _case_rows(population: Population) -> list[dict[str, Any]]:
    """The earliest printed case in the bank exhibiting each phenomenon.

    ALL SIX, OR THIS BANK IS NOT ONE A PACK CAN BE EXPORTED FROM. A reader confirming
    that the stated rule produces the printed answers cannot confirm it for a
    phenomenon they were not shown, so a shortfall is the bank's and it is named here
    rather than quietly narrowing what the pack is.
    """
    found: dict[str, dict[str, Any]] = {}
    for instance in population.instances:
        if len(found) == len(WORKSHEET_CASES):
            break
        for side in ("a", "b"):
            task = instance.side(side)
            size = retail.A_CLASS_SIZE if side == "a" else retail.B_CLASS_SIZE
            for case in validation.parsed_cases(task.table):
                gifts = [i for i in case.instruments if i["type"] == retail.GIFT_CARD]
                balances = [int(i["balance"] or 0) for i in gifts]
                kind = validation.parsed_kind(case)
                holds = {
                    WORKSHEET_CASES[0]: kind == retail.GIFTS_ONLY,
                    WORKSHEET_CASES[1]: kind == retail.ORIGINALS_ONLY,
                    WORKSHEET_CASES[2]: kind == retail.BOTH_GIFT_PURCHASE,
                    WORKSHEET_CASES[3]: kind == retail.BOTH_NO_GIFT_PURCHASE,
                    WORKSHEET_CASES[4]: 0 in balances,
                    WORKSHEET_CASES[5]: size == retail.B_CLASS_SIZE
                    and len(balances) == retail.B_CLASS_SIZE
                    and balances[0] == sorted(balances)[1],
                }
                for phenomenon, seen in holds.items():
                    if not seen or phenomenon in found:
                        continue
                    found[phenomenon] = {
                        "case": phenomenon,
                        "instance": instance.ordinal,
                        "side": side.upper(),
                        "case_id": case.case_id,
                        "kind": kind,
                        "gift_used": case.gift_used,
                        "instruments": _printed(case),
                        "destinations": {
                            "/".join(
                                convention[axis.name] for axis in retail.AXES
                            ): validation.audit_destination(
                                case.instruments, convention
                            )
                            for convention in retail.ALL_CONVENTIONS
                        },
                    }
    missing = [case for case in WORKSHEET_CASES if case not in found]
    if missing:
        raise ValueError(
            "a review pack carries all %d worksheet cases and this bank of %d instances "
            "exhibits %d. Missing: %s. Materialize more instances of this genre and "
            "export again"
            % (
                len(WORKSHEET_CASES),
                len(population.instances),
                len(found),
                "; ".join(missing),
            )
        )
    return [found[case] for case in WORKSHEET_CASES]


#: The equal-amount fixtures. The admitted distribution never produces a tie, on
#: purpose, so the public tie rule would otherwise reach a reader as prose nobody saw
#: applied. These are hand cases and they are labelled as such: they are not drawn from
#: the bank and no served schedule contains them.
TIE_FIXTURES = (
    ("two gift cards tied on the greatest balance",
     (("D3", retail.GIFT_CARD, 90, 10), ("D5", retail.GIFT_CARD, 90, 20),
      ("D2", retail.GIFT_CARD, 10, 30))),
    ("two gift cards tied on the least balance",
     (("D1", retail.GIFT_CARD, 70, 10), ("D6", retail.GIFT_CARD, 20, 20),
      ("D4", retail.GIFT_CARD, 20, 30))),
    ("three gift cards on one balance",
     (("D5", retail.GIFT_CARD, 50, 10), ("D1", retail.GIFT_CARD, 50, 20),
      ("D3", retail.GIFT_CARD, 50, 30))),
    ("two original contributors tied on the greatest contribution",
     (("D2", retail.CREDIT_CARD, None, 80), ("D6", retail.PAYPAL, None, 80),
      ("D1", retail.CREDIT_CARD, None, 10))),
    ("two original contributors tied on the least contribution",
     (("D4", retail.PAYPAL, None, 60), ("D3", retail.CREDIT_CARD, None, 30),
      ("D5", retail.PAYPAL, None, 30))),
    ("three original contributors on one contribution",
     (("D6", retail.CREDIT_CARD, None, 40), ("D2", retail.PAYPAL, None, 40),
      ("D4", retail.CREDIT_CARD, None, 40))),
)

#: The separating eight-case table, as a set of printed instrument lists. Its answers
#: are recomputed here by the independent selector rather than stored, so the matrix in
#: the pack is a calculation a reader can repeat and not a transcription.
MATRIX_CASES = (
    ("R1", (("D1", retail.GIFT_CARD, 20, 10), ("D2", retail.GIFT_CARD, 80, 20))),
    ("R2", (("D4", retail.GIFT_CARD, 80, 30), ("D3", retail.GIFT_CARD, 20, 10))),
    ("R3", (("D2", retail.CREDIT_CARD, None, 20), ("D5", retail.PAYPAL, None, 80))),
    ("R4", (("D6", retail.PAYPAL, None, 80), ("D1", retail.CREDIT_CARD, None, 20))),
    ("R5", (("D5", retail.GIFT_CARD, 20, 10), ("D1", retail.CREDIT_CARD, None, 20),
            ("D3", retail.GIFT_CARD, 80, 0), ("D4", retail.PAYPAL, None, 80))),
    ("R6", (("D2", retail.GIFT_CARD, 80, 0), ("D5", retail.PAYPAL, None, 80),
            ("D6", retail.GIFT_CARD, 20, 10), ("D3", retail.CREDIT_CARD, None, 20))),
    ("R7", (("D3", retail.GIFT_CARD, 20, 0), ("D6", retail.PAYPAL, None, 80),
            ("D1", retail.GIFT_CARD, 80, 0), ("D4", retail.CREDIT_CARD, None, 20))),
    ("R8", (("D4", retail.GIFT_CARD, 80, 0), ("D2", retail.CREDIT_CARD, None, 20),
            ("D5", retail.GIFT_CARD, 20, 0), ("D6", retail.PAYPAL, None, 80))),
)


def _hand_case(printed: Sequence[tuple[str, str, int | None, int]]) -> tuple[dict, ...]:
    """One hand case as the parsed records the independent selector reads."""
    return tuple(
        {
            "code": code,
            "type": kind,
            "balance": None if balance is None else balance * retail.CENTS,
            "contribution": value * retail.CENTS,
        }
        for code, kind, balance, value in printed
    )


def _hand_sheet(cases: Sequence[tuple[str, Sequence]]) -> list[dict[str, Any]]:
    """Each hand case with its destination under all 27 conventions."""
    out: list[dict[str, Any]] = []
    for name, printed in cases:
        instruments = _hand_case(printed)
        out.append(
            {
                "case": name,
                "instruments": [
                    {
                        "code": i["code"], "type": i["type"],
                        "balance": None if i["balance"] is None else "%d.%02d" % divmod(
                            i["balance"], retail.CENTS
                        ),
                        "contribution": "%d.%02d" % divmod(
                            i["contribution"], retail.CENTS
                        ),
                    }
                    for i in instruments
                ],
                "destinations": {
                    "/".join(convention[axis.name] for axis in retail.AXES):
                        validation.audit_destination(instruments, convention)
                    for convention in retail.ALL_CONVENTIONS
                },
            }
        )
    return out


def _first_with_surface(population: Population, name: str) -> tuple[Instance, str]:
    """The earliest admitted instance that draws one surface, and which side draws it."""
    for instance in population.instances:
        for side in ("a", "b"):
            if instance.side(side).surface == name:
                return instance, side
    raise ValueError(f"no admitted instance in this bank draws the surface {name!r}")


# --------------------------------------------------------------------------
# what a reader of a receipt that reports four cases of twenty four has to see
# --------------------------------------------------------------------------


def _position(convention: Mapping[str, str]) -> int:
    """Which of the 27 registered conventions this one is."""
    wanted = {axis.name: convention[axis.name] for axis in retail.AXES}
    for place, candidate in enumerate(retail.ALL_CONVENTIONS):
        if candidate == wanted:
            return place
    raise ValueError("this convention is not one of the 27 the sampler draws")


def _support(task: Task) -> list[tuple[str, ...]]:
    """This side's answer key under every convention, in the registered order."""
    return [
        tuple(retail.key_for(task.table, convention))
        for convention in retail.ALL_CONVENTIONS
    ]


def _posterior(instance: Instance, side: str) -> list[int]:
    """The conventions a reader of this side's reduced receipt cannot tell apart.

    A reported case says whether the filed code was that convention's answer and, when
    it was not, what that answer is, so two conventions print the same reported case
    exactly when their codes on it agree. The set is a fact about the mask and the two
    schedules and not about what was filed, which is why one filing's pack can show it.
    """
    task = instance.side(side)
    keys = _support(task)
    drawn = keys[_position(instance.convention)]
    return [
        n for n, key in enumerate(keys)
        if all(key[row] == drawn[row] for row in task.mask)
    ]


def _witnessed(instance: Instance, side: str) -> set[str]:
    """The axes some reported case disagrees with a one-option alternative on.

    The same event the law-level gate averages over, taken on the mask that was drawn.
    An axis with no such case is an axis this receipt says nothing about, and a mask
    that drew four cases of one public kind is not redrawn for it.
    """
    task = instance.side(side)
    keys = _support(task)
    here = instance.convention
    drawn = keys[_position(here)]
    out: set[str] = set()
    for axis in retail.AXES:
        for option in axis.options:
            if option == here[axis.name]:
                continue
            other = dict(here)
            other[axis.name] = option
            key = keys[_position(other)]
            if any(drawn[row] != key[row] for row in task.mask):
                out.add(axis.name)
    return out


def _ambiguous(instance: Instance, side: str) -> bool:
    """Whether this receipt leaves conventions that answer the sibling differently."""
    members = _posterior(instance, side)
    if len(members) < 2:
        return False
    sibling = _support(instance.side("b" if side == "a" else "a"))
    return len({sibling[n] for n in members}) > 1


def _aliased_pair(instance: Instance, side: str, members: Sequence[int]) -> tuple[int, int]:
    """Two conventions this receipt cannot tell apart whose sibling keys differ."""
    sibling = _support(instance.side("b" if side == "a" else "a"))
    for one in members:
        for other in members:
            if one < other and sibling[one] != sibling[other]:
                return (one, other)
    raise ValueError("this receipt leaves no pair with different held-out keys")


def _held_out_cost(instance: Instance, side: str) -> tuple[int, int]:
    """How many cases of the sibling the surviving conventions do not agree on."""
    members = _posterior(instance, side)
    sibling = _support(instance.side("b" if side == "a" else "a"))
    rows = len(sibling[0])
    return (
        sum(1 for row in range(rows) if len({sibling[n][row] for n in members}) > 1),
        rows,
    )


def _spelled(position: int) -> str:
    """One convention as a reader of the pack reads it."""
    convention = retail.ALL_CONVENTIONS[position]
    return ", ".join(
        "%s=%s" % (axis.name, convention[axis.name]) for axis in retail.AXES
    )


def _wrong_code(task: Task, position: int) -> str:
    """A code printed in that case which is not that case's answer under the draw."""
    correct = task.key[position]
    case = task.table.rows[position]
    return next(
        (i.code for i in case.instruments if i.code != correct), correct
    )


def _filing_over(task: Task, values: Sequence[str | None]) -> str:
    """A filing over this task's printed cases, with None for a case left out."""
    return "\n".join(
        "%s,%s" % (identifier, value)
        for identifier, value in zip(
            retail.GENERATOR.row_identifiers(task.table), values
        )
        if value is not None
    )


def _found(population: Population, holds) -> tuple[Instance, str] | None:
    """The earliest admitted instance and side the condition holds on."""
    for instance in population.instances:
        for side in ("a", "b"):
            if holds(instance, side):
                return (instance, side)
    return None


def _document(root: Path, name: str, head: Sequence[str], payload: bytes) -> str:
    """One assembled document: what a reader is being shown, then the cell itself."""
    return _write(
        root, f"{RENDERS}/{name}.txt",
        "\n".join(head) + "\n\n" + payload.decode("ascii"),
    )


def _sampled_renders(population: Population, root: Path) -> list[Render]:
    """What a reader of a receipt reporting four cases of twenty four has to see.

    NONE OF IT IS VISIBLE IN THE COVERAGE THE OTHER CATEGORIES ENUMERATE. A surface, an
    option, a filing class and a row count are all satisfied by a cell that reports
    every case, so a pack built from those alone would put a reader in front of nothing
    this policy does. These are the eleven the shared coverage asks a sampled family
    for, in this genre's own cases.

    THE FIRST FIVE ARE BUILT, NOT SEARCHED FOR. Which cases a filing gets right is the
    exporter's to choose, so they are made on the first admitted instance by filing
    against its own committed mask. THE REST ARE FACTS ABOUT THE DRAW. Whether a mask
    witnesses every axis, and whether what it leaves is one convention or several, are
    properties of the four cases the stream drew, and a mask that leaves an axis
    unwitnessed is not redrawn. A bank too small to exhibit one of those is refused
    here rather than exported with the case missing.
    """
    generator = retail.GENERATOR
    if not generator.RECEIPT_POLICY.samples:
        return []
    from shogym.envs.receipts.review import SAMPLED_CASES

    out: list[Render] = []
    first = population.instances[0]
    task = first.a
    key = list(task.key)
    mask = list(task.mask)

    def built(name: str, case: str, values: Sequence[str | None]) -> None:
        payload = _cells(
            first, task, dict(first.convention), _filing_over(task, values)
        )[GRADED]
        out.append(
            Render("sampled", case, "cell", _write(root, f"{RENDERS}/{name}.txt", payload))
        )

    built("sampled-passed", SAMPLED_CASES[0], key)
    built(
        "sampled-failed", SAMPLED_CASES[1],
        [_wrong_code(task, p) if p in mask else v for p, v in enumerate(key)],
    )
    built(
        "sampled-unfiled", SAMPLED_CASES[2],
        [None if p in mask else v for p, v in enumerate(key)],
    )
    built(
        "sampled-empty", SAMPLED_CASES[3],
        ["" if p in mask else v for p, v in enumerate(key)],
    )
    wrong_elsewhere = [
        v if p in mask else _wrong_code(task, p) for p, v in enumerate(key)
    ]
    out.append(
        Render(
            "sampled", SAMPLED_CASES[4], "document",
            _document(
                root, "sampled-suppressed",
                (
                    "instance %s/%d/%s" % (first.generator, first.ordinal, task.label),
                    "reported cases %s" % ", ".join(str(r + 1) for r in mask),
                    "cases filed with a destination the drawn convention does not "
                    "select: %s"
                    % ", ".join(str(p + 1) for p in range(len(key)) if p not in mask),
                    "every one of them failed and the cell below says nothing about "
                    "any of them",
                ),
                _cells(
                    first, task, dict(first.convention),
                    _filing_over(task, wrong_elsewhere),
                )[GRADED],
            ),
        )
    )

    axes = {axis.name for axis in retail.AXES}
    whole = _found(population, lambda i, s: _witnessed(i, s) == axes)
    if whole is None:
        raise ValueError(
            "a review pack shows a receipt whose reported cases witness every axis and "
            "this bank of %d instances draws no such mask. Materialize more instances "
            "of this genre and export again" % len(population.instances)
        )
    short = _found(population, lambda i, s: _witnessed(i, s) != axes)
    if short is None:
        raise ValueError(
            "a review pack shows a receipt that leaves an axis unwitnessed, which here "
            "is a mask whose four cases never separate one of the three choices, and "
            "every mask in this bank of %d instances witnesses all three. Materialize "
            "more instances of this genre and export again" % len(population.instances)
        )
    for name, case, (instance, side) in (
        ("sampled-witnesses-every-axis", SAMPLED_CASES[5], whole),
        ("sampled-leaves-an-axis-unwitnessed", SAMPLED_CASES[6], short),
    ):
        shown = instance.side(side)
        seen = sorted(_witnessed(instance, side))
        out.append(
            Render(
                "sampled", case, "document",
                _document(
                    root, name,
                    (
                        "instance %s/%d/%s"
                        % (instance.generator, instance.ordinal, shown.label),
                        "reported cases %s"
                        % ", ".join(str(r + 1) for r in shown.mask),
                        "axes some reported case disagrees with a one-option "
                        "alternative on: %s" % (", ".join(seen) or "none"),
                        "axes it says nothing about: %s"
                        % (", ".join(sorted(axes - set(seen))) or "none"),
                    ),
                    _cells(
                        instance, shown, dict(instance.convention), _mixed_filing(shown)
                    )[GRADED],
                ),
            )
        )

    pinned = _found(population, lambda i, s: len(_posterior(i, s)) == 1)
    if pinned is None:
        raise ValueError(
            "a review pack shows a receipt that pins the convention and no mask in "
            "this bank of %d instances does. Materialize more instances of this genre "
            "and export again" % len(population.instances)
        )
    left = _found(population, _ambiguous)
    if left is None:
        raise ValueError(
            "a review pack shows a receipt that leaves several conventions with "
            "different held-out keys and no mask in this bank of %d instances does. "
            "Materialize more instances of this genre and export again"
            % len(population.instances)
        )
    for name, case, (instance, side) in (
        ("sampled-posterior-of-one", SAMPLED_CASES[7], pinned),
        ("sampled-posterior-of-several", SAMPLED_CASES[8], left),
    ):
        shown = instance.side(side)
        members = _posterior(instance, side)
        out.append(
            Render(
                "sampled", case, "document",
                _document(
                    root, name,
                    (
                        "instance %s/%d/%s"
                        % (instance.generator, instance.ordinal, shown.label),
                        "reported cases %s"
                        % ", ".join(str(r + 1) for r in shown.mask),
                        "conventions this receipt cannot tell apart: %d of %d"
                        % (len(members), len(retail.ALL_CONVENTIONS)),
                    )
                    + tuple("  %s" % _spelled(n) for n in members)
                    + (
                        "held-out cases they disagree on: %d of %d"
                        % _held_out_cost(instance, side),
                    ),
                    _cells(
                        instance, shown, dict(instance.convention), _mixed_filing(shown)
                    )[GRADED],
                ),
            )
        )

    cells = _cells(first, task, dict(first.convention), _mixed_filing(task))
    out.append(
        Render(
            "sampled", SAMPLED_CASES[9], "document",
            _write(
                root, f"{RENDERS}/sampled-mask-commitment.txt",
                "\n".join(
                    (
                        "instance %s/%d/%s"
                        % (first.generator, first.ordinal, task.label),
                        "the receipt policy committed with this instance: %s"
                        % generator.RECEIPT_POLICY.name,
                        "the cases it drew, committed before any filing existed: %s"
                        % ", ".join(str(r + 1) for r in mask),
                        "",
                        "",
                    )
                )
                + "\n\n".join(
                    cells[kind].decode("ascii") for kind in (GRADED, PLACEBO, ORACLE)
                ),
            ),
        )
    )

    instance, side = left
    shown = instance.side(side)
    members = _posterior(instance, side)
    raw = _mixed_filing(shown)
    pair = _aliased_pair(instance, side, members)
    rendered = [
        _cells(
            instance, _retasked(shown, retail.ALL_CONVENTIONS[n]),
            dict(retail.ALL_CONVENTIONS[n]), raw,
        )[GRADED]
        for n in pair
    ]
    if rendered[0] != rendered[1]:
        raise ValueError(
            "two conventions this receipt cannot tell apart rendered two different "
            "cells, so the pack cannot show what it says it shows"
        )
    disagree, rows = _held_out_cost(instance, side)
    out.append(
        Render(
            "sampled", SAMPLED_CASES[10], "document",
            _document(
                root, "sampled-identical-reduced-receipts",
                (
                    "instance %s/%d/%s"
                    % (instance.generator, instance.ordinal, shown.label),
                    "reported cases %s" % ", ".join(str(r + 1) for r in shown.mask),
                    "these two conventions render the cell below byte for byte:",
                    "  %s" % _spelled(pair[0]),
                    "  %s" % _spelled(pair[1]),
                    "and they disagree on %d of the %d cases of the schedule this "
                    "reader files next" % (disagree, rows),
                ),
                rendered[0],
            ),
        )
    )
    return out


def export(bank: Bank, population: Population, directory: str | Path) -> Path:
    """Write the review pack and its worksheets, and return the manifest's path.

    Deterministic in the bank and its population: the same bank exports the same bytes,
    so two readers are looking at one pack and a second export is a comparison rather
    than a new document.
    """
    root = Path(directory)
    generator = retail.GENERATOR
    if bank.generator != generator.name:
        raise ValueError(
            f"this exporter writes packs for {generator.name!r} and the bank names "
            f"{bank.generator!r}"
        )
    if not population.instances:
        raise ValueError(
            "a review pack is a reading of instances and this bank holds none"
        )
    # Before anything is written, because a bank that cannot show the reader every
    # worksheet case is refused rather than exported, and a refusal that had already
    # written a directory of renders would leave a pack somebody could read as one.
    cases = _case_rows(population)
    for name in generator.surface_templates():
        _first_with_surface(population, name)
    root.mkdir(parents=True, exist_ok=True)

    renders: list[Render] = []
    first = population.instances[0]

    # ----- one task text per surface template, earliest instance that draws it -----
    for name in generator.surface_templates():
        instance, side = _first_with_surface(population, name)
        path = _write(
            root, f"{RENDERS}/surface-{name}-{instance.ordinal:04d}{side}.txt",
            instance.side(side).text,
        )
        renders.append(Render("surface", name, "task", path))

    # ----- all three cells for every registered filing class, on both siblings -----
    for shape in checks.FILING_CLASSES:
        for side in ("a", "b"):
            task = first.side(side)
            raw = checks.filing_of(generator, first, side, shape)
            cells = _cells(first, task, first.convention, raw)
            for kind in (GRADED, PLACEBO, ORACLE):
                path = _write(
                    root, f"{RENDERS}/filing-{shape}-{side}-{kind}.txt", cells[kind]
                )
                renders.append(Render("filing", shape, "cell", path))

    # ----- the row count the bank holds -----
    path = _write(
        root, f"{RENDERS}/rows-{retail.ROWS:04d}.txt",
        _cells(
            first, first.a, first.convention,
            checks.filing_of(generator, first, "a", "canonical"),
        )[GRADED],
    )
    renders.append(Render("rows", str(retail.ROWS), "cell", path))

    # ----- an A receipt of mixed verdicts, one of omissions, and the masked compare ---
    envelope = frozen_envelope(first.envelope)
    comparisons: list[dict[str, Any]] = []
    for label, raw in (
        ("mixed", _mixed_filing(first.a)), ("omissions", _omission_filing(first.a))
    ):
        cells = _cells(first, first.a, first.convention, raw)
        for kind in (GRADED, PLACEBO):
            _write(root, f"{RENDERS}/receipt-{label}-{kind}.txt", cells[kind])
        canonical = generator.parse_and_canonicalize(first.a, raw)
        ast = generator.render_receipt(
            first.a, canonical, first.a.key,
            feedback_for(generator, first.a, envelope),
        )
        ranges = slot_ranges(ast, envelope)
        comparisons.append(
            {
                "filing": label,
                "graded_digest": digest(cells[GRADED]),
                "placebo_digest": digest(cells[PLACEBO]),
                "masked_digest": digest(mask_slots(cells[GRADED], ranges)),
                "masked_bytes_equal": mask_slots(cells[GRADED], ranges)
                == mask_slots(cells[PLACEBO], ranges),
                "slot_ranges": [list(span) for span in ranges[:2]],
            }
        )

    # ----- all 27 oracle cells on one pair, each covering its three options -----
    for position, convention in enumerate(retail.ALL_CONVENTIONS):
        payload = serialize(
            generator.render_oracle(first.a.task_id, convention, first.a.n_rows),
            first.envelope,
        )
        path = _write(root, f"{RENDERS}/oracle-{position:02d}.txt", payload)
        for axis in retail.AXES:
            renders.append(
                Render("option", f"{axis.name}={convention[axis.name]}", "cell", path)
            )

    # ----- every option of every axis, rendered on every surface, with its worksheet --
    worksheets: list[dict[str, Any]] = []
    for name in generator.surface_templates():
        instance, side = _first_with_surface(population, name)
        task = instance.side(side)
        raw = _mixed_filing(task)
        drawn = dict(instance.convention)
        for axis in retail.AXES:
            for option in axis.options:
                alternative = dict(drawn, **{axis.name: option})
                retasked = _retasked(task, alternative)
                cells = _cells(instance, retasked, alternative, raw)
                path = _write(
                    root,
                    f"{RENDERS}/counterfactual-{name}-{axis.name}-{option}.txt",
                    cells[GRADED],
                )
                renders.append(
                    Render("counterfactual", "alternative convention", "cell", path)
                )
                renders.append(
                    Render("counterfactual", f"{name} {axis.name}={option}", "cell", path)
                )
                sheet = {
                    "surface": name,
                    "instance": instance.ordinal,
                    "side": side.upper(),
                    "axis_changed": axis.name,
                    "drawn": drawn,
                    "alternative": alternative,
                    "oracle_alternative": list(_oracle_sentences(alternative)),
                    "rows": _worksheet_rows(task, alternative, raw),
                }
                worksheets.append(sheet)
                _write(
                    root,
                    f"{WORKSHEETS}/{name}-{axis.name}-{option}.json",
                    json.dumps(sheet, indent=1, sort_keys=True),
                )

    _write(root, f"{WORKSHEETS}/cases.json", json.dumps(cases, indent=1, sort_keys=True))
    _write(
        root, f"{WORKSHEETS}/ties.json",
        json.dumps(
            {
                "note": (
                    "hand fixtures. The admitted distribution produces no tie, so the "
                    "public tie rule is exhibited here and in no served schedule."
                ),
                "cases": _hand_sheet(TIE_FIXTURES),
            },
            indent=1, sort_keys=True,
        ),
    )
    _write(
        root, f"{WORKSHEETS}/matrix.json",
        json.dumps(
            {
                "note": (
                    "the separating eight-case table, with every destination recomputed "
                    "here by the independent selector rather than transcribed."
                ),
                "cases": _hand_sheet(MATRIX_CASES),
                "distinct_answer_vectors": len(
                    {
                        tuple(
                            validation.audit_destination(_hand_case(printed), convention)
                            for _, printed in MATRIX_CASES
                        )
                        for convention in retail.ALL_CONVENTIONS
                    }
                ),
            },
            indent=1, sort_keys=True,
        ),
    )
    _write(
        root, f"{WORKSHEETS}/validation.json",
        json.dumps(
            {
                "note": (
                    "the validation matrix and the two copying sensitivities for the "
                    "first admitted pair. The shipped family is the registered closure "
                    "the copy bar is read against; the all-bijection family is a "
                    "separate registration over a larger family of maps."
                ),
                "instance": first.ordinal,
                "pair": validation.pair_report(generator, first),
                "cell_comparisons": comparisons,
            },
            indent=1, sort_keys=True,
        ),
    )

    renders.extend(_sampled_renders(population, root))

    coverage = required_coverage(generator, checks.FILING_CLASSES, [retail.ROWS])
    missing = coverage.missing([(r.category, r.key) for r in renders])
    if missing:
        raise ValueError("this export does not cover " + ", ".join(missing))

    manifest = {
        # UNSET, AND THE FIELD IS STILL HERE. A pack names exactly six things and this
        # names six; the one a machine cannot supply is null, so `review.verify` refuses
        # the pack by name until a person puts theirs to it.
        "reviewer": None,
        "checklist": list(CHECKLIST),
        "seeds": list(population.ordinals),
        "family": generator.name,
        "bank": bank_identity(bank),
        "renders": [
            {"category": r.category, "key": r.key, "kind": r.kind, "path": r.path}
            for r in renders
        ],
    }
    pack = root / PACK
    pack.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")

    index = {
        "family": generator.name,
        "bank": bank_identity(bank),
        "note": (
            "explanatory material for the roster release. These are not served tasks "
            "and are not part of the review manifest or of any bundle."
        ),
        "cases": [entry["case"] for entry in cases],
        "files": {
            str(path.relative_to(root)): digest(path.read_bytes())
            for path in sorted((root / WORKSHEETS).rglob("*.json"))
        },
    }
    (root / WORKSHEET_INDEX).write_text(
        json.dumps(index, indent=1, sort_keys=True), encoding="utf-8"
    )
    return pack


def read_pack(directory: str | Path) -> dict[str, Any]:
    """The exported manifest, for a caller that wants to add the attestation."""
    return json.loads((Path(directory) / PACK).read_text(encoding="utf-8"))


def attested(
    directory: str | Path, reviewer: str, checklist: Sequence[str] | None = None
) -> Path:
    """Write the same pack with a named reviewer. For a person, after they have read it.

    It is a separate call and it takes the name, because the export cannot know it and
    a default would be a machine signing for a person.
    """
    root = Path(directory)
    manifest = read_pack(root)
    manifest["reviewer"] = reviewer
    if checklist is not None:
        manifest["checklist"] = list(checklist)
    pack = root / PACK
    pack.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return pack


def read_fork(instance: Instance, side: str, raw: object):
    """One instance's three cells through the same atomic path a run uses.

    Exposed so a reader can render an instance the pack did not cover without reaching
    for a second rendering route.
    """
    return render_fork(retail.GENERATOR, instance, side, raw)


__all__ = [
    "CHECKLIST",
    "MATRIX_CASES",
    "PACK",
    "RENDERS",
    "TIE_FIXTURES",
    "WORKSHEETS",
    "WORKSHEET_CASES",
    "WORKSHEET_INDEX",
    "Render",
    "attested",
    "export",
    "read_fork",
    "read_pack",
]
