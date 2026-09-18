"""Export what a human has to have read before this genre can be released.

A person reading rendered instances is the boundary against a family that is wrong in a
way no mechanical check can see. That only holds if the person saw the family, so the
pack is enumerated from the generator's own declarations rather than chosen by whoever
exported it: every surface template, every option of every axis, every registered filing
class, the row count the bank holds, and counterfactual renders under conventions that
were not the ones drawn.

WHAT THIS ADDS BEYOND THE SHARED COVERAGE. All 36 oracle cells on one pair, so the
reader sees every sentence the oracle arm can state rather than the one this draw
produced; and a one-axis counterfactual on every surface for every axis, so the reader
can confirm that the two environments, the three losses and the two orders mean the same
thing on a comma-separated batch as on a tab-separated one.

AND A WORKSHEET, WHICH IS NOT A TASK. Beside the renders it writes a private trace
worksheet: for each counterfactual row the proto form, the form after the public nasal
pass, the form after the first hidden pass, the daughter, what was filed, the verdict
and the correction, next to the oracle's own wording. Intermediate forms and axis labels
belong there and nowhere else, and no cell prints them. The worksheet is deliberately
NOT in the review manifest and not a bundle field: it is explanatory material for the
roster release, carried with its own hash, and labelling it as a render would put a
document nobody serves inside the evidence that says what was served.

THE ATTESTATION IS LEFT UNSET. The manifest carries every field a pack carries and
names no reviewer, so `review.verify` refuses it until a person puts their name to it.
An exporter that filled that in would be a machine attesting that a human read
something.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from shogym.envs.receipts import checks
from shogym.envs.receipts.bank import Bank, Population, bank_identity, render_fork
from shogym.envs.receipts.generators import soundchange, soundchange_audit
from shogym.envs.receipts.protocol import Instance, Task
from shogym.envs.receipts.receipt_ast import GRADED, ORACLE, PLACEBO, serialize
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
    "every option of every axis means the same thing on all four surface templates",
    "the public mechanics enumerate the choices without selecting any of them",
    "the receipt names forms and never an axis, a pass or an intermediate string",
    "a failed row's correction is that row's whole daughter form, not a prefix of it",
    "the oracle states the drawn rule and adds no coaching, no worked row and no scope "
    "a reader could map onto a name printed on the batch",
    "the placebo carries no grade, no answer and nothing that reads as either",
    "the counterfactual worksheets show the stated rule producing the printed answers",
    "the two batches share no form and neither batch repeats one",
)

#: The phenomena the worksheet has to exhibit, each a question about one row.
WORKSHEET_CASES = (
    "a k with no left neighbour, which the two-sided environment refuses",
    "a k whose right vowel the deletion takes away",
    "a k whose left vowel the deletion takes away",
    "a form carrying two k occurrences",
    "a form the deletion empties of more than one vowel",
    "a form carrying the nb cluster the public pass assimilates",
    "a cluster the deletion creates, which the public pass never sees again",
)


@dataclass(frozen=True)
class Render:
    """One exported artifact and what it is evidence of."""

    category: str
    key: str
    kind: str
    path: str


def _drawn_sentences(convention: Mapping[str, str]) -> tuple[str, ...]:
    """The oracle's own wording for one convention, for the worksheet's right-hand side."""
    return soundchange.GENERATOR.render_oracle("0" * 16, convention, soundchange.ROWS).body


def _retasked(task: Task, convention: Mapping[str, str]) -> Task:
    """The same task scored under another convention, for a counterfactual render."""
    return Task(
        label=task.label,
        task_id=task.task_id,
        surface=task.surface,
        table=task.table,
        text=task.text,
        key=tuple(soundchange.key_for(task.table, convention)),
        mask=task.mask,
    )


def _cells(instance: Instance, task: Task, raw: object) -> dict[str, bytes]:
    """The three cells one filing would produce on one task, through the shared judge."""
    from shogym.envs.receipts.render import judge_cells

    generator = soundchange.GENERATOR
    canonical = generator.parse_and_canonicalize(task, raw)
    judged = judge_cells(
        generator, task, canonical, instance.convention, instance.envelope
    )
    if judged.problems:
        raise ValueError(judged.problems[0])
    return dict(judged.payloads)


def _counterfactual_cell(
    instance: Instance, task: Task, convention: Mapping[str, str], raw: object
) -> dict[str, bytes]:
    """The cells under a convention that was not the one drawn."""
    from shogym.envs.receipts.render import judge_cells

    generator = soundchange.GENERATOR
    retasked = _retasked(task, convention)
    canonical = generator.parse_and_canonicalize(retasked, raw)
    judged = judge_cells(
        generator, retasked, canonical, convention, instance.envelope
    )
    if judged.problems:
        raise ValueError(judged.problems[0])
    return dict(judged.payloads)


def _mixed_filing(task: Task, every: int = 3) -> str:
    """A filing that passes most rows and fails some, deterministically.

    Deterministic in the task's own printed order rather than drawn from a stream,
    because two readers of one pack have to be looking at one artifact and because the
    worksheet's verdict column is only worth reading if some rows failed.
    """
    lines = []
    for position, (identifier, correct) in enumerate(
        zip(soundchange.GENERATOR.row_identifiers(task.table), task.key)
    ):
        filed = correct if position % every else soundchange.nasal_pass(
            task.table.rows[position].proto
        )
        lines.append("%s,%s" % (identifier, filed))
    return "\n".join(lines)


def _write(root: Path, name: str, payload: bytes | str) -> str:
    """One exported file, returned as the path the manifest names it by."""
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        target.write_bytes(payload)
    else:
        target.write_text(payload, encoding="utf-8")
    return name


def _worksheet_rows(
    instance: Instance, task: Task, convention: Mapping[str, str], raw: str
) -> list[dict[str, str]]:
    """One row of the private worksheet per printed row, with the passes spelled out."""
    generator = soundchange.GENERATOR
    retasked = _retasked(task, convention)
    canonical = generator.parse_and_canonicalize(retasked, raw)
    _, outcomes = generator.score(retasked, canonical)
    out: list[dict[str, str]] = []
    for row, outcome in zip(task.table.rows, outcomes):
        entry = dict(soundchange_audit.intermediate_forms(row.proto, convention))
        entry.update(
            {
                "form_id": row.row_id,
                "filed": outcome.filed if outcome.was_filed else "(none)",
                "verdict": "PASS" if outcome.matched else "FAIL",
                "correction": "" if outcome.matched else outcome.correct,
            }
        )
        out.append(entry)
    return out


def _case_rows(population: Population) -> list[dict[str, Any]]:
    """The earliest row in the bank exhibiting each phenomenon the worksheet needs.

    ALL SEVEN, OR THIS BANK IS NOT ONE A PACK CAN BE EXPORTED FROM. A case no row in the
    bank exhibits used to be left out, and the pack went to the reader with the material
    for one of the seven missing while the manifest said nothing about it: a small bank
    can easily hold no form whose daughter carries a cluster the deletion created, which
    is the case that shows the public pass runs once and never sees that cluster. A
    reader confirming the stated rule produces the printed answers cannot confirm it for
    a phenomenon they were not shown, so the shortfall is the bank's and it is named
    here rather than quietly narrowing what the pack is.
    """
    found: dict[str, dict[str, Any]] = {}
    for instance in population.instances:
        if len(found) == len(WORKSHEET_CASES):
            break
        for side in ("a", "b"):
            task = instance.side(side)
            for row in task.table.rows:
                if len(found) == len(WORKSHEET_CASES):
                    break
                after = soundchange_audit.audit_nasal(row.proto)
                for convention in soundchange.ALL_CONVENTIONS:
                    vowel = soundchange_audit.removed_vowel(convention["loss"])
                    answer = soundchange_audit.audit_daughter(row.proto, convention)
                    deleted = after.count(vowel) - answer.count(vowel)
                    seen = {
                        WORKSHEET_CASES[0]: after.startswith("k"),
                        WORKSHEET_CASES[1]: any(
                            after[n + 1: n + 2] == vowel
                            for n, ch in enumerate(after) if ch == "k"
                        ),
                        WORKSHEET_CASES[2]: any(
                            after[n - 1: n] == vowel
                            for n, ch in enumerate(after) if ch == "k" and n
                        ),
                        WORKSHEET_CASES[3]: after.count("k") >= 2,
                        WORKSHEET_CASES[4]: deleted >= 2,
                        WORKSHEET_CASES[5]: "nb" in row.proto,
                        WORKSHEET_CASES[6]: "nb" in answer and "nb" not in after,
                    }
                    for case, holds in seen.items():
                        if holds and case not in found:
                            found[case] = {
                                "case": case,
                                "instance": instance.ordinal,
                                "side": side.upper(),
                                "form_id": row.row_id,
                                "convention": dict(convention),
                                "oracle": list(_drawn_sentences(convention)),
                                **soundchange_audit.intermediate_forms(
                                    row.proto, convention
                                ),
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


def _position(convention: Mapping[str, str]) -> int:
    """Where one convention sits in the declared product order."""
    drawn = dict(convention)
    return next(
        n for n, other in enumerate(soundchange.ALL_CONVENTIONS) if dict(other) == drawn
    )


def _support(task: Task) -> list[tuple[str, ...]]:
    """This side's answer key under every convention in the support, in product order."""
    return [
        tuple(soundchange.key_for(task.table, convention))
        for convention in soundchange.ALL_CONVENTIONS
    ]


def _posterior(instance: Instance, side: str) -> list[int]:
    """The conventions a reader of this side's reduced receipt cannot tell apart.

    A reported row says whether the filed value was that convention's answer and, when
    it was not, what that answer is, so two conventions print the same reported row
    exactly when their answers on it agree. The set is therefore a fact about the mask
    and the support and not about what was filed, which is why one filing's pack can
    show it.
    """
    keys = _support(instance.side(side))
    drawn = keys[_position(instance.convention)]
    return [
        n
        for n, key in enumerate(keys)
        if all(key[row] == drawn[row] for row in instance.side(side).mask)
    ]


def _witnessed(instance: Instance, side: str) -> set[str]:
    """The axes some reported row disagrees with a one-axis alternative on.

    The same event the law-level gate averages over, taken on the mask that was drawn:
    an axis with no such row is an axis this receipt says nothing about, and the mask is
    not redrawn when that happens.
    """
    task = instance.side(side)
    keys = _support(task)
    drawn = keys[_position(instance.convention)]
    out: set[str] = set()
    for axis in soundchange.AXES:
        for option in axis.options:
            if option == instance.convention[axis.name]:
                continue
            other = keys[
                _position(dict(instance.convention, **{axis.name: option}))
            ]
            if any(drawn[row] != other[row] for row in task.mask):
                out.add(axis.name)
    return out


def _filing_over(task: Task, values: Sequence[str | None]) -> str:
    """A filing over this task's printed rows, with None for a row left out entirely."""
    return "\n".join(
        "%s,%s" % (identifier, value)
        for identifier, value in zip(
            soundchange.GENERATOR.row_identifiers(task.table), values
        )
        if value is not None
    )


def _wrong(task: Task, position: int) -> str:
    """A value for one row that is not that row's answer under any convention.

    The form after the public nasal pass: the batch grammar puts a k in every form and
    every hidden cascade replaces or deletes something, so the form as it stands before
    the two hidden passes is a legal daughter-shaped string that no convention produces.
    """
    return soundchange.nasal_pass(task.table.rows[position].proto)


def _document(root: Path, name: str, head: Sequence[str], payload: bytes) -> str:
    """One assembled document: what a reader is being shown, then the cell itself."""
    return _write(
        root,
        f"{RENDERS}/{name}.txt",
        "\n".join(head) + "\n\n" + payload.decode("ascii"),
    )


def _sampled_renders(population: Population, root: Path) -> list[Render]:
    """What a reader of a receipt that reports two forms of twenty four has to see.

    NONE OF IT IS VISIBLE IN THE COVERAGE THE OTHER CATEGORIES ENUMERATE. A surface, an
    option, a filing class and a row count are all satisfied by a cell that reports every
    row, so a pack built from those alone would put a reader in front of nothing this
    policy does. These are the eleven the shared coverage asks a sampled family for, in
    this genre's own rows.

    THE FIRST FIVE ARE BUILT, NOT SEARCHED FOR. Which rows a filing gets right is the
    exporter's to choose, so the five cell cases are made on the first admitted instance
    by filing against its own committed mask: right on the reported rows, wrong on them,
    absent on them, empty on them, and wrong on every row the mask did not draw. A pack
    that waited for a bank to happen to exhibit them would be a pack whose contents moved
    with the key.

    THE REST ARE FACTS ABOUT THE DRAW AND ARE SEARCHED FOR. Whether a mask witnesses
    every axis, and whether what it leaves is one convention or several, are properties
    of the rows the stream drew; a bank too small to exhibit one of them is refused here
    rather than exported with the case missing.
    """
    generator = soundchange.GENERATOR
    if not generator.RECEIPT_POLICY.samples:
        return []
    from shogym.envs.receipts.review import SAMPLED_CASES

    out: list[Render] = []
    first = population.instances[0]
    task = first.a
    key = list(task.key)
    mask = list(task.mask)

    def built(name: str, case: str, values: Sequence[str | None]) -> None:
        payload = _cells(first, task, _filing_over(task, values))[GRADED]
        out.append(
            Render("sampled", case, "cell", _write(root, f"{RENDERS}/{name}.txt", payload))
        )

    built("sampled-passed", SAMPLED_CASES[0], key)
    built(
        "sampled-failed",
        SAMPLED_CASES[1],
        [_wrong(task, p) if p in mask else v for p, v in enumerate(key)],
    )
    built(
        "sampled-unfiled",
        SAMPLED_CASES[2],
        [None if p in mask else v for p, v in enumerate(key)],
    )
    built(
        "sampled-empty",
        SAMPLED_CASES[3],
        ["" if p in mask else v for p, v in enumerate(key)],
    )
    # A suppressed failure is the one case the cell cannot show by itself: the rows that
    # failed print the same committed tokens as the rows that passed, which is the whole
    # of what this policy does and the thing a reader is most likely to mistake for a
    # pass. So the reader is told which rows were filed wrong and can see that the cell
    # says nothing about any of them.
    wrong_elsewhere = [v if p in mask else _wrong(task, p) for p, v in enumerate(key)]
    out.append(
        Render(
            "sampled", SAMPLED_CASES[4], "document",
            _document(
                root, "sampled-suppressed",
                (
                    "instance %s/%d/%s" % (first.generator, first.ordinal, task.label),
                    "reported rows %s" % ", ".join(str(r + 1) for r in mask),
                    "rows filed with a value no cascade produces: %s"
                    % ", ".join(
                        str(p + 1) for p in range(len(key)) if p not in mask
                    ),
                    "every one of them failed and the cell below says nothing about any "
                    "of them",
                ),
                _cells(first, task, _filing_over(task, wrong_elsewhere))[GRADED],
            ),
        )
    )

    axes = tuple(axis.name for axis in soundchange.AXES)
    whole = _found(population, lambda i, s: _witnessed(i, s) == set(axes))
    if whole is None:
        raise ValueError(
            "a review pack shows a receipt whose reported rows witness every axis and "
            "this bank of %d instances draws no such mask. Materialize more instances "
            "of this genre and export again" % len(population.instances)
        )
    short = _found(population, lambda i, s: _witnessed(i, s) != set(axes))
    if short is None:
        raise ValueError(
            "a review pack shows a receipt that leaves an axis unwitnessed and every "
            "mask in this bank of %d instances witnesses all of them. Materialize more "
            "instances of this genre and export again" % len(population.instances)
        )
    for name, case, (instance, side) in (
        ("sampled-witnesses-every-axis", SAMPLED_CASES[5], whole),
        ("sampled-leaves-an-axis-unwitnessed", SAMPLED_CASES[6], short),
    ):
        held = instance.side(side)
        seen = sorted(_witnessed(instance, side))
        out.append(
            Render(
                "sampled", case, "document",
                _document(
                    root, name,
                    (
                        "instance %s/%d/%s" % (instance.generator, instance.ordinal, held.label),
                        "reported rows %s" % ", ".join(str(r + 1) for r in held.mask),
                        "axes some reported row disagrees with a one-axis alternative "
                        "on: %s" % (", ".join(seen) or "none"),
                        "axes it says nothing about: %s"
                        % (", ".join(a for a in axes if a not in seen) or "none"),
                    ),
                    _cells(instance, held, _mixed_filing(held))[GRADED],
                ),
            )
        )

    pinned = _found(population, lambda i, s: len(_posterior(i, s)) == 1)
    if pinned is None:
        raise ValueError(
            "a review pack shows a receipt that pins the cascade and no mask in this "
            "bank of %d instances does. Materialize more instances of this genre and "
            "export again" % len(population.instances)
        )
    left = _found(population, _ambiguous)
    if left is None:
        raise ValueError(
            "a review pack shows a receipt that leaves several cascades with different "
            "held-out keys and no mask in this bank of %d instances does. Materialize "
            "more instances of this genre and export again" % len(population.instances)
        )
    for name, case, (instance, side) in (
        ("sampled-posterior-of-one", SAMPLED_CASES[7], pinned),
        ("sampled-posterior-of-several", SAMPLED_CASES[8], left),
    ):
        held = instance.side(side)
        members = _posterior(instance, side)
        out.append(
            Render(
                "sampled", case, "document",
                _document(
                    root, name,
                    (
                        "instance %s/%d/%s" % (instance.generator, instance.ordinal, held.label),
                        "reported rows %s" % ", ".join(str(r + 1) for r in held.mask),
                        "cascades this receipt cannot tell apart: %d of %d"
                        % (len(members), len(soundchange.ALL_CONVENTIONS)),
                    )
                    + tuple(
                        "  %s" % _spelled(soundchange.ALL_CONVENTIONS[n]) for n in members
                    )
                    + (
                        "held-out rows they disagree on: %d of %d"
                        % _held_out_cost(instance, side),
                    ),
                    _cells(instance, held, _mixed_filing(held))[GRADED],
                ),
            )
        )

    held = first.a
    cells = _cells(first, held, _mixed_filing(held))
    out.append(
        Render(
            "sampled", SAMPLED_CASES[9], "document",
            _write(
                root, f"{RENDERS}/sampled-mask-commitment.txt",
                "\n".join(
                    (
                        "instance %s/%d/%s" % (first.generator, first.ordinal, held.label),
                        "the receipt policy committed with this instance: %s"
                        % generator.RECEIPT_POLICY.name,
                        "the rows it drew, committed before any filing existed: %s"
                        % ", ".join(str(r + 1) for r in held.mask),
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
    held = instance.side(side)
    members = _posterior(instance, side)
    raw = _mixed_filing(held)
    pair = _aliased_pair(instance, side, members)
    rendered = [
        _counterfactual_cell(instance, held, soundchange.ALL_CONVENTIONS[n], raw)[GRADED]
        for n in pair
    ]
    if rendered[0] != rendered[1]:
        raise ValueError(
            "two cascades this receipt cannot tell apart rendered two different cells, "
            "so the pack cannot show what it says it shows"
        )
    disagree, rows = _held_out_cost(instance, side)
    out.append(
        Render(
            "sampled", SAMPLED_CASES[10], "document",
            _document(
                root, "sampled-identical-reduced-receipts",
                (
                    "instance %s/%d/%s" % (instance.generator, instance.ordinal, held.label),
                    "reported rows %s" % ", ".join(str(r + 1) for r in held.mask),
                    "these two cascades render the cell below byte for byte:",
                    "  %s" % _spelled(soundchange.ALL_CONVENTIONS[pair[0]]),
                    "  %s" % _spelled(soundchange.ALL_CONVENTIONS[pair[1]]),
                    "and they disagree on %d of the %d rows of the batch this reader "
                    "files next" % (disagree, rows),
                ),
                rendered[0],
            ),
        )
    )
    return out


def _found(population: Population, holds) -> tuple[Instance, str] | None:
    """The earliest admitted instance and side the condition holds on."""
    for instance in population.instances:
        for side in ("a", "b"):
            if holds(instance, side):
                return (instance, side)
    return None


def _ambiguous(instance: Instance, side: str) -> bool:
    """Whether this receipt leaves cascades that answer the sibling differently."""
    members = _posterior(instance, side)
    if len(members) < 2:
        return False
    sibling = _support(instance.side("b" if side == "a" else "a"))
    return len({sibling[n] for n in members}) > 1


def _aliased_pair(instance: Instance, side: str, members: Sequence[int]) -> tuple[int, int]:
    """Two of the cascades this receipt cannot tell apart whose sibling keys differ."""
    sibling = _support(instance.side("b" if side == "a" else "a"))
    for one in members:
        for other in members:
            if one < other and sibling[one] != sibling[other]:
                return (one, other)
    raise ValueError("this receipt leaves no pair with different held-out keys")


def _held_out_cost(instance: Instance, side: str) -> tuple[int, int]:
    """How many rows of the sibling the surviving cascades do not agree on."""
    members = _posterior(instance, side)
    sibling = _support(instance.side("b" if side == "a" else "a"))
    rows = len(sibling[0])
    return (
        sum(1 for row in range(rows) if len({sibling[n][row] for n in members}) > 1),
        rows,
    )


def _spelled(convention: Mapping[str, str]) -> str:
    """One cascade as a reader of the pack reads it."""
    return ", ".join("%s=%s" % (axis.name, convention[axis.name]) for axis in soundchange.AXES)


def export(bank: Bank, population: Population, directory: str | Path) -> Path:
    """Write the review pack and its worksheets, and return the manifest's path.

    Deterministic in the bank and its population: the same bank exports the same bytes,
    so two readers are looking at one pack and a second export is a comparison rather
    than a new document. The coverage is taken from the whole population, earliest
    admitted instance first, and the options no drawn convention realized get
    counterfactual renders built for them rather than being left out.
    """
    root = Path(directory)
    generator = soundchange.GENERATOR
    if bank.generator != generator.name:
        raise ValueError(
            f"this exporter writes packs for {generator.name!r} and the bank names "
            f"{bank.generator!r}"
        )
    if not population.instances:
        raise ValueError("a review pack is a reading of instances and this bank holds none")
    # Before anything is written, because a bank that cannot show the reader every
    # worksheet case is refused rather than exported, and a refusal that had already
    # written a directory of renders would leave a pack somebody could read as one.
    cases = _case_rows(population)
    root.mkdir(parents=True, exist_ok=True)

    renders: list[Render] = []
    first = population.instances[0]

    # ----- one task text per surface template, earliest instance that draws it -----
    for name in generator.surface_templates():
        for instance in population.instances:
            for side in ("a", "b"):
                task = instance.side(side)
                if task.surface != name:
                    continue
                path = _write(
                    root, f"{RENDERS}/surface-{name}-{instance.ordinal:04d}{side}.txt",
                    task.text,
                )
                renders.append(Render("surface", name, "task", path))
                break
            if renders and renders[-1].key == name:
                break

    # ----- all three cells for every registered filing class, on both siblings -----
    for shape in checks.FILING_CLASSES:
        for side in ("a", "b"):
            task = first.side(side)
            raw = checks.filing_of(generator, first, side, shape)
            cells = _cells(first, task, raw)
            for kind in (GRADED, PLACEBO, ORACLE):
                path = _write(
                    root,
                    f"{RENDERS}/filing-{shape}-{side}-{kind}.txt",
                    cells[kind],
                )
                renders.append(Render("filing", shape, "cell", path))

    # ----- the row count the bank holds -----
    path = _write(
        root, f"{RENDERS}/rows-{soundchange.ROWS:04d}.txt",
        _cells(first, first.a, checks.filing_of(generator, first, "a", "canonical"))[GRADED],
    )
    renders.append(Render("rows", str(soundchange.ROWS), "cell", path))

    # ----- all 36 oracle cells on one pair, each covering its four options -----
    envelope = first.envelope
    for position, convention in enumerate(soundchange.ALL_CONVENTIONS):
        payload = serialize(
            generator.render_oracle(first.a.task_id, convention, first.a.n_rows), envelope
        )
        path = _write(root, f"{RENDERS}/oracle-{position:02d}.txt", payload)
        for axis in soundchange.AXES:
            renders.append(
                Render("option", f"{axis.name}={convention[axis.name]}", "cell", path)
            )

    # ----- one counterfactual per surface per axis, and the worksheets beside them ---
    worksheets: list[dict[str, Any]] = []
    for name in generator.surface_templates():
        instance, side = _first_with_surface(population, name)
        task = instance.side(side)
        raw = _mixed_filing(task)
        drawn = dict(instance.convention)
        for axis in soundchange.AXES:
            other = next(o for o in axis.options if o != drawn[axis.name])
            alternative = dict(drawn, **{axis.name: other})
            cells = _counterfactual_cell(instance, task, alternative, raw)
            path = _write(
                root,
                f"{RENDERS}/counterfactual-{name}-{axis.name}-{other}.txt",
                cells[GRADED],
            )
            renders.append(Render("counterfactual", "alternative convention", "cell", path))
            renders.append(
                Render("counterfactual", f"{name} {axis.name}={other}", "cell", path)
            )
            sheet = {
                "surface": name,
                "instance": instance.ordinal,
                "side": side.upper(),
                "axis_changed": axis.name,
                "drawn": drawn,
                "alternative": alternative,
                "oracle_drawn": list(_drawn_sentences(drawn)),
                "oracle_alternative": list(_drawn_sentences(alternative)),
                "rows": _worksheet_rows(instance, task, alternative, raw),
            }
            worksheets.append(sheet)
            _write(
                root,
                f"{WORKSHEETS}/{name}-{axis.name}-{other}.json",
                json.dumps(sheet, indent=1, sort_keys=True),
            )

    _write(
        root, f"{WORKSHEETS}/cases.json", json.dumps(cases, indent=1, sort_keys=True)
    )

    renders.extend(_sampled_renders(population, root))

    coverage = required_coverage(
        generator, checks.FILING_CLASSES, [soundchange.ROWS]
    )
    seen = [(r.category, r.key) for r in renders]
    missing = coverage.missing(seen)
    if missing:
        raise ValueError(
            "this export does not cover " + ", ".join(missing)
        )

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


def _first_with_surface(population: Population, name: str) -> tuple[Instance, str]:
    """The earliest admitted instance that draws one surface, and which side draws it."""
    for instance in population.instances:
        for side in ("a", "b"):
            if instance.side(side).surface == name:
                return instance, side
    raise ValueError(f"no admitted instance in this bank draws the surface {name!r}")


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
    return render_fork(soundchange.GENERATOR, instance, side, raw)


__all__ = [
    "CHECKLIST",
    "PACK",
    "RENDERS",
    "WORKSHEETS",
    "WORKSHEET_CASES",
    "WORKSHEET_INDEX",
    "Render",
    "attested",
    "export",
    "read_fork",
    "read_pack",
]
