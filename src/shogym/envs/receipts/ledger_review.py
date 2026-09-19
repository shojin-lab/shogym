"""Export what a human has to have read before the ledger genre can be released.

A person reading rendered instances is the boundary against a family that is wrong in a
way no mechanical check can see. That only holds if the person saw the family, so the
pack is enumerated from the generator's own declarations rather than chosen by whoever
exported it: every surface template, every option of every axis, every registered filing
class, the row counts the bank holds, and counterfactual renders under conventions that
were not the one drawn.

WHAT THIS ADDS BEYOND THE SHARED COVERAGE. All 72 oracle cells on one pair, so the reader
sees every sentence the oracle arm can state rather than the four this draw produced; and
a one-axis counterfactual graded cell on both siblings for every axis, so the reader can
confirm that an anchor, a basis, a boundary and a missing-record option mean the same
thing on a pipe-delimited backlog as on a comma-separated schedule.

AND THE ELEVEN A SAMPLED RECEIPT OWES. Ledger's receipt reports four records of twenty
four, so a reader shown one cell has been shown nothing about twenty of its rows. The
shared coverage asks a sampled family for eleven more renders and this writes them in this
genre's own words: records and bands. The first five are BUILT, by filing against the
instance's own committed mask, because which records a filing gets right is the exporter's
to choose and a pack that waited for a bank to happen to exhibit them would be a pack whose
contents moved with the key. The rest are facts about the draw and are SEARCHED for, each
taken on the earliest admitted instance and side that exhibits it.

A BANK THAT CANNOT EXHIBIT ONE OF THEM IS REFUSED BY NAME, AND BEFORE ANYTHING IS
WRITTEN. Every search runs before the directory is made, so a refusal leaves no half
written pack for somebody to read as one, and the message names the case that was missing
rather than narrowing what a pack is.

NO CONVENTION IS INVENTED HERE. Every convention this module renders under is a member of
`ledger.ALL_CONVENTIONS`, reached by its position in the declared product order, and
`_position` refuses anything else. The value a counterfactual files on a wrong row is
chosen against the whole support rather than against the drawn rule, so the same table
files the same way whichever convention the bank drew.

AND A WORKSHEET, WHICH IS NOT A TASK. Beside the renders it writes a private worksheet:
for each counterfactual row the three dates, the day count and the band under the drawn
convention and under the alternative, what was filed, the verdict and the correction, next
to the oracle's own wording for both. Day counts and axis labels belong there and nowhere
else, and no cell prints them. The worksheet is deliberately NOT in the review manifest and
not a bundle field: it is explanatory material for the roster release, carried with its own
hash, and labelling it as a render would put a document nobody serves inside the evidence
that says what was served.

THE ATTESTATION IS LEFT UNSET. The manifest carries every field a pack carries and names no
reviewer, so `review.verify` refuses it until a person puts their name to it. An exporter
that filled that in would be a machine attesting that a human read something.

THE SCREEN PROCEDURE IS AUDITED HERE AND NOT RUN HERE. `screen_refusals` holds a recorded
allocation against the registered procedure, which is the roster's and not this genre's:
twelve reserved exploratory identities, the next thirty six as the final screen, case j to
state j mod 4, nine per state, the common standing instruction, one frozen instrument, one
named model, four pinned initial states, and the predeclared release condition of a mean
oracle grade of at least 0.90. The identities are read against the bank's own admitted
order and the outcomes against the recorded screen the pilot wrote, because an allocation
audited against its own fields is an allocation that says whatever it likes. No model has
been run for this genre and nothing here runs one.

THE ALLOCATION ITSELF IS ONE FUNCTION FOR EVERY GENRE. `screen_allocation` is imported
rather than written again: the reservation, the case count and the state rule are the
roster's registered procedure, and a second copy of a registered number is one of them
going stale. The refusals around it are still this module's, because the audit names the
family it audits, and folding the two genres' refusals into one shared audit is left for
whoever next touches both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from shogym.envs.receipts import checks
from shogym.envs.receipts.bank import Bank, Population, bank_identity, render_fork
from shogym.envs.receipts.components_review import (
    CASES_PER_STATE,
    EXPLORATORY_PREFIX,
    FINAL_CASES,
    INITIAL_STATES,
    MIN_MEAN_ORACLE,
    STANDING_INSTRUCTION,
    screen_allocation,
)
from shogym.envs.receipts.generators import ledger
from shogym.envs.receipts.protocol import Instance, Task
from shogym.envs.receipts.receipt_ast import GRADED, ORACLE, PLACEBO, serialize
from shogym.envs.receipts.review import SAMPLED_CASES, required_coverage
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
    "the policy extract states the band table and the holidays and decides none of the "
    "four open questions",
    "every option of every axis means the same thing on all eight surface templates, "
    "whatever the delimiter and whatever the band vocabulary",
    "the receipt names records and bands and never an axis, a date column or a day count",
    "a failed record's correction is that record's own band and nothing else",
    "the oracle states the drawn convention and adds no coaching, no worked record and no "
    "scope a reader could map onto the organisation printed on the schedule",
    "the placebo carries no grade, no answer and nothing that reads as either",
    "the worksheets show the stated convention producing the printed bands, holidays and "
    "weekends included",
    "the reported records are the ones the committed mask drew, and the lines for the "
    "other records say nothing about whether they were right",
    "the two schedules share no record identifier and neither repeats one",
)

#: Band-shaped words no domain in this genre prints, for a row a counterfactual files
#: wrongly. A wrong value has to be wrong under EVERY convention, or a filing meant to
#: fail would pass under some member of the support and the render would say the opposite
#: of what the pack claims it says. These are the words the bank's own review filings
#: already use, so a reader of two packs reads one vocabulary.
OFF_BAND = ("Provisional", "Deferred", "Held", "Cleared")


@dataclass(frozen=True)
class Render:
    """One exported artifact and what it is evidence of."""

    category: str
    key: str
    kind: str
    path: str


@dataclass(frozen=True)
class Selected:
    """Everything the export needs a bank to exhibit, resolved before anything is written.

    The five cell cases are not here: they are built on the first admitted instance from
    its own committed mask, so no bank can fail to hold them. What is here is the material
    a search either finds or does not, and a bank that does not hold it is refused by name.
    """

    first: Instance
    surfaces: tuple[tuple[str, int, str], ...]
    counts: tuple[tuple[int, int, str], ...]
    whole: tuple[Instance, str]
    short: tuple[Instance, str]
    pinned: tuple[Instance, str]
    left: tuple[Instance, str]
    aliased: tuple[int, int]


# --------------------------------------------------------------------------
# the declared support, and nothing outside it
# --------------------------------------------------------------------------


def _position(convention: Mapping[str, str]) -> int:
    """Where one convention sits in the declared product order, refused when it is not in it.

    Refused rather than appended. Every render this module writes is a render under a
    registered convention, so a convention assembled here that the family does not draw
    would be a cell in a review pack that no bank member could ever serve.
    """
    drawn = dict(convention)
    for n, other in enumerate(ledger.ALL_CONVENTIONS):
        if dict(other) == drawn:
            return n
    raise ValueError(
        "this exporter renders the conventions this genre declares and %r is not one of "
        "them" % (drawn,)
    )


def _spelled(convention: Mapping[str, str]) -> str:
    """One convention as a reader of the pack reads it."""
    return ", ".join("%s=%s" % (axis.name, convention[axis.name]) for axis in ledger.AXES)


def _support(task: Task) -> list[tuple[str, ...]]:
    """This side's answer key under every convention in the support, in product order."""
    return [
        tuple(ledger.key_for(task.table, convention))
        for convention in ledger.ALL_CONVENTIONS
    ]


def _retasked(task: Task, convention: Mapping[str, str]) -> Task:
    """The same task scored under another convention, for a counterfactual render."""
    return Task(
        label=task.label,
        task_id=task.task_id,
        surface=task.surface,
        table=task.table,
        text=task.text,
        key=tuple(ledger.key_for(task.table, convention)),
        mask=task.mask,
    )


def _cells(
    instance: Instance, task: Task, convention: Mapping[str, str], raw: object
) -> dict[str, bytes]:
    """The three cells one filing would produce on one task, through the shared judge."""
    from shogym.envs.receipts.render import judge_cells

    _position(convention)
    generator = ledger.GENERATOR
    canonical = generator.parse_and_canonicalize(task, raw)
    judged = judge_cells(generator, task, canonical, convention, instance.envelope)
    if judged.problems:
        raise ValueError(judged.problems[0])
    return dict(judged.payloads)


def _counterfactual_cells(
    instance: Instance, task: Task, convention: Mapping[str, str], raw: object
) -> dict[str, bytes]:
    """The cells under a convention that was not the one drawn."""
    return _cells(instance, _retasked(task, convention), convention, raw)


# --------------------------------------------------------------------------
# filings the pack files for itself
# --------------------------------------------------------------------------


def _wrong(task: Task, position: int) -> str:
    """A band for one record that is not that record's answer under any convention.

    Taken against the whole declared support, so it is wrong whichever convention the bank
    drew and the same table files the same way in every pack.
    """
    answers = {key[position] for key in _support(task)}
    for value in OFF_BAND:
        if value not in answers:
            return value
    raise ValueError(
        "every band this exporter files on a wrong record is an answer to record %d under "
        "some convention, so nothing here can make that record fail" % (position + 1)
    )


def _filing_over(task: Task, values: Sequence[str | None]) -> str:
    """A filing over this task's printed records, with None for one left out entirely."""
    return "\n".join(
        "%s,%s" % (identifier, value)
        for identifier, value in zip(
            ledger.GENERATOR.row_identifiers(task.table), values
        )
        if value is not None
    )


def _mixed_filing(task: Task, every: int = 3) -> str:
    """A filing that gets most records right and some wrong, deterministically.

    Deterministic in the task's own printed order rather than drawn from a stream, because
    two readers of one pack have to be looking at one artifact and because the verdict
    column is only worth reading if some records failed.
    """
    key = list(task.key)
    return _filing_over(
        task,
        [
            _wrong(task, position) if position % every == 0 else value
            for position, value in enumerate(key)
        ],
    )


# --------------------------------------------------------------------------
# what a mask leaves
# --------------------------------------------------------------------------


def _posterior(instance: Instance, side: str) -> list[int]:
    """The conventions a reader of this side's reduced receipt cannot tell apart.

    A reported record says whether the filed band was that convention's answer and, when it
    was not, what that answer is, so two conventions print the same reported record exactly
    when their answers on it agree. The set is therefore a fact about the mask and the
    support and not about what was filed, which is why one filing's pack can show it.
    """
    task = instance.side(side)
    keys = _support(task)
    drawn = keys[_position(instance.convention)]
    return [
        n for n, key in enumerate(keys) if all(key[row] == drawn[row] for row in task.mask)
    ]


def _witnessed(instance: Instance, side: str) -> set[str]:
    """The axes some reported record disagrees with a one-axis alternative on.

    The same event the law-level gate averages over, taken on the mask that was drawn: an
    axis with no such record is an axis this receipt says nothing about, and the mask is
    not redrawn when that happens. On this genre the missing-record axis is witnessed by a
    reported record with no dates and by nothing else, so an unwitnessed `missing` is a
    receipt that reported no such record.
    """
    task = instance.side(side)
    keys = _support(task)
    drawn = keys[_position(instance.convention)]
    out: set[str] = set()
    for axis in ledger.AXES:
        for option in axis.options:
            if option == instance.convention[axis.name]:
                continue
            other = keys[_position(dict(instance.convention, **{axis.name: option}))]
            if any(drawn[row] != other[row] for row in task.mask):
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
    """Two of the conventions this receipt cannot tell apart whose sibling keys differ."""
    sibling = _support(instance.side("b" if side == "a" else "a"))
    for one in members:
        for other in members:
            if one < other and sibling[one] != sibling[other]:
                return (one, other)
    raise ValueError("this receipt leaves no pair with different held-out keys")


def _held_out_cost(instance: Instance, side: str) -> tuple[int, int]:
    """How many records of the sibling the surviving conventions do not agree on."""
    members = _posterior(instance, side)
    sibling = _support(instance.side("b" if side == "a" else "a"))
    rows = len(sibling[0])
    return (
        sum(1 for row in range(rows) if len({sibling[n][row] for n in members}) > 1),
        rows,
    )


def _found(
    population: Population, holds: Callable[[Instance, str], bool]
) -> tuple[Instance, str] | None:
    """The earliest admitted instance and side the condition holds on."""
    for instance in population.instances:
        for side in ("a", "b"):
            if holds(instance, side):
                return (instance, side)
    return None


def _refused(case: str, population: Population) -> ValueError:
    """One refusal, naming the case the bank does not exhibit."""
    return ValueError(
        "a review pack shows %r and this bank of %d instances exhibits no such receipt. "
        "Materialize more instances of this genre and export again"
        % (case, len(population.instances))
    )


def _selected(population: Population) -> Selected:
    """Resolve everything the export has to find, or refuse naming what is missing.

    BEFORE ANYTHING IS WRITTEN. A refusal that had already written a directory of renders
    would leave a pack somebody could read as one, and a pack quietly narrowed to what a
    small bank happened to hold is a reader confirming a family they were not shown.
    """
    if not population.instances:
        raise ValueError("a review pack is a reading of instances and this bank holds none")
    generator = ledger.GENERATOR

    surfaces: list[tuple[str, int, str]] = []
    for name in generator.surface_templates():
        drawn = _found(population, lambda i, s, n=name: i.side(s).surface == n)
        if drawn is None:
            raise ValueError(
                "a review pack shows every surface this genre draws and no instance in "
                "this bank of %d draws %r. Materialize more instances of this genre and "
                "export again" % (len(population.instances), name)
            )
        surfaces.append((name, drawn[0].ordinal, drawn[1]))

    counts: list[tuple[int, int, str]] = []
    seen: set[int] = set()
    for instance in population.instances:
        for label in ("a", "b"):
            count = instance.side(label).n_rows
            if count not in seen:
                seen.add(count)
                counts.append((count, instance.ordinal, label))
    counts.sort()

    axes = {axis.name for axis in ledger.AXES}
    whole = _found(population, lambda i, s: _witnessed(i, s) == axes)
    if whole is None:
        raise _refused(SAMPLED_CASES[5], population)
    short = _found(population, lambda i, s: _witnessed(i, s) != axes)
    if short is None:
        raise _refused(SAMPLED_CASES[6], population)
    pinned = _found(population, lambda i, s: len(_posterior(i, s)) == 1)
    if pinned is None:
        raise _refused(SAMPLED_CASES[7], population)
    left = _found(population, _ambiguous)
    if left is None:
        raise _refused(SAMPLED_CASES[8], population)
    aliased = _aliased_pair(left[0], left[1], _posterior(left[0], left[1]))
    return Selected(
        first=population.instances[0],
        surfaces=tuple(surfaces),
        counts=tuple(counts),
        whole=whole,
        short=short,
        pinned=pinned,
        left=left,
        aliased=aliased,
    )


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _write(root: Path, name: str, payload: bytes | str) -> str:
    """One exported file, returned as the path the manifest names it by."""
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        target.write_bytes(payload)
    else:
        target.write_text(payload, encoding="utf-8")
    return name


def _document(root: Path, name: str, head: Sequence[str], payload: bytes) -> str:
    """One assembled document: what a reader is being shown, then the cell itself."""
    return _write(
        root, f"{RENDERS}/{name}.txt", "\n".join(head) + "\n\n" + payload.decode("ascii")
    )


def _sampled_renders(chosen: Selected, root: Path) -> list[Render]:
    """What a reader of a receipt that reports four records of twenty four has to see.

    NONE OF IT IS VISIBLE IN THE COVERAGE THE OTHER CATEGORIES ENUMERATE. A surface, an
    option, a filing class and a row count are all satisfied by a cell that reports every
    record, so a pack built from those alone would put a reader in front of nothing this
    policy does.
    """
    generator = ledger.GENERATOR
    if not generator.RECEIPT_POLICY.samples:
        return []

    out: list[Render] = []
    first = chosen.first
    task = first.a
    key = list(task.key)
    mask = list(task.mask)
    convention = first.convention

    def built(name: str, case: str, values: Sequence[str | None]) -> None:
        payload = _cells(first, task, convention, _filing_over(task, values))[GRADED]
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

    # A suppressed failure is the one case the cell cannot show by itself: the records that
    # failed print the same committed tokens as the records that passed, which is the whole
    # of what this policy does and the thing a reader is most likely to mistake for a pass.
    # So the reader is told which records were filed wrong and can see that the cell says
    # nothing about any of them.
    elsewhere = [v if p in mask else _wrong(task, p) for p, v in enumerate(key)]
    out.append(
        Render(
            "sampled",
            SAMPLED_CASES[4],
            "document",
            _document(
                root,
                "sampled-suppressed",
                (
                    "instance %s/%d/%s" % (first.generator, first.ordinal, task.label),
                    "reported records %s" % ", ".join(str(r + 1) for r in mask),
                    "records filed with a band no convention gives them: %s"
                    % ", ".join(str(p + 1) for p in range(len(key)) if p not in mask),
                    "every one of them failed and the cell below says nothing about any "
                    "of them",
                ),
                _cells(first, task, convention, _filing_over(task, elsewhere))[GRADED],
            ),
        )
    )

    axes = tuple(axis.name for axis in ledger.AXES)
    for name, case, (instance, side) in (
        ("sampled-witnesses-every-axis", SAMPLED_CASES[5], chosen.whole),
        ("sampled-leaves-an-axis-unwitnessed", SAMPLED_CASES[6], chosen.short),
    ):
        held = instance.side(side)
        seen = sorted(_witnessed(instance, side))
        out.append(
            Render(
                "sampled",
                case,
                "document",
                _document(
                    root,
                    name,
                    (
                        "instance %s/%d/%s"
                        % (instance.generator, instance.ordinal, held.label),
                        "reported records %s" % ", ".join(str(r + 1) for r in held.mask),
                        "axes some reported record disagrees with a one-axis alternative "
                        "on: %s" % (", ".join(seen) or "none"),
                        "axes it says nothing about: %s"
                        % (", ".join(a for a in axes if a not in seen) or "none"),
                    ),
                    _cells(
                        instance, held, instance.convention, _mixed_filing(held)
                    )[GRADED],
                ),
            )
        )

    for name, case, (instance, side) in (
        ("sampled-posterior-of-one", SAMPLED_CASES[7], chosen.pinned),
        ("sampled-posterior-of-several", SAMPLED_CASES[8], chosen.left),
    ):
        held = instance.side(side)
        members = _posterior(instance, side)
        out.append(
            Render(
                "sampled",
                case,
                "document",
                _document(
                    root,
                    name,
                    (
                        "instance %s/%d/%s"
                        % (instance.generator, instance.ordinal, held.label),
                        "reported records %s" % ", ".join(str(r + 1) for r in held.mask),
                        "conventions this receipt cannot tell apart: %d of %d"
                        % (len(members), len(ledger.ALL_CONVENTIONS)),
                    )
                    + tuple(
                        "  %s" % _spelled(ledger.ALL_CONVENTIONS[n]) for n in members
                    )
                    + (
                        "held-out records they disagree on: %d of %d"
                        % _held_out_cost(instance, side),
                    ),
                    _cells(
                        instance, held, instance.convention, _mixed_filing(held)
                    )[GRADED],
                ),
            )
        )

    cells = _cells(first, task, convention, _mixed_filing(task))
    out.append(
        Render(
            "sampled",
            SAMPLED_CASES[9],
            "document",
            _write(
                root,
                f"{RENDERS}/sampled-mask-commitment.txt",
                "\n".join(
                    (
                        "instance %s/%d/%s" % (first.generator, first.ordinal, task.label),
                        "the receipt policy committed with this instance: %s"
                        % generator.RECEIPT_POLICY.name,
                        "the records it drew, committed before any filing existed: %s"
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

    instance, side = chosen.left
    held = instance.side(side)
    raw = _mixed_filing(held)
    rendered = [
        _counterfactual_cells(instance, held, ledger.ALL_CONVENTIONS[n], raw)[GRADED]
        for n in chosen.aliased
    ]
    if rendered[0] != rendered[1]:
        raise ValueError(
            "two conventions this receipt cannot tell apart rendered two different cells, "
            "so the pack cannot show what it says it shows"
        )
    disagree, rows = _held_out_cost(instance, side)
    out.append(
        Render(
            "sampled",
            SAMPLED_CASES[10],
            "document",
            _document(
                root,
                "sampled-identical-reduced-receipts",
                (
                    "instance %s/%d/%s" % (instance.generator, instance.ordinal, held.label),
                    "reported records %s" % ", ".join(str(r + 1) for r in held.mask),
                    "these two conventions render the cell below byte for byte:",
                    "  %s" % _spelled(ledger.ALL_CONVENTIONS[chosen.aliased[0]]),
                    "  %s" % _spelled(ledger.ALL_CONVENTIONS[chosen.aliased[1]]),
                    "and they disagree on %d of the %d records of the schedule this reader "
                    "files next" % (disagree, rows),
                ),
                rendered[0],
            ),
        )
    )
    return out


def _worksheet_rows(
    task: Task, drawn: Mapping[str, str], alternative: Mapping[str, str], raw: object
) -> list[dict[str, Any]]:
    """One row of the private worksheet per printed record, with the count spelled out."""
    generator = ledger.GENERATOR
    retasked = _retasked(task, alternative)
    canonical = generator.parse_and_canonicalize(retasked, raw)
    _, outcomes = generator.score(retasked, canonical)
    table = task.table
    dom = table.dom
    under_drawn = ledger.key_for(table, drawn)
    under_other = ledger.key_for(table, alternative)
    out: list[dict[str, Any]] = []
    for position, (row, outcome) in enumerate(zip(table.rows, outcomes)):
        out.append(
            {
                "record": row.row_id,
                "dates": (
                    {}
                    if row.dates is None
                    else {role: row.dates[role].isoformat() for role in ledger.ROLES}
                ),
                "count_drawn": (
                    ""
                    if row.dates is None
                    else ledger.daycount(
                        row.dates[drawn["anchor"]],
                        dom["refdate"],
                        drawn["basis"],
                        table.holidays,
                    )
                ),
                "count_alternative": (
                    ""
                    if row.dates is None
                    else ledger.daycount(
                        row.dates[alternative["anchor"]],
                        dom["refdate"],
                        alternative["basis"],
                        table.holidays,
                    )
                ),
                "band_drawn": under_drawn[position],
                "band_alternative": under_other[position],
                "filed": outcome.filed if outcome.was_filed else ledger.UNFILED_TOKEN,
                "verdict": "PASS" if outcome.matched else "FAIL",
                "correction": "" if outcome.matched else outcome.correct,
            }
        )
    return out


def _sentences(convention: Mapping[str, str]) -> list[str]:
    """The oracle's own wording for one convention, for the worksheet's right-hand side."""
    return list(
        ledger.GENERATOR.render_oracle("0" * 16, convention, ledger.SHAPE.rows).body
    )


def export(bank: Bank, population: Population, directory: str | Path) -> Path:
    """Write the review pack and its worksheets, and return the manifest's path.

    Deterministic in the bank and its population: the same bank exports the same bytes, so
    two readers are looking at one pack and a second export is a comparison rather than a
    new document. The coverage is taken from the whole population, earliest admitted
    instance first, and the options no drawn convention realized get counterfactual renders
    built for them rather than being left out.
    """
    root = Path(directory)
    generator = ledger.GENERATOR
    if bank.generator != generator.name:
        raise ValueError(
            f"this exporter writes packs for {generator.name!r} and the bank names "
            f"{bank.generator!r}"
        )
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            f"{root} is not empty. A review pack is what a person read, and a directory "
            "holding an earlier export as well has no single answer to what that was"
        )
    chosen = _selected(population)
    root.mkdir(parents=True, exist_ok=True)

    renders: list[Render] = []
    first = chosen.first
    drawn = dict(first.convention)

    # ----- one task text per surface template, earliest instance that draws it -----
    for name, ordinal, side in chosen.surfaces:
        task = population.instance(ordinal).side(side)
        path = _write(root, f"{RENDERS}/surface-{name}-{ordinal:04d}{side}.txt", task.text)
        renders.append(Render("surface", name, "task", path))

    # ----- all three cells for every registered filing class, on both siblings -----
    for shape in checks.FILING_CLASSES:
        for side in ("a", "b"):
            task = first.side(side)
            raw = checks.filing_of(generator, first, side, shape)
            cells = _cells(first, task, first.convention, raw)
            for kind in (GRADED, PLACEBO, ORACLE):
                path = _write(root, f"{RENDERS}/filing-{shape}-{side}-{kind}.txt", cells[kind])
                renders.append(Render("filing", shape, "cell", path))

    # ----- the row counts the bank holds -----
    for count, ordinal, side in chosen.counts:
        instance = population.instance(ordinal)
        raw = checks.filing_of(generator, instance, side, "canonical")
        payload = _cells(instance, instance.side(side), instance.convention, raw)[GRADED]
        path = _write(root, f"{RENDERS}/rows-{count:04d}.txt", payload)
        renders.append(Render("rows", str(count), "cell", path))

    # ----- all 72 oracle cells on one pair, each covering its four options -----
    #
    # The whole declared support, because the oracle arm can state any of them and a reader
    # shown the four sentences this draw produced has read one convention out of seventy
    # two. The cells under a convention that was not drawn are review counterfactuals and
    # are labelled as such, so nothing here reads as a bank member.
    for position, convention in enumerate(ledger.ALL_CONVENTIONS):
        payload = serialize(
            generator.render_oracle(first.a.task_id, convention, first.a.n_rows),
            first.envelope,
        )
        path = _write(root, f"{RENDERS}/oracle-{position:02d}.txt", payload)
        for axis in ledger.AXES:
            renders.append(
                Render("option", f"{axis.name}={convention[axis.name]}", "cell", path)
            )
        if dict(convention) != drawn:
            renders.append(
                Render(
                    "counterfactual",
                    "review counterfactual: the oracle under %s" % _spelled(convention),
                    "cell",
                    path,
                )
            )

    # ----- one counterfactual graded cell per axis on each sibling, and the worksheets ---
    worksheets: list[dict[str, Any]] = []
    counterfactuals: list[dict[str, Any]] = []
    for side in ("a", "b"):
        task = first.side(side)
        raw = _mixed_filing(task)
        for axis in ledger.AXES:
            other = next(o for o in axis.options if o != drawn[axis.name])
            alternative = dict(drawn, **{axis.name: other})
            cells = _counterfactual_cells(first, task, alternative, raw)
            path = _write(
                root, f"{RENDERS}/counterfactual-{side}-{axis.name}-{other}.txt", cells[GRADED]
            )
            renders.append(Render("counterfactual", "alternative convention", "cell", path))
            renders.append(
                Render(
                    "counterfactual",
                    "review counterfactual under %s=%s, side %s"
                    % (axis.name, other, side.upper()),
                    "cell",
                    path,
                )
            )
            counterfactuals.append(
                {
                    "path": path,
                    "side": side.upper(),
                    "axis_changed": axis.name,
                    "drawn": drawn,
                    "alternative": alternative,
                }
            )
            sheet = {
                "instance": first.ordinal,
                "side": side.upper(),
                "surface": task.surface,
                "axis_changed": axis.name,
                "drawn": drawn,
                "alternative": alternative,
                "reference_date": task.table.dom["refdate"].isoformat(),
                "holidays": [day.isoformat() for day in task.table.holidays],
                "reported_records": [row + 1 for row in task.mask],
                "oracle_drawn": _sentences(drawn),
                "oracle_alternative": _sentences(alternative),
                "rows": _worksheet_rows(task, drawn, alternative, raw),
            }
            worksheets.append(sheet)
            _write(
                root,
                f"{WORKSHEETS}/{side}-{axis.name}-{other}.json",
                json.dumps(sheet, indent=1, sort_keys=True),
            )

    renders.extend(_sampled_renders(chosen, root))

    coverage = required_coverage(
        generator,
        checks.FILING_CLASSES,
        [task.n_rows for i in population.instances for task in (i.a, i.b)],
    )
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
            "explanatory material for the roster release. These are not served tasks and "
            "are not part of the review manifest or of any bundle. The counterfactual "
            "renders listed here are review counterfactuals: they show what a receipt "
            "would have said under a convention that was not drawn, and no bank member "
            "holds them."
        ),
        "counterfactuals": counterfactuals,
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

    It is a separate call and it takes the name, because the export cannot know it and a
    default would be a machine signing for a person.
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

    Exposed so a reader can render an instance the pack did not cover without reaching for
    a second rendering route.
    """
    return render_fork(ledger.GENERATOR, instance, side, raw)


# --------------------------------------------------------------------------
# the screen procedure, audited
# --------------------------------------------------------------------------


def _number(value: object) -> float | None:
    """One recorded score as a float, or nothing when it is not a number at all."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value)
    except (OverflowError, ValueError):
        return None


def _allocation_refusals(
    record: Mapping[str, Any], registered: Mapping[str, Any], held: Sequence[int]
) -> list[str]:
    """Where a recorded allocation departs from the registered one, read against the bank."""
    problems: list[str] = []
    cases = list(record.get("cases") or [])
    exploratory = list(record.get("exploratory") or [])
    if len(cases) != FINAL_CASES:
        problems.append(
            f"the final screen holds {len(cases)} cases and the procedure registers "
            f"{FINAL_CASES}"
        )
    filed = [entry.get("ordinal") for entry in cases]
    if len(set(filed)) != len(filed):
        problems.append("the final screen repeats an instance")
    shared = sorted(set(filed) & set(exploratory), key=repr)
    if shared:
        problems.append(
            "the final screen reuses %d exploratory identities, the first being %r"
            % (len(shared), shared[0])
        )
    if len(exploratory) != EXPLORATORY_PREFIX:
        problems.append(
            f"{len(exploratory)} identities were reserved for exploration and the "
            f"procedure reserves {EXPLORATORY_PREFIX}"
        )
    elif registered["exploratory"] and exploratory != list(registered["exploratory"]):
        problems.append(
            "the reserved identities are not the first %d the bank admitted, which are %s"
            % (EXPLORATORY_PREFIX, registered["exploratory"])
        )
    outside = [value for value in filed if value not in set(held)]
    if outside:
        problems.append(
            "the final screen names %d identities this bank does not hold, the first "
            "being %r" % (len(outside), outside[0])
        )
    indexed = [entry.get("case") for entry in cases]
    unindexed = [
        value for value in indexed if isinstance(value, bool) or not isinstance(value, int)
    ]
    if unindexed:
        problems.append(
            "%d final cases carry no index, so the state rule is read against an order "
            "nobody can check" % len(unindexed)
        )
    elif indexed != list(range(len(cases))):
        problems.append(
            "the final cases are indexed %r and the procedure indexes them 0 to %d in the "
            "bank's own order" % (indexed[:4], len(cases) - 1)
        )
    elif registered["cases"] and filed != [
        entry["ordinal"] for entry in registered["cases"]
    ]:
        problems.append(
            "the final screen is not the %d identities the bank admitted after the "
            "reserved ones, in the order it admitted them" % FINAL_CASES
        )
    per_state: dict[Any, int] = {}
    for entry in cases:
        state = entry.get("state")
        per_state[state] = per_state.get(state, 0) + 1
        if entry.get("case") is not None and state != entry["case"] % INITIAL_STATES:
            problems.append(
                "case %r is allocated to state %r and the rule allocates it to %d"
                % (entry.get("case"), state, entry["case"] % INITIAL_STATES)
            )
    if sorted(per_state, key=repr) != list(range(INITIAL_STATES)):
        problems.append(
            "the final screen names states %s and the procedure names %s"
            % (sorted(per_state, key=repr), list(range(INITIAL_STATES)))
        )
    short = sorted(
        (state for state, seen in per_state.items() if seen != CASES_PER_STATE), key=repr
    )
    if short:
        problems.append(
            "states %s do not hold %d cases each (%s)"
            % (short, CASES_PER_STATE, {s: per_state[s] for s in short})
        )
    return problems


def _pin_refusals(record: Mapping[str, Any], instrument: Mapping[str, str]) -> list[str]:
    """What the recorded screen pins, and what it leaves a label rather than a state."""
    problems: list[str] = []
    if str(record.get("standing_instruction") or "") != STANDING_INSTRUCTION:
        problems.append(
            "the recorded standing instruction is not the registered one, so the arms did "
            "not start from the wording this screen is read against"
        )
    stated = dict(record.get("instrument") or {})
    for name, value in sorted(instrument.items()):
        if stated.get(name) != value:
            problems.append(
                "the screen names %s %r and this build is %r, so the evidence was "
                "gathered under another instrument" % (name, stated.get(name), value)
            )
    if not str(record.get("model") or "").strip():
        problems.append("the allocation names no model, so nothing in it says what was run")
    pinned = dict(record.get("states") or {})
    frozen: dict[int, str] = {}
    for state in range(INITIAL_STATES):
        stamp = pinned.get(str(state), pinned.get(state))
        if not isinstance(stamp, str) or not stamp.strip():
            problems.append(
                "initial state %d is not pinned to a digest, so a case's restoration is a "
                "label rather than a state" % state
            )
        else:
            frozen[state] = stamp
    drifted = [
        entry
        for entry in list(record.get("cases") or [])
        if entry.get("state") in frozen
        and entry.get("state_digest") != frozen[entry["state"]]
    ]
    if drifted:
        problems.append(
            "%d cases name an initial state digest the allocation does not pin, the first "
            "being case %r" % (len(drifted), drifted[0].get("case"))
        )
    return problems


def screen_refusals(
    record: Mapping[str, Any],
    instrument: Mapping[str, str],
    ordinals: Sequence[int],
    screen: Mapping[str, Any],
    release: bool = True,
) -> list[str]:
    """What a recorded screen does not establish for this genre. Empty when it meets the procedure.

    The executable screen proves neither the state allocation nor the authenticity of a
    named case from its label, and the bundle verifier enforces neither the per-state
    requirement nor the oracle release condition. This is the companion audit those
    sentences point at: it reads the recorded allocation and says where it departs from the
    registered procedure.

    IT IS READ AGAINST THE BANK AND THE SCREEN RECORD, NOT AGAINST ITSELF. An audit of the
    allocation's own fields accepts whatever the allocation says: a reservation that was
    never made, final identities no bank holds, cases in an order with no index to read the
    state rule against, a model nobody named, and grades that are not numbers. So the
    identities are held against `ordinals`, the bank's admitted order, and the outcomes
    against `screen`, the recorded `ScreenRecord` the pilot wrote, which validates its own
    rows.

    THE CASES AND THE RECORDED PAIRS ARE ONE SET, MATCHED ONE TO ONE. Counting the two
    sides establishes nothing about which case was which execution: thirty six cases can
    all name a single pair, report its outcomes truthfully, and carry the bank's own final
    ordinals in its own order, and the thirty five pairs nobody named would still be
    averaged into the release statistic. So every case names a pair, no pair is named twice,
    no pair goes unnamed, each case binds its pair to an identity the allocation puts in the
    final screen and to the A filing the record binds to that pair, and the release
    statistic is taken over the matched pairs and over nothing else.
    """
    from shogym.receipts import Outcomes, PairRecord, ScreenRecord

    problems: list[str] = []
    cases = list(record.get("cases") or [])
    held = [int(value) for value in ordinals]
    try:
        registered: Mapping[str, Any] = screen_allocation(held)
    except ValueError as exc:
        problems.append(str(exc))
        registered = {"exploratory": [], "cases": []}

    problems.extend(_allocation_refusals(record, registered, held))
    problems.extend(_pin_refusals(record, instrument))

    try:
        validated = ScreenRecord.from_payload(dict(screen))
    except (TypeError, ValueError) as exc:
        problems.append(
            "the screen record these cases are read against is not a readable screen: %s"
            % exc
        )
        return problems
    if validated.run.family != ledger.GENERATOR.name:
        problems.append(
            "the screen record was taken on %r and this audits %r"
            % (validated.run.family, ledger.GENERATOR.name)
        )
    model = str(record.get("model") or "").strip()
    if model and validated.run.model != model:
        problems.append(
            "the allocation names model %r and the screen record was taken with %r"
            % (model, validated.run.model)
        )
    pairs = {pair.instance: pair for pair in validated.run.pairs}
    if len(validated.run.pairs) != FINAL_CASES:
        problems.append(
            "the screen record holds %d pairs and the final screen is %d cases"
            % (len(validated.run.pairs), FINAL_CASES)
        )

    bound: dict[str, list[Any]] = {}
    for entry in cases:
        bound.setdefault(str(entry.get("instance") or ""), []).append(entry.get("case"))
    unpaired = [entry for entry in cases if str(entry.get("instance") or "") not in pairs]
    if unpaired:
        problems.append(
            "%d cases name no pair in the screen record, the first being case %r"
            % (len(unpaired), unpaired[0].get("case"))
        )
    doubled = sorted(
        (name for name, over in bound.items() if name in pairs and len(over) > 1), key=repr
    )
    if doubled:
        problems.append(
            "%d recorded pairs are named by more than one final case, the first being %r "
            "on cases %r, so the allocation is not %d executions"
            % (len(doubled), doubled[0], bound[doubled[0]][:4], FINAL_CASES)
        )
    unnamed = sorted((name for name in pairs if name not in bound), key=repr)
    if unnamed:
        problems.append(
            "the screen record holds %d pairs no final case names, the first being %r, so "
            "the record carries executions this allocation does not account for"
            % (len(unnamed), unnamed[0])
        )

    allocated = {entry["ordinal"] for entry in registered["cases"]}
    matched: list[PairRecord] = []
    strayed: list[tuple[Any, str]] = []
    misfiled: list[tuple[Any, str]] = []
    moved: list[tuple[Any, str]] = []
    for entry in cases:
        pair = pairs.get(str(entry.get("instance") or ""))
        if pair is None:
            continue
        holds = len(bound[pair.instance]) == 1
        if allocated and entry.get("ordinal") not in allocated:
            strayed.append((entry.get("case"), pair.instance))
            holds = False
        filing = entry.get("filing")
        if not isinstance(filing, str) or filing.strip() != pair.filing:
            misfiled.append((entry.get("case"), pair.instance))
            holds = False
        for branch in ("placebo", "graded", "oracle"):
            if _number(entry.get(branch)) != getattr(pair, branch):
                moved.append((entry.get("case"), branch))
                holds = False
                break
        if holds:
            matched.append(pair)
    if strayed:
        problems.append(
            "%d cases bind a recorded pair to an identity the allocation does not put in "
            "the final screen, the first being case %r on pair %r"
            % (len(strayed), strayed[0][0], strayed[0][1])
        )
    if misfiled:
        problems.append(
            "%d cases do not carry the A filing the screen record binds to the pair they "
            "name, the first being case %r on pair %r"
            % (len(misfiled), misfiled[0][0], misfiled[0][1])
        )
    if moved:
        problems.append(
            "%d cases report an outcome the screen record does not, the first being the %s "
            "of case %r" % (len(moved), moved[0][1], moved[0][0])
        )
    if problems:
        # The mean is a statement about executed outcomes, and there are none to average
        # until the cases, the pins and the record agree about what was executed.
        return problems
    try:
        grades = Outcomes(
            placebo=tuple(pair.placebo for pair in matched),
            graded=tuple(pair.graded for pair in matched),
            oracle=tuple(pair.oracle for pair in matched),
        ).oracle
    except ValueError as exc:
        problems.append("the screen record's branch scores are not scores: %s" % exc)
        return problems
    mean = sum(grades) / len(grades)
    if release and mean < MIN_MEAN_ORACLE:
        problems.append(
            "the mean oracle grade over the %d matched cases is %.6f and the predeclared "
            "release condition is %.2f" % (len(grades), mean, MIN_MEAN_ORACLE)
        )
    return problems


__all__ = [
    "CASES_PER_STATE",
    "CHECKLIST",
    "EXPLORATORY_PREFIX",
    "FINAL_CASES",
    "INITIAL_STATES",
    "MIN_MEAN_ORACLE",
    "OFF_BAND",
    "PACK",
    "RENDERS",
    "STANDING_INSTRUCTION",
    "WORKSHEETS",
    "WORKSHEET_INDEX",
    "Render",
    "Selected",
    "attested",
    "export",
    "read_fork",
    "read_pack",
    "screen_allocation",
    "screen_refusals",
]
