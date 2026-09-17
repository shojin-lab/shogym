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
