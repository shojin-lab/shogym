"""Export what a human has to have read before this genre can be released.

A person reading rendered instances is the boundary against a family that is wrong in a
way no mechanical check can see. That only holds if the person saw the family, so the
pack is enumerated from the generator's own declarations rather than chosen by whoever
exported it: the surface template, every option of the axis, every registered filing
class, the row count the bank holds, and counterfactual renders under the rules that were
not drawn.

WHAT THIS ADDS BEYOND THE SHARED COVERAGE. The graded, placebo and oracle cells of every
filing class on BOTH siblings, so the reader follows one correct filing and one incorrect
one all the way through; both raw task texts; all three oracle cells on one pair, so every
sentence the oracle arm can state is in front of the reader rather than the one this draw
produced; and a graded cell on each sibling under each of the two rules that were not
drawn, labelled in the manifest as a review counterfactual so that nothing here reads as a
sampled bank member.

AND A WORKSHEET, WHICH IS NOT A TASK. Beside the renders it writes a private worksheet:
per row the occupied cells, the island count under each of the three rules, which rule the
row singles out, the contact graph signature, what was filed, the verdict and the
correction, next to the oracle's own wording. Strata, signatures and counterfactual counts
belong there and nowhere else, and no cell prints them. The worksheet is deliberately NOT
in the review manifest and not a bundle field: it is explanatory material for the roster
release, carried with its own hash, and labelling it as a render would put a document
nobody serves inside the evidence that says what was served.

THE ATTESTATION IS LEFT UNSET. The manifest carries every field a pack carries and names
no reviewer, so `review.verify` refuses it until a person puts their name to it. An
exporter that filled that in would be a machine attesting that a human read something.

THE SCREEN PROCEDURE IS AUDITED HERE AND NOT RUN HERE. `screen_refusals` holds a recorded
allocation against the registered procedure: twelve reserved exploratory identities, the
next thirty six as the final screen, case j to state j mod 4, nine per state, the common
standing instruction, one frozen instrument, and the predeclared release condition of a
mean oracle grade of at least 0.90. No model has been run for this genre and nothing here
runs one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from shogym.envs.receipts import checks
from shogym.envs.receipts.bank import (
    Bank,
    Population,
    bank_identity,
    load_bank,
    population,
    render_fork,
)
from shogym.envs.receipts.generators import components, components_audit
from shogym.envs.receipts.protocol import Instance, Task
from shogym.envs.receipts.receipt_ast import GRADED, ORACLE, PLACEBO, serialize
from shogym.envs.receipts.registry import bank_path
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
    "the public mechanics state all three contact rules and select none of them",
    "the same option means the same contact predicate on every board of both siblings",
    "a border row and a diagonal whose intervening cells are occupied behave as the "
    "mechanics say, and neither wrapping nor diagonal blocking appears on one surface only",
    "the receipt names boards and never a rule, a stratum or a contact",
    "a failed row's correction is that row's own island count and nothing else",
    "the oracle states the drawn rule, adds no coaching and no worked board, and scopes "
    "itself by the schedules the reader files rather than by a name printed on one",
    "the placebo carries no grade, no answer and nothing that reads as either",
    "the counterfactual renders are labelled as review counterfactuals and are not bank "
    "members",
    "the reconstructed graphs of the worked boards and of a row in every stratum agree "
    "with the printed answers, including the board whose side contacts close a cycle",
    "a correct filing and an incorrect one behave as stated through all seven filing "
    "classes, including the first of two duplicates and an omitted row",
)

#: The phenomena the worksheet has to exhibit, each a question about one row.
WORKSHEET_CASES = (
    "a board whose side contacts close a cycle",
    "a board two cells of which touch only through a third",
    "a board of one occupied cell",
    "a board of six occupied cells",
    "a board on the border of the grid",
    "a diagonal pair whose two intervening side neighbours are both occupied",
    "a separated control, which every rule answers alike",
)

# ----- the registered model room screen, audited rather than run -----------------

#: The identities reserved for exploration, and the final screen after them.
EXPLORATORY_PREFIX = 12
FINAL_CASES = 36
INITIAL_STATES = 4
CASES_PER_STATE = 9
#: The predeclared release condition beside the machine screen bars. It is an ADDITIONAL
#: roster condition: the executable screen does not enforce it and the release audit
#: recomputes it from the same sealed outcomes.
MIN_MEAN_ORACLE = 0.90

#: The common standing instruction every arm starts with, in its registered wording.
#: IT IS NOT INSTALLED BY THIS REPOSITORY. The harness that carries standing instructions is
#: a different one, and at the time of this build it still carries the shorter sentence this
#: one replaces, which told a reader that a payload was not a task and not that it was part
#: of the reader's own instructions. This constant is what the screen evidence is audited
#: against, so evidence gathered under the old wording is refused here rather than accepted
#: and explained afterwards.
STANDING_INSTRUCTION = (
    "A payload is a message for you about your work, from the same server that gives you "
    "your tasks. It is part of your instructions: read it, keep what it tells you for the "
    "tasks that follow, and pull again."
)


@dataclass(frozen=True)
class Render:
    """One exported artifact and what it is evidence of."""

    category: str
    key: str
    kind: str
    path: str


def _write(root: Path, name: str, payload: bytes | str) -> str:
    """One exported file, returned as the path the manifest names it by."""
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        target.write_bytes(payload)
    else:
        target.write_text(payload, encoding="utf-8")
    return name


def _retasked(task: Task, convention: Mapping[str, str]) -> Task:
    """The same task scored under another rule, for a counterfactual render."""
    return Task(
        label=task.label,
        task_id=task.task_id,
        surface=task.surface,
        table=task.table,
        text=task.text,
        key=tuple(components.key_for(task.table, convention)),
    )


def _cells_for(
    instance: Instance, task: Task, convention: Mapping[str, str], raw: object
) -> dict[str, bytes]:
    """The three cells one filing would produce on one task, through the shared judge."""
    from shogym.envs.receipts.render import judge_cells

    generator = components.GENERATOR
    canonical = generator.parse_and_canonicalize(task, raw)
    judged = judge_cells(generator, task, canonical, convention, instance.envelope)
    if judged.problems:
        raise ValueError(judged.problems[0])
    return dict(judged.payloads)


def _mixed_filing(task: Task, every: int = 3) -> str:
    """A filing that passes most rows and fails some, deterministically.

    Deterministic in the task's own printed order rather than drawn from a stream, because
    two readers of one pack have to be looking at one artifact and because the worksheet's
    verdict column is only worth reading if some rows failed.
    """
    lines = []
    generator = components.GENERATOR
    for position, (identifier, correct) in enumerate(
        zip(generator.row_identifiers(task.table), task.key)
    ):
        wrong = "6" if correct != "6" else "1"
        lines.append("%s,%s" % (identifier, correct if position % every else wrong))
    return "\n".join(lines)


def _worksheet_rows(
    task: Task, convention: Mapping[str, str], raw: str
) -> list[dict[str, Any]]:
    """One private worksheet row per printed row, with the rules spelled out side by side."""
    generator = components.GENERATOR
    retasked = _retasked(task, convention)
    canonical = generator.parse_and_canonicalize(retasked, raw)
    _, outcomes = generator.score(retasked, canonical)
    out: list[dict[str, Any]] = []
    for row, outcome in zip(task.table.rows, outcomes):
        counts = components_audit.flood_counts(row.cells)
        out.append(
            {
                "identifier": row.row_id,
                "cells": components.print_cells(row.cells),
                "side_contacts": counts[0],
                "corner_contacts": counts[1],
                "combined_contacts": counts[2],
                "singles_out": components_audit.classify(row.cells),
                "signature": list(components_audit.graph_signature(tuple(row.cells))),
                "filed": outcome.filed if outcome.was_filed else "(unfiled)",
                "verdict": "PASS" if outcome.matched else "FAIL",
                "correction": "" if outcome.matched else outcome.correct,
            }
        )
    return out


def _case_rows(held: Population) -> list[dict[str, Any]]:
    """The earliest row in the bank exhibiting each phenomenon the worksheet needs."""
    found: dict[str, dict[str, Any]] = {}
    for instance in held.instances:
        if len(found) == len(WORKSHEET_CASES):
            break
        for side in ("a", "b"):
            task = instance.side(side)
            for row in task.table.rows:
                cells = tuple(row.cells)
                counts = components_audit.flood_counts(cells)
                occupied = set(cells)
                seen = {
                    WORKSHEET_CASES[0]: components.has_side_cycle(cells),
                    WORKSHEET_CASES[1]: _joined_only_through_a_third(cells),
                    WORKSHEET_CASES[2]: len(cells) == 1,
                    WORKSHEET_CASES[3]: len(cells) == components.MAX_CELLS,
                    WORKSHEET_CASES[4]: any(
                        part in (1, components.SIDE) for cell in cells for part in cell
                    ),
                    WORKSHEET_CASES[5]: any(
                        components_audit.pair_colour(one, other) == 2
                        and (one[0], other[1]) in occupied
                        and (other[0], one[1]) in occupied
                        for one in cells
                        for other in cells
                        if one != other
                    ),
                    WORKSHEET_CASES[6]: components_audit.classify(cells) == "control",
                }
                for case, holds in seen.items():
                    if holds and case not in found:
                        found[case] = {
                            "case": case,
                            "instance": instance.ordinal,
                            "side": side.upper(),
                            "identifier": row.row_id,
                            "cells": components.print_cells(cells),
                            "side_contacts": counts[0],
                            "corner_contacts": counts[1],
                            "combined_contacts": counts[2],
                            "signature": list(
                                components_audit.graph_signature(cells)
                            ),
                        }
    return [found[case] for case in WORKSHEET_CASES if case in found]


def _joined_only_through_a_third(cells: Sequence[tuple[int, int]]) -> bool:
    """Whether some two cells share an island under some rule without touching directly."""
    for option in components_audit.OPTIONS:
        neighbours = components_audit.neighbourhood(cells, option)
        for cell in cells:
            start = (cell[0], cell[1])
            direct = set(neighbours[start])
            reached = set(direct)
            frontier = list(direct)
            while frontier:
                here = frontier.pop()
                for other in neighbours[here]:
                    if other not in reached and other != start:
                        reached.add(other)
                        frontier.append(other)
            if reached - direct - {start}:
                return True
    return False


def export(bank: Bank, held: Population, directory: str | Path) -> Path:
    """Write the review pack and its worksheets, and return the manifest's path.

    Deterministic in the bank and its population: the same bank exports the same bytes, so
    two readers are looking at one pack and a second export is a comparison rather than a
    new document. A nonempty output directory is refused, because a pack mixed with an
    earlier one is a pack nobody can say what was read from.
    """
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError(
            f"{root} is not empty. A review pack is what a person read, and a directory "
            "holding an earlier export as well has no single answer to what that was"
        )
    root.mkdir(parents=True, exist_ok=True)
    generator = components.GENERATOR
    if bank.generator != generator.name:
        raise ValueError(
            f"this exporter writes packs for {generator.name!r} and the bank names "
            f"{bank.generator!r}"
        )
    if not held.instances:
        raise ValueError("a review pack is a reading of instances and this bank holds none")

    renders: list[Render] = []
    first = held.instances[0]
    drawn = dict(first.convention)

    # ----- both raw task texts, and the one surface template ------------------
    for side in ("a", "b"):
        task = first.side(side)
        path = _write(
            root, f"{RENDERS}/task-{side}-{first.ordinal:04d}.txt", task.text
        )
        renders.append(Render("surface", task.surface, "task", path))

    # ----- all three cells for every registered filing class, on both siblings --
    for shape in checks.FILING_CLASSES:
        for side in ("a", "b"):
            task = first.side(side)
            raw = checks.filing_of(generator, first, side, shape)
            cells = _cells_for(first, task, first.convention, raw)
            for kind in (GRADED, PLACEBO, ORACLE):
                path = _write(
                    root, f"{RENDERS}/filing-{shape}-{side}-{kind}.txt", cells[kind]
                )
                renders.append(Render("filing", shape, "cell", path))

    # ----- the row count the bank holds ---------------------------------------
    path = _write(
        root,
        f"{RENDERS}/rows-{components.ROWS:04d}.txt",
        _cells_for(
            first, first.a, first.convention,
            checks.filing_of(generator, first, "a", "canonical"),
        )[GRADED],
    )
    renders.append(Render("rows", str(components.ROWS), "cell", path))

    # ----- one oracle cell per option, so every sentence the arm can state is read
    for option in components_audit.OPTIONS:
        convention = {"contact_kernel": option}
        payload = serialize(
            generator.render_oracle(first.a.task_id, convention, first.a.n_rows),
            first.envelope,
        )
        path = _write(root, f"{RENDERS}/oracle-{option}.txt", payload)
        renders.append(Render("option", f"contact_kernel={option}", "cell", path))

    # ----- a counterfactual graded cell on each sibling under each other rule ---
    worksheets: list[dict[str, Any]] = []
    counterfactuals: list[dict[str, Any]] = []
    for side in ("a", "b"):
        task = first.side(side)
        raw = _mixed_filing(task)
        for option in components_audit.OPTIONS:
            if option == drawn["contact_kernel"]:
                continue
            alternative = {"contact_kernel": option}
            cells = _cells_for(
                first, _retasked(task, alternative), alternative, raw
            )
            path = _write(
                root, f"{RENDERS}/counterfactual-{option}-{side}.txt", cells[GRADED]
            )
            renders.append(
                Render("counterfactual", "alternative convention", "cell", path)
            )
            renders.append(
                Render(
                    "counterfactual",
                    "review counterfactual under %s, side %s" % (option, side.upper()),
                    "cell",
                    path,
                )
            )
            counterfactuals.append(
                {"path": path, "side": side.upper(), "rule": option, "drawn": drawn}
            )
            sheet = {
                "instance": first.ordinal,
                "side": side.upper(),
                "drawn": drawn,
                "alternative": alternative,
                "oracle_drawn": list(
                    generator.render_oracle("0" * 16, drawn, components.ROWS).body
                ),
                "oracle_alternative": list(
                    generator.render_oracle("0" * 16, alternative, components.ROWS).body
                ),
                "rows": _worksheet_rows(task, alternative, raw),
            }
            worksheets.append(sheet)
            _write(
                root,
                f"{WORKSHEETS}/{side}-{option}.json",
                json.dumps(sheet, indent=1, sort_keys=True),
            )

    cases = _case_rows(held)
    _write(root, f"{WORKSHEETS}/cases.json", json.dumps(cases, indent=1, sort_keys=True))

    coverage = required_coverage(generator, checks.FILING_CLASSES, [components.ROWS])
    missing = coverage.missing([(r.category, r.key) for r in renders])
    if missing:
        raise ValueError("this export does not cover " + ", ".join(missing))

    manifest = {
        # UNSET, AND THE FIELD IS STILL HERE. A pack names exactly six things and this
        # names six; the one a machine cannot supply is null, so `review.verify` refuses
        # the pack by name until a person puts theirs to it.
        "reviewer": None,
        "checklist": list(CHECKLIST),
        "seeds": list(held.ordinals),
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
            "would have said under a rule that was not drawn, and no bank member holds "
            "them."
        ),
        "cases": [entry["case"] for entry in cases],
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
    return render_fork(components.GENERATOR, instance, side, raw)


# --------------------------------------------------------------------------
# the screen procedure, audited
# --------------------------------------------------------------------------


def screen_allocation(ordinals: Sequence[int]) -> dict[str, Any]:
    """The registered allocation of a bank's admitted identities to the screen.

    The first twelve are the exploratory prefix and are never final cases. The next
    thirty six are the final screen, and case j goes to initial state j mod 4, which gives
    nine distinct cases per state. Nothing here balances observed outcomes or selects
    identities after the fact: the allocation is a function of the bank's own order.
    """
    held = [int(ordinal) for ordinal in ordinals]
    needed = EXPLORATORY_PREFIX + FINAL_CASES
    if len(held) < needed:
        raise ValueError(
            f"the registered screen needs {needed} admitted identities and this bank "
            f"holds {len(held)}"
        )
    final = held[EXPLORATORY_PREFIX:needed]
    return {
        "exploratory": held[:EXPLORATORY_PREFIX],
        "cases": [
            {"case": position, "ordinal": ordinal, "state": position % INITIAL_STATES}
            for position, ordinal in enumerate(final)
        ],
    }


def screen_refusals(
    record: Mapping[str, Any], instrument: Mapping[str, str], release: bool = True
) -> list[str]:
    """What a recorded screen does not establish. Empty when it meets the procedure.

    The executable screen proves neither the state allocation nor the authenticity of a
    named case from its label, and the bundle verifier enforces neither the per-state
    requirement nor the oracle release condition. This is the companion audit those
    sentences point at: it reads the recorded allocation and says where it departs from
    the registered procedure.
    """
    problems: list[str] = []
    cases = list(record.get("cases") or [])
    exploratory = [int(value) for value in (record.get("exploratory") or [])]
    if len(cases) != FINAL_CASES:
        problems.append(
            f"the final screen holds {len(cases)} cases and the procedure registers "
            f"{FINAL_CASES}"
        )
    ordinals = [entry.get("ordinal") for entry in cases]
    if len(set(ordinals)) != len(ordinals):
        problems.append("the final screen repeats an instance")
    shared = sorted(set(ordinals) & set(exploratory))
    if shared:
        problems.append(
            "the final screen reuses %d exploratory identities, the first being %r"
            % (len(shared), shared[0])
        )
    if len(exploratory) > EXPLORATORY_PREFIX:
        problems.append(
            f"{len(exploratory)} identities were reserved for exploration and the "
            f"procedure reserves {EXPLORATORY_PREFIX}"
        )
    per_state: dict[int, int] = {}
    for entry in cases:
        state = entry.get("state")
        per_state[state] = per_state.get(state, 0) + 1
        if entry.get("case") is not None and state != entry["case"] % INITIAL_STATES:
            problems.append(
                "case %r is allocated to state %r and the rule allocates it to %d"
                % (entry.get("case"), state, entry["case"] % INITIAL_STATES)
            )
    if sorted(per_state) != list(range(INITIAL_STATES)):
        problems.append(
            "the final screen names states %s and the procedure names %s"
            % (sorted(per_state), list(range(INITIAL_STATES)))
        )
    short = sorted(
        state for state, seen in per_state.items() if seen != CASES_PER_STATE
    )
    if short:
        problems.append(
            "states %s do not hold %d cases each (%s)"
            % (short, CASES_PER_STATE, {s: per_state[s] for s in short})
        )
    if str(record.get("standing_instruction") or "") != STANDING_INSTRUCTION:
        problems.append(
            "the recorded standing instruction is not the registered one, so the arms "
            "did not start from the wording this screen is read against"
        )
    stated = dict(record.get("instrument") or {})
    for name, value in sorted(instrument.items()):
        if stated.get(name) != value:
            problems.append(
                "the screen names %s %r and this build is %r, so the evidence was "
                "gathered under another instrument" % (name, stated.get(name), value)
            )
    grades = [float(entry.get("oracle", 0.0)) for entry in cases]
    mean = sum(grades) / len(grades) if grades else 0.0
    if release and mean < MIN_MEAN_ORACLE:
        problems.append(
            "the mean oracle grade is %.6f and the predeclared release condition is "
            "%.2f" % (mean, MIN_MEAN_ORACLE)
        )
    return problems


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Export the pack from the registered bank this build would deal from."""
    parser = argparse.ArgumentParser(
        description="export the components review pack a person has to read"
    )
    parser.add_argument("--out", required=True, metavar="DIRECTORY")
    args = parser.parse_args(list(argv) if argv is not None else None)
    path = bank_path(components.GENERATOR.name)
    if not path.is_file():
        print(f"no frozen bank for 'components' at {path}; materialize one first")
        return 1
    try:
        bank = load_bank(path)
        held = population(bank, components.GENERATOR)
        pack = export(bank, held, args.out)
    except (OSError, ValueError) as exc:
        print(f"this does not make a review pack: {exc}")
        return 1
    print(f"review pack {pack}")
    print(f"bank identity {bank_identity(bank)}")
    print(
        "%d instances, ordinals %s"
        % (len(held.instances), list(held.ordinals))
    )
    print(
        "the pack names no reviewer. A person reads it and then calls "
        "`components_review.attested(directory, name)`"
    )
    return 0


__all__ = [
    "CASES_PER_STATE",
    "CHECKLIST",
    "EXPLORATORY_PREFIX",
    "FINAL_CASES",
    "INITIAL_STATES",
    "MIN_MEAN_ORACLE",
    "PACK",
    "RENDERS",
    "STANDING_INSTRUCTION",
    "WORKSHEETS",
    "WORKSHEET_CASES",
    "WORKSHEET_INDEX",
    "Render",
    "attested",
    "export",
    "main",
    "read_fork",
    "read_pack",
    "screen_allocation",
    "screen_refusals",
]


if __name__ == "__main__":
    sys.exit(main())
