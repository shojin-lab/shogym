"""The ledger review pack: what it covers, what it refuses, and what a bundle makes of it.

Each test states the failure it is here to catch, because a test whose failure condition is
"something changed" is a test nobody can act on.

THE TWO KEYS HERE ARE FIXED, and that is the subject rather than a convenience. A pack shows
eleven things a sampled receipt does and four of them are facts about the rows a stream drew,
so a bank under a fresh key each run is a bank that sometimes holds them and sometimes does
not. The complete bank's key draws all eight surfaces and all four searched cases in the
smallest bank that can hold eight surfaces; the incomplete bank's key draws every surface and
no mask that pins the convention, which is the bank the refusal is about.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shogym.envs.receipts import admission
from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import bundle as bundle_mod
from shogym.envs.receipts import checks, ledger_review, review
from shogym.envs.receipts.generators import ledger
from shogym.envs.receipts.protocol import Task
from shogym.envs.receipts.receipt_ast import GRADED, ORACLE, PLACEBO, serialize
from shogym.envs.receipts.render import judge_cells
from shogym.envs.receipts.streams import digest

#: A bank that draws every surface this genre has and exhibits all four searched sampled
#: cases. Four instances is the smallest bank that can draw eight surfaces, because each
#: instance draws one of each pool.
COMPLETE_MASTER = hashlib.sha256(b"ledger-review-pack:3").digest()
COMPLETE_SIZE = 4

#: A bank that draws every surface and no mask that pins the convention to one member of
#: the support. It is the bank the refusal is about: the shortfall is the bank's, and a pack
#: exported from it would have gone to a reader with one of the eleven silently missing.
INCOMPLETE_MASTER = hashlib.sha256(b"ledger-review-pack:7").digest()
INCOMPLETE_SIZE = 4


def _screen_artifact(pairs: int = 40) -> dict:
    """A recorded room screen for this family, structurally valid and not measured here.

    The numbers stand in for a pilot nobody ran in a test. What is being exercised is the
    path that recomputes everything mechanical about a bundle, not the screen.
    """
    return {
        "family": ledger.GENERATOR.name,
        "model": "a scripted policy",
        "task_seeds": [str(i) for i in range(pairs)],
        "pairs": [
            {
                "instance": f"task-{i:02d}",
                "filing": f"filing-{i:02d}",
                "placebo": 0.4,
                "graded": 0.6,
                "oracle": 0.95,
                "ideal": 0.82,
            }
            for i in range(pairs)
        ],
        "min_room": 0.05,
        "min_ratio": 0.25,
        "min_pairs": 36,
        "min_oracle": 0.90,
        "min_learning_gap": 0.10,
        "floor": 0.0,
        "floor_rule": "drop",
        "candidates_screened": 1,
        "selection_note": "",
    }


@pytest.fixture(scope="module")
def complete(tmp_path_factory: pytest.TempPathFactory):
    """One small complete bank of this genre and the pack exported from it.

    Module scoped because filling a bank walks admission over every ordinal it considers,
    which is the expensive thing this package does, and every test here reads one bank.
    """
    room = tmp_path_factory.mktemp("ledger-pack")
    bank, held = bank_mod.materialized(ledger.GENERATOR, COMPLETE_MASTER, COMPLETE_SIZE)
    root = room / "pack"
    pack = ledger_review.export(bank, held, root)
    return bank, held, root, pack


def _rows(cell: str) -> dict[int, str]:
    """The receipt's printed rows by their printed ordinal, for reading one row's columns."""
    out: dict[int, str] = {}
    for line in cell.splitlines():
        head = line.split()
        if head and head[0].isdigit():
            out[int(head[0])] = line
    return out


def test_the_review_pack_covers_the_family_and_names_no_reviewer(complete) -> None:
    """What the exported pack has to contain, and the one thing it must not.

    It fails if the pack misses a surface template, an option of any axis, a registered
    filing class, a row count the bank holds, a counterfactual render or any of the eleven
    a sampled family owes, if the 72 oracle cells are not all there, if a counterfactual is
    missing for any axis on either sibling, or if the exporter names a reviewer. It also
    fails if the worksheets are labelled as renders: they are explanatory material for the
    roster release, they are not served tasks, and a bundle that carried them as evidence of
    what was served would be saying something nobody checked.
    """
    bank, held, root, pack = complete
    manifest = json.loads(pack.read_text(encoding="utf-8"))
    assert set(manifest) == set(review.REQUIRED_FIELDS)
    assert manifest["reviewer"] is None
    assert manifest["family"] == "ledger"
    assert manifest["bank"] == bank_mod.bank_identity(bank)
    assert manifest["seeds"] == list(held.ordinals)
    assert manifest["checklist"] == list(ledger_review.CHECKLIST)

    coverage = review.required_coverage(
        ledger.GENERATOR,
        checks.FILING_CLASSES,
        [task.n_rows for i in held.instances for task in (i.a, i.b)],
    )
    seen = [(e["category"], e["key"]) for e in manifest["renders"]]
    assert coverage.missing(seen) == []
    assert {key for category, key in seen if category == "sampled"} == set(
        review.SAMPLED_CASES
    )

    oracles = {
        e["path"] for e in manifest["renders"] if e["path"].startswith("renders/oracle-")
    }
    assert len(oracles) == len(ledger.ALL_CONVENTIONS) == 72
    for side in ("A", "B"):
        for axis in ledger.AXES:
            assert any(
                key.startswith(f"review counterfactual under {axis.name}=")
                and key.endswith(f"side {side}")
                for _, key in seen
            ), (side, axis.name)

    for entry in manifest["renders"]:
        assert not entry["path"].startswith(ledger_review.WORKSHEETS)
        assert (root / entry["path"]).is_file()

    index = json.loads((root / ledger_review.WORKSHEET_INDEX).read_text("utf-8"))
    assert index["family"] == "ledger"
    assert len(index["counterfactuals"]) == 2 * len(ledger.AXES)
    assert index["files"]
    for name, hashed in index["files"].items():
        assert digest((root / name).read_bytes()) == hashed
    written = sorted((root / ledger_review.WORKSHEETS).glob("a-anchor-*.json"))
    assert len(written) == 1
    sheet = json.loads(written[0].read_text("utf-8"))
    assert sheet["axis_changed"] == "anchor"
    assert len(sheet["rows"]) == ledger.SHAPE.rows
    for row in sheet["rows"]:
        assert set(row) >= {
            "record",
            "count_drawn",
            "count_alternative",
            "band_drawn",
            "band_alternative",
            "verdict",
            "correction",
        }


def test_the_pack_shows_what_a_receipt_of_four_records_does(complete) -> None:
    """The eleven a sampled family's pack has to carry, in this genre's own records.

    FAILS IF a reported record's cell does not show the verdict the case names, if the
    suppressed failure is offered as a cell rather than as the document that says which
    records failed, if the mask commitment document does not print the three cells beside
    the rows the stream drew, or if the pair of conventions the receipt cannot tell apart
    render two different cells. The last is the point of the policy rather than a defect in
    it, and a pack that could not show it would be hiding what the family does.
    """
    _bank, held, root, pack = complete
    manifest = json.loads(pack.read_text(encoding="utf-8"))
    sampled = {
        entry["key"]: entry
        for entry in manifest["renders"]
        if entry["category"] == "sampled"
    }
    assert sorted(sampled) == sorted(review.SAMPLED_CASES)
    assert len(review.SAMPLED_CASES) == 11

    for case in review.SAMPLED_CASES[:4]:
        assert sampled[case]["kind"] == "cell"
        assert (root / sampled[case]["path"]).stat().st_size == ledger.ENVELOPE_SIZE
    for case in review.SAMPLED_CASES[4:]:
        assert sampled[case]["kind"] == "document"

    first = held.instances[0]
    reported = [position + 1 for position in first.a.mask]
    assert len(reported) == ledger.GENERATOR.RECEIPT_POLICY.reported == 4
    read = {
        case: _rows((root / sampled[case]["path"]).read_text("ascii"))
        for case in review.SAMPLED_CASES[:4]
    }
    for ordinal in reported:
        assert "PASS" in read[review.SAMPLED_CASES[0]][ordinal]
        assert "FAIL" in read[review.SAMPLED_CASES[1]][ordinal]
        assert ledger.UNFILED_TOKEN in read[review.SAMPLED_CASES[2]][ordinal]
        assert ledger.BLANK_TOKEN in read[review.SAMPLED_CASES[3]][ordinal]

    suppressed = (root / sampled[review.SAMPLED_CASES[4]]["path"]).read_text("ascii")
    assert "says nothing about any of them" in suppressed
    assert suppressed.count("FAIL") == 0

    commitment = (root / sampled[review.SAMPLED_CASES[9]]["path"]).read_text("ascii")
    assert ledger.GENERATOR.RECEIPT_POLICY.name in commitment
    assert "committed before any filing existed: %s" % ", ".join(
        str(r) for r in reported
    ) in commitment
    assert commitment.count("SUBMISSION RECEIPT") == 3

    aliased = (root / sampled[review.SAMPLED_CASES[10]]["path"]).read_text("ascii")
    assert "render the cell below byte for byte" in aliased
    assert "records of the schedule this reader files next" in aliased


def test_a_bank_that_cannot_exhibit_a_case_is_refused_by_name(tmp_path: Path) -> None:
    """A pack is all eleven cases, and a small bank does not always draw all eleven.

    It fails if an export drops a sampled case the bank does not exhibit, which would send a
    pack to a reader with no example of a receipt that pins the convention while the
    manifest said nothing was missing; if the refusal does not name the case; or if it
    leaves a half written pack behind for somebody to read as one. The key here draws every
    surface and no mask that pins the convention, so what is being exercised is the sampled
    search rather than the surface one, which is refused by name just above it.
    """
    bank, held = bank_mod.materialized(
        ledger.GENERATOR, INCOMPLETE_MASTER, INCOMPLETE_SIZE
    )
    room = tmp_path / "pack"
    with pytest.raises(ValueError) as refused:
        ledger_review.export(bank, held, room)
    said = str(refused.value)
    assert review.SAMPLED_CASES[7] in said
    assert "Materialize more instances" in said
    assert not room.exists()

    # And a bank with no instances at all is a reading of nothing, before any search runs.
    empty = bank_mod.Population(instances=(), considered=0)
    with pytest.raises(ValueError, match="holds none"):
        ledger_review.export(bank, empty, tmp_path / "empty")
    assert not (tmp_path / "empty").exists()


def test_the_export_is_the_same_bytes_twice(complete, tmp_path: Path) -> None:
    """One bank exports one pack, so two readers are looking at one document.

    It fails if any exported byte moves between two exports of one bank: the manifest, a
    rendered cell, an assembled document or a worksheet. A pack whose contents moved between
    exports would make a reader's attestation a statement about a document nobody else can
    produce, and the bundle that carries it addressable by nothing stable.
    """
    bank, held, root, _pack = complete
    twice = tmp_path / "twice"
    ledger_review.export(bank, held, twice)
    first = {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    second = {
        str(path.relative_to(twice)): path.read_bytes()
        for path in sorted(twice.rglob("*"))
        if path.is_file()
    }
    assert sorted(first) == sorted(second)
    assert first == second

    # And a directory that already holds an export is refused rather than mixed into.
    with pytest.raises(ValueError, match="not empty"):
        ledger_review.export(bank, held, twice)


def test_the_exporter_renders_only_conventions_the_family_declares(complete) -> None:
    """Every convention the pack shows comes from the declared support, not from the draw.

    It fails if the oracle cells are not the whole declared support in its own product
    order, if any counterfactual graded cell is not the cell this package renders under a
    member of that support, or if a convention the family does not declare can be rendered
    at all. A pack whose counterfactuals came from the live draw would show a reader fewer
    rules the smaller the bank, and a pack that assembled its own conventions would put a
    cell in front of a reader that no bank member could ever serve.
    """
    _bank, held, root, pack = complete
    manifest = json.loads(pack.read_text(encoding="utf-8"))
    first = held.instances[0]

    declared = [dict(convention) for convention in ledger.ALL_CONVENTIONS]
    for position, convention in enumerate(declared):
        payload = (root / f"{ledger_review.RENDERS}/oracle-{position:02d}.txt").read_bytes()
        assert payload == serialize(
            ledger.GENERATOR.render_oracle(
                first.a.task_id, convention, first.a.n_rows
            ),
            first.envelope,
        )
        stated = payload.decode("ascii")
        for axis, option in convention.items():
            assert ledger.PHRASES[axis][option] in " ".join(stated.split())

    index = json.loads((root / ledger_review.WORKSHEET_INDEX).read_text("utf-8"))
    for entry in index["counterfactuals"]:
        alternative = entry["alternative"]
        assert alternative in declared
        task = first.side(entry["side"].lower())
        raw = ledger_review._mixed_filing(task)
        retasked = Task(
            label=task.label,
            task_id=task.task_id,
            surface=task.surface,
            table=task.table,
            text=task.text,
            key=tuple(ledger.key_for(task.table, alternative)),
            mask=task.mask,
        )
        judged = judge_cells(
            ledger.GENERATOR,
            retasked,
            ledger.GENERATOR.parse_and_canonicalize(retasked, raw),
            alternative,
            first.envelope,
        )
        assert judged.problems == ()
        assert (root / entry["path"]).read_bytes() == judged.payloads[GRADED]

    # The one place a convention could be invented refuses instead.
    with pytest.raises(ValueError, match="declares"):
        ledger_review._position({"anchor": "event", "basis": "calendar"})
    with pytest.raises(ValueError, match="declares"):
        ledger_review._position(
            {
                "anchor": "event",
                "basis": "calendar",
                "boundary": "lower",
                "missing": "unstated",
            }
        )

    # And no render is a cell of a shape this family does not serve.
    for entry in manifest["renders"]:
        if entry["kind"] == "cell":
            assert (root / entry["path"]).stat().st_size == ledger.ENVELOPE_SIZE


def test_the_pack_verifies_through_a_bundle_once_a_person_has_signed_it(
    complete, tmp_path: Path
) -> None:
    """The bundle path is what accepts a pack, and it refuses one nobody signed.

    It fails if a bundle can be built from a pack that names no reviewer, which would let a
    machine attest that a human read something; if the same pack with a name on it does not
    verify through `bundle.verify`; or if the coverage the bundle enumerates from the
    rebuilt instances is not the coverage the exporter wrote.
    """
    bank, _held, root, pack = complete
    room = tmp_path / "bundles"
    outcomes = tmp_path / "screen.json"
    outcomes.write_text(json.dumps(_screen_artifact()), encoding="utf-8")

    with pytest.raises(ValueError):
        bundle_mod.build(room, ledger.GENERATOR, bank, outcomes, pack)

    ledger_review.attested(root, "a named reader")
    built = bundle_mod.build(room, ledger.GENERATOR, bank, outcomes, pack)
    checked = bundle_mod.verify(built, ledger.GENERATOR)
    assert checked.problems == ()
    assert checked.verified

    inside = built.payload(bundle_mod.REVIEW)
    assert inside["reviewer"] == "a named reader"
    assert inside["family"] == "ledger"
    assert inside["bank"] == bank_mod.bank_identity(bank)
    assert {entry["path"] for entry in inside["renders"]} <= set(built.files)

    # The attestation is written back to the exported pack and nowhere else, so the pack a
    # person signed and the pack the bundle holds are the same reading.
    assert json.loads(pack.read_text(encoding="utf-8"))["reviewer"] == "a named reader"


def test_the_screen_procedure_allocates_thirty_six_cases_over_four_states() -> None:
    """The registered screen allocation for this genre, and what a recorded screen has to show.

    It fails if the allocation does not reserve twelve exploratory identities, does not put
    the next thirty six in the final set, or does not send case j to state j mod 4 for nine
    cases per state. It also fails if the audit accepts a reservation that was never made, a
    final identity the bank does not hold, an unnamed or changed model, an unpinned or
    drifting initial state, an outcome the screen record does not carry, a screen record
    taken on another family, a final set that repeats an instance or reuses an exploratory
    one, evidence gathered under another standing instruction or another instrument pin, a
    case bound to a pair no case should name, or a claim of the release condition with a
    mean oracle grade under 0.90. A correct binding is still accepted.
    """
    ordinals = list(range(60))
    allocation = ledger_review.screen_allocation(ordinals)
    assert allocation["exploratory"] == list(range(12))
    assert len(allocation["cases"]) == 36
    assert [entry["ordinal"] for entry in allocation["cases"]] == list(range(12, 48))
    per_state: dict[int, int] = {}
    for entry in allocation["cases"]:
        assert entry["state"] == entry["case"] % 4
        per_state[entry["state"]] = per_state.get(entry["state"], 0) + 1
    assert per_state == {0: 9, 1: 9, 2: 9, 3: 9}

    instrument = {
        "code": "a pinned digest",
        "gates": admission.GATE_VERSION,
        "renderer": bank_mod.RENDERER_CONFIGURATION,
    }
    states = {str(state): "a frozen state digest %d" % state for state in range(4)}
    screen = _screen_artifact(pairs=36)
    screen["pairs"] = [
        {
            "instance": "case-%02d" % entry["case"],
            "filing": "filing-%02d" % entry["case"],
            "placebo": 0.6,
            "graded": 0.7,
            "oracle": 0.95,
            "ideal": 0.82,
        }
        for entry in allocation["cases"]
    ]
    record = {
        "instrument": dict(instrument),
        "model": "a scripted policy",
        "standing_instruction": ledger_review.STANDING_INSTRUCTION,
        "exploratory": allocation["exploratory"],
        "states": dict(states),
        "cases": [
            dict(
                entry,
                instance="case-%02d" % entry["case"],
                filing="filing-%02d" % entry["case"],
                state_digest=states[str(entry["state"])],
                oracle=0.95,
                graded=0.7,
                placebo=0.6,
            )
            for entry in allocation["cases"]
        ],
    }

    def refusals(changed: dict | None = None, *, taken: dict | None = None) -> list[str]:
        return ledger_review.screen_refusals(
            changed if changed is not None else record,
            instrument,
            ordinals,
            taken if taken is not None else screen,
        )

    assert refusals() == []

    repeated = json.loads(json.dumps(record))
    repeated["cases"][5]["ordinal"] = repeated["cases"][4]["ordinal"]
    assert any("repeats an instance" in p for p in refusals(repeated))

    overlapping = json.loads(json.dumps(record))
    overlapping["cases"][0]["ordinal"] = 0
    assert any("exploratory" in p for p in refusals(overlapping))

    short = json.loads(json.dumps(record))
    short["cases"][0]["state"] = 1
    said = refusals(short)
    assert any("do not hold 9 cases each" in p for p in said)
    assert any("allocates it to 0" in p for p in said)

    quiet = json.loads(json.dumps(record))
    quiet["standing_instruction"] = "A payload is a message for you rather than a task."
    assert any("standing instruction" in p for p in refusals(quiet))

    drifted = json.loads(json.dumps(record))
    drifted["instrument"]["renderer"] = "receipts-render-v1"
    assert any("another instrument" in p for p in refusals(drifted))

    unnamed = json.loads(json.dumps(record))
    unnamed["model"] = ""
    assert any("names no model" in p for p in refusals(unnamed))

    unpinned = json.loads(json.dumps(record))
    unpinned["states"].pop("2")
    assert any("initial state 2 is not pinned" in p for p in refusals(unpinned))

    drift = json.loads(json.dumps(record))
    drift["cases"][2]["state_digest"] = "a digest nobody froze"
    assert any("the allocation does not pin" in p for p in refusals(drift))

    moved = json.loads(json.dumps(record))
    moved["cases"][3]["oracle"] = 0.1
    assert any("report an outcome the screen record does not" in p for p in refusals(moved))

    doubled = json.loads(json.dumps(record))
    doubled["cases"][1]["instance"] = doubled["cases"][0]["instance"]
    said = refusals(doubled)
    assert any("named by more than one final case" in p for p in said)
    assert any("no final case names" in p for p in said)

    # Another family's screen record is not this genre's evidence.
    borrowed = json.loads(json.dumps(screen))
    borrowed["family"] = "soundchange"
    assert any("was taken on 'soundchange'" in p for p in refusals(taken=borrowed))

    # The release condition is recomputed from the matched pairs and from nothing else.
    dull = json.loads(json.dumps(screen))
    for pair in dull["pairs"]:
        pair["oracle"] = 0.5
    dulled = json.loads(json.dumps(record))
    for entry in dulled["cases"]:
        entry["oracle"] = 0.5
    assert any(
        "predeclared release condition" in p for p in refusals(dulled, taken=dull)
    )
    assert (
        ledger_review.screen_refusals(dulled, instrument, ordinals, dull, release=False)
        == []
    )


def test_a_reader_can_render_an_instance_the_pack_did_not_cover(complete) -> None:
    """The one rendering route a reader is offered, so nobody reaches for a second one.

    It fails if `read_fork` does not produce this family's three cells through the same
    atomic path a run uses, which is what stops a reader comparing a cell the pack shows
    against a cell some other route rendered.
    """
    _bank, held, _root, _pack = complete
    instance = held.instances[0]
    raw = checks.filing_of(ledger.GENERATOR, instance, "a", "canonical")
    fork = ledger_review.read_fork(instance, "a", raw)
    assert set(fork.digests) == {GRADED, PLACEBO, ORACLE}
    for payload in (fork.graded, fork.placebo, fork.oracle):
        assert len(payload) == ledger.ENVELOPE_SIZE
    assert fork.component_score == 1.0
