"""The second version of the sound change genre: a third environment, and 54 conventions.

Each test states the failure it is here to catch. The first version's own module holds
the mechanics both versions share; these hold what the second adds, what it must not
move in the first, and what the audit derives for it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from pathlib import Path

import pytest

from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import bundle as bundle_mod
from shogym.envs.receipts import checks, copy_profiles, review, soundchange_review, streams
from shogym.envs.receipts.generators import soundchange, soundchange_v2
from shogym.envs.receipts.generators import soundchange_audit as audit
from shogym.envs.receipts.protocol import (
    Axis,
    conventions,
    draw,
    option_mentions,
    policy_of,
)
from shogym.envs.receipts.receipt_law import (
    REGISTERED_MAX_IDEAL,
    REGISTERED_MIN_IDEAL,
    bank_law,
    law_for,
)
from shogym.envs.receipts.registry import GENRES, load_generator

MASTER = bytes(range(32))
GENERATOR = soundchange_v2.GENERATOR

#: The nine forms of the first version's compact illustration. They separate all 54
#: conventions of this version as they separate the first version's 36.
ILLUSTRATION = (
    "nbaketip", "piketap", "pekatip", "pikatep", "pekitap",
    "pakitep", "kepitap", "kipatep", "kapetip",
)

#: What the eighteen new conventions make of those nine forms, written out so that a
#: change to either cascade fails against fixed values rather than against the other
#: implementation. A row reads by hand: under replacement first only a k that begins the
#: form is replaced, and under deletion first a k whose left vowel went is replaced too,
#: unless its right vowel went with it. The first version's 36 rows are its own module's
#: fixture and hold here as well, because an environment that is one of the first two
#: means what it meant.
NO_LEFT_VOWEL_KEYS = """
g e RD mbaktip piktap pkatip pikatp pkitap pakitp gpitap gipatp gaptip
g e DR mbaktip piktap pgatip pikatp pgitap pakitp kpitap gipatp gaptip
g i RD mbaketp pketap pekatp pkatep pektap paktep geptap gpatep gapetp
g i DR mbaketp pgetap pekatp pgatep pektap paktep geptap kpatep gapetp
g a RD mbketip piketp pektip piktep pekitp pkitep gepitp giptep gpetip
g a DR mbgetip piketp pektip piktep pekitp pgitep gepitp giptep kpetip
x e RD mbaktip piktap pkatip pikatp pkitap pakitp xpitap xipatp xaptip
x e DR mbaktip piktap pxatip pikatp pxitap pakitp kpitap xipatp xaptip
x i RD mbaketp pketap pekatp pkatep pektap paktep xeptap xpatep xapetp
x i DR mbaketp pxetap pekatp pxatep pektap paktep xeptap kpatep xapetp
x a RD mbketip piketp pektip piktep pekitp pkitep xepitp xiptep xpetip
x a DR mbxetip piketp pektip piktep pekitp pxitep xepitp xiptep kpetip
s e RD mbaktip piktap pkatip pikatp pkitap pakitp spitap sipatp saptip
s e DR mbaktip piktap psatip pikatp psitap pakitp kpitap sipatp saptip
s i RD mbaketp pketap pekatp pkatep pektap paktep septap spatep sapetp
s i DR mbaketp psetap pekatp psatep pektap paktep septap kpatep sapetp
s a RD mbketip piketp pektip piktep pekitp pkitep sepitp siptep spetip
s a DR mbsetip piketp pektip piktep pekitp psitep sepitp siptep kpetip
"""


def _generated_forms(count: int = 1500, seed: int = 20260927) -> list[str]:
    """Distinct forms from the batch grammar itself, under a fixed stream."""
    picker = random.Random(seed)
    finals = soundchange.FINAL_A + soundchange.FINAL_B
    return sorted({soundchange.invent_word(picker, finals) for _ in range(count)})


# ----- the support ------------------------------------------------------------


def test_the_support_is_fifty_four_and_the_first_version_keeps_thirty_six() -> None:
    """The count the docstring states, beside the count the first version keeps.

    It fails if this version's support is not the product of 3, 3, 3 and 2 options, if
    its docstring or the audit's support disagrees with that count or that order, if the
    first version's support moved, or if the two versions are not both registered under
    names of their own.
    """
    options = {axis.name: axis.options for axis in soundchange_v2.AXES}
    assert options["environment"] == ("vowel_pair", "right_vowel", "no_left_vowel")
    assert options["reflex"] == soundchange.AXES[0].options
    assert options["loss"] == soundchange.AXES[2].options
    assert options["sequence"] == soundchange.AXES[3].options
    assert len(soundchange_v2.ALL_CONVENTIONS) == 54
    assert "That is 54 conventions" in (soundchange_v2.__doc__ or "")
    assert list(soundchange_v2.SUPPORT.conventions) == list(soundchange_v2.ALL_CONVENTIONS)
    assert list(soundchange_v2.ALL_CONVENTIONS) == conventions(GENERATOR.AXES)
    assert not hasattr(GENERATOR, "SUPPORT")

    assert len(soundchange.ALL_CONVENTIONS) == 36
    assert soundchange.GENERATOR.AXES[1].options == ("vowel_pair", "right_vowel")
    assert list(audit.FIRST_VERSION.conventions) == list(soundchange.ALL_CONVENTIONS)
    assert "That is 36 conventions" in (soundchange.__doc__ or "")

    assert GENRES["soundchange_v2"] == soundchange_v2.__name__
    assert load_generator("soundchange_v2") is GENERATOR
    assert load_generator("soundchange") is soundchange.GENERATOR
    assert copy_profiles.profile_of(GENERATOR) == copy_profiles.SOUNDCHANGE_V1
    assert policy_of(GENERATOR) is policy_of(soundchange.GENERATOR)


def test_the_second_version_is_in_its_own_pin_and_not_in_anyone_else_s() -> None:
    """What a bundle of either version hashes, and what moving the second one moves.

    It fails if the second version's module or the audit it filters with is outside its
    own code pin, if the gate label or the renderer configuration moved to register it,
    or if the second version's module is inside the pin of the first version or of
    ledger: a later change to it would then refuse their bundles as stale for a change
    that decides nothing about them.
    """
    from shogym.envs.receipts import admission
    from shogym.envs.receipts.generators import ledger

    assert admission.GATE_VERSION == "receipts-gates-v4"
    assert bank_mod.RENDERER_CONFIGURATION == "receipts-render-v3"
    pinned = bank_mod.pinned_modules(GENERATOR)
    for name in (
        "shogym.envs.receipts.generators.soundchange_v2",
        "shogym.envs.receipts.generators.soundchange",
        "shogym.envs.receipts.generators.soundchange_audit",
    ):
        assert name in pinned
    for other in (soundchange.GENERATOR, ledger.GENERATOR):
        assert soundchange_v2.__name__ not in bank_mod.pinned_modules(other)


def test_every_pair_of_conventions_is_separated_by_forms_the_grammar_makes() -> None:
    """No two of the 54 rules answer every form alike, and the two ruled out would.

    It fails if two conventions give identical daughters on every form of a generated
    sample, or on the nine illustration forms, or if either candidate the docstring rules
    out would not collapse: a k needing only a vowel on its left is the two-sided
    environment whenever replacement runs first, so it leaves 45 distinct rules of 54,
    and an unconditional replacement commutes with the deletion, so it leaves 36.
    """
    forms = _generated_forms()
    keys = [
        tuple(soundchange_v2.daughter(w, c) for w in forms)
        for c in soundchange_v2.ALL_CONVENTIONS
    ]
    assert len(set(keys)) == 54
    illustrated = {
        tuple(soundchange_v2.daughter(w, c) for w in ILLUSTRATION)
        for c in soundchange_v2.ALL_CONVENTIONS
    }
    assert len(illustrated) == 54

    first = {
        tuple(audit.audit_daughter(w, c) for w in forms) for c in soundchange.ALL_CONVENTIONS
    }
    for pattern, expected in ((r"(?<=[aeiou])k", 45), (r"k", 36)):
        candidate = re.compile(pattern)
        added = set()
        for convention in soundchange.ALL_CONVENTIONS:
            if convention["environment"] != "vowel_pair":
                continue
            phone = audit.introduced_phone(convention["reflex"])
            answers = []
            for word in forms:
                word = audit.audit_nasal(word)
                if audit.replaces_first(convention["sequence"]):
                    word = audit.audit_delete(candidate.sub(phone, word), convention["loss"])
                else:
                    word = candidate.sub(phone, audit.audit_delete(word, convention["loss"]))
                answers.append(word)
            added.add(tuple(answers))
        assert len(first | added) == expected, pattern


def test_the_worked_keys_of_the_third_environment_are_the_stated_ones() -> None:
    """The eighteen new rows, worked by hand, on the nine illustration forms.

    It fails if the production cascade or the independent one makes any of the eighteen
    worked daughter vectors differently, or if a convention whose environment is one of
    the first version's two answers differently here than it does there.
    """
    for line in NO_LEFT_VOWEL_KEYS.strip().splitlines():
        reflex, vowel, order, *expected = line.split()
        convention = {
            "reflex": f"out_{reflex}",
            "environment": soundchange_v2.NO_LEFT_VOWEL,
            "loss": f"erase_{vowel}",
            "sequence": "replace_then_erase" if order == "RD" else "erase_then_replace",
        }
        assert tuple(soundchange_v2.daughter(w, convention) for w in ILLUSTRATION) == tuple(
            expected
        ), line
        assert tuple(audit.audit_daughter(w, convention) for w in ILLUSTRATION) == tuple(
            expected
        ), line
    forms = _generated_forms(400)
    for convention in soundchange.ALL_CONVENTIONS:
        for word in forms:
            assert soundchange_v2.daughter(word, convention) == soundchange.daughter(
                word, convention
            )


# ----- the cascade, against the second implementation ------------------------


def test_the_production_cascade_and_the_audit_agree_on_all_fifty_four() -> None:
    """Two implementations of the third environment, and the strings that separate them.

    It fails if any of the 54 cascades disagrees with the lookaround validator on a
    string of length zero through five over `aeiknb`, on the generated forms, or on the
    cases that decide the new condition: a k at the start of a form, a k after a
    consonant, a k after o or u, which no draw deletes, and a k at the end.
    """
    letters = "aeiknb"
    strings = [""]
    for length in range(1, 6):
        strings.extend("".join(p) for p in itertools.product(letters, repeat=length))
    strings.extend(ILLUSTRATION)
    strings.extend(_generated_forms(300))
    strings.extend(["ka", "tka", "kka", "oka", "uka", "ak", "tak", "kok", "nbka", "bkeka"])
    for convention in soundchange_v2.ALL_CONVENTIONS:
        for word in strings:
            assert soundchange_v2.daughter(word, convention) == audit.audit_daughter(
                word, convention
            ), (word, convention)


def test_the_third_environment_wants_a_vowel_on_the_right_and_none_on_the_left() -> None:
    """The one condition a word boundary meets, and the ones it does not.

    It fails if a k at the start of a form or after a consonant is left alone, if a k
    between two vowels or before a consonant or at the end of a form is replaced, or if
    only the first of two matching k is replaced.
    """
    replace = soundchange_v2.replace_pass
    new = soundchange_v2.NO_LEFT_VOWEL
    assert replace("ka", "g", new) == "ga"
    assert replace("tka", "g", new) == "tga"
    assert replace("aka", "g", new) == "aka"
    assert replace("oku", "g", new) == "oku"
    assert replace("akt", "g", new) == "akt"
    assert replace("tak", "g", new) == "tak"
    assert replace("kk", "g", new) == "kk"
    assert replace("kka", "g", new) == "kga"
    assert replace("katka", "g", new) == "gatga"
    # The first version's two keep their meaning here.
    assert replace("ka", "g", "vowel_pair") == "ka"
    assert replace("aka", "g", "vowel_pair") == "aga"
    assert replace("tka", "g", "right_vowel") == "tga"
    with pytest.raises(ValueError):
        replace("ka", "g", "left_vowel")

    # The order separates on a k whose left vowel the deletion takes.
    witness = {"reflex": "out_g", "environment": new, "loss": "erase_i"}
    first, second = (dict(witness, sequence=s) for s in soundchange.AXES[3].options)
    assert soundchange_v2.daughter("tikap", first) == "tkap"
    assert soundchange_v2.daughter("tikap", second) == "tgap"
    # And on a k that begins the form whose right vowel the deletion takes.
    assert soundchange_v2.daughter("kipat", first) == "gpat"
    assert soundchange_v2.daughter("kipat", second) == "kpat"


def test_an_axis_the_audit_cannot_derive_is_a_failed_check_not_a_price() -> None:
    """A generator that declares an environment the audit has no operation for.

    It fails if the audit builds a support for an option it cannot turn into an
    operation, or if the three checks the profile brings pass or raise on one instead of
    failing by name.
    """
    with pytest.raises(ValueError):
        audit.Support(
            (
                soundchange.AXES[0],
                Axis("environment", ("vowel_pair", "left_vowel")),
                soundchange.AXES[2],
                soundchange.AXES[3],
            )
        )

    class Undeclared(soundchange_v2.SoundChangeV2Generator):
        AXES = (
            soundchange.AXES[0],
            Axis("environment", ("vowel_pair", "right_vowel", "left_vowel")),
            soundchange.AXES[2],
            soundchange.AXES[3],
        )

    instance = draw(GENERATOR, MASTER, 0)
    for check in (audit.check_cascade, audit.check_phone_lookup, audit.check_analogy):
        result = check(Undeclared(), instance)
        assert not result.passed
        assert "not a sound change support" in result.detail


# ----- the pair and the instance ---------------------------------------------


def test_a_generated_pair_separates_all_54_and_moves_on_every_axis() -> None:
    """What the construction filter promises, over this version's whole support.

    It fails if either side repeats a form, if the sides intersect, if either side's
    54 answer vectors are not distinct, if an option change moves fewer than four rows
    somewhere in the support, if a daughter outgrows the registered correction slot, or
    if a reader with no receipt scores above the registered bar.
    """
    a_table, b_table = soundchange_v2.build_pair(MASTER, 0)
    a = tuple(row.proto for row in a_table.rows)
    b = tuple(row.proto for row in b_table.rows)
    assert not set(a) & set(b)
    assert audit.pair_refusal(a, b, soundchange_v2.SUPPORT) == ""
    for forms in (a, b):
        assert len(set(forms)) == 24
        keys = audit.support_keys(forms, soundchange_v2.SUPPORT)
        assert len(keys) == 54
        assert audit.distinct_keys(keys) == 54
        for axis, (fewest, _) in audit.movement(keys, soundchange_v2.SUPPORT).items():
            assert fewest >= audit.MIN_MOVED_ROWS, axis
        assert max(len(v) for key in keys for v in key) <= soundchange.MAX_DAUGHTER
        assert audit.no_receipt_optimum(keys) <= audit.MAX_NO_RECEIPT


def test_the_second_version_draws_under_its_own_name_and_the_first_is_untouched() -> None:
    """Two versions under one key share only the convention stream every genre shares.

    It fails if the second version's forms, identifiers, filler, task identifiers or
    masks are drawn from the first version's coordinates, or if the first version stops
    drawing its filler from the coordinates it has always used.
    """
    first = draw(soundchange.GENERATOR, MASTER, 0)
    second = draw(GENERATOR, MASTER, 0)
    assert first.a.task_id != second.a.task_id
    for field in ("proto", "row_id"):
        assert not {getattr(r, field) for r in first.a.table.rows} & {
            getattr(r, field) for r in second.a.table.rows
        }
    assert first.envelope.filler != second.envelope.filler
    for instance, name in ((first, "soundchange"), (second, "soundchange_v2")):
        assert instance.envelope.filler == streams.filler_stream(
            MASTER, soundchange.FILLER_ALPHABET, soundchange.ENVELOPE_SIZE, name, 0, "pad"
        )
    assert second.envelope.size == first.envelope.size


def test_the_task_states_three_environments_and_names_no_option() -> None:
    """The mechanics a reader is served, and nothing the draw decided.

    It fails if the task text does not list the third environment and say what a word
    boundary does to it, if it prints an option identifier, or if it moves with the
    drawn convention.
    """
    instance = draw(GENERATOR, MASTER, 0)
    text = instance.a.text
    assert "one of these three possibilities" in text
    assert "the k is the\n  first phone of the form" in text
    assert "is the one condition a word boundary meets" in text
    assert option_mentions(GENERATOR.AXES, text) == []
    assert option_mentions(GENERATOR.AXES, instance.b.text) == []
    assert checks.check_invariance(GENERATOR, instance).passed
    assert checks.check_lint(GENERATOR, instance).passed


def test_the_oracle_states_each_of_the_54_rules_and_reads_it_back() -> None:
    """Every rule the oracle arm can state, rendered and parsed with one table.

    It fails if a rendered statement names any option but the drawn one on an axis, if
    the parser reads back another rule, or if the statement outgrows the registered body
    allowance.
    """
    instance = draw(GENERATOR, MASTER, 0)
    assert checks.check_oracle(GENERATOR, instance).passed
    for convention in soundchange_v2.ALL_CONVENTIONS:
        ast = GENERATOR.render_oracle(instance.a.task_id, convention, 24)
        assert GENERATOR.parse_oracle(ast) == convention
        rendered = " ".join(" ".join(ast.body).split())
        assert len("\n".join(ast.body).encode("ascii")) < soundchange.ORACLE_BODY_ALLOWANCE
        for axis in GENERATOR.AXES:
            phrases = soundchange_v2.ORACLE_TEMPLATE.phrases[axis.name]
            present = [o for o, p in phrases.items() if " ".join(p.split()) in rendered]
            assert present == [convention[axis.name]], (axis.name, present)


# ----- admission and the bank law ---------------------------------------------


#: The key this module's bank is filled under. Fixed, so the admission walk this module
#: pays for is the same walk on every run.
BANK_MASTER = hashlib.sha256(b"soundchange_v2 bank").digest()
SHARES_BANK = pytest.mark.xdist_group("receipts-soundchange-v2-bank")


@pytest.fixture(scope="module")
def filled():
    """A bank of two instances, filled through the registered admission rule."""
    return bank_mod.materialized(GENERATOR, BANK_MASTER, 2)


@SHARES_BANK
def test_a_bank_of_the_second_version_fills_under_the_registered_rule(filled) -> None:
    """Admission, the profile's three checks and the band, over all 54 rules.

    It fails if a bank of this version cannot be filled under the registered bars, if
    the band on the bank-mean ideal level is not met, or if the three checks the copy
    profile brings are asked about fewer than the 54 rules the family draws from.
    """
    bank, held = filled
    assert bank.generator == "soundchange_v2"
    assert len(held.instances) == 2
    laws = [law_for(GENERATOR, instance, "a") for instance in held.instances]
    band = bank_law(laws)
    assert band.passed, band.line()
    assert REGISTERED_MIN_IDEAL <= band.ideal <= REGISTERED_MAX_IDEAL
    instance = held.instances[0]
    cascade = audit.check_cascade(GENERATOR, instance)
    assert cascade.passed and "on all 54 draws" in cascade.detail
    lookup = audit.check_phone_lookup(GENERATOR, instance)
    assert lookup.passed and "over all 276 masks" in lookup.detail
    assert audit.check_analogy(GENERATOR, instance).passed


@SHARES_BANK
def test_a_pack_is_exported_for_the_second_version_and_its_bundle_verifies(
    filled, tmp_path: Path
) -> None:
    """The review pack a person reads before this version is dealt, and the bundle.

    It fails if the exporter refuses a bank of this version, if the pack misses any
    coverage the shared review asks for, if it carries fewer than the 54 oracle cells a
    reader has to see, if it names a reviewer, or if the bundle built over the bank, a
    screen record and the attested pack does not verify under the registered rule.
    """
    bank, held = filled
    root = tmp_path / "pack"
    pack = soundchange_review.export(bank, held, root)
    manifest = json.loads(pack.read_text(encoding="utf-8"))
    assert manifest["family"] == "soundchange_v2"
    assert manifest["reviewer"] is None
    coverage = review.required_coverage(
        GENERATOR, checks.FILING_CLASSES, [soundchange.ROWS]
    )
    seen = [(e["category"], e["key"]) for e in manifest["renders"]]
    assert coverage.missing(seen) == []
    oracles = {e["path"] for e in manifest["renders"] if e["path"].startswith("renders/oracle-")}
    assert len(oracles) == 54
    assert ("option", "environment=no_left_vowel") in seen

    screen = tmp_path / "screen.json"
    screen.write_text(
        json.dumps(
            {
                "family": GENERATOR.name,
                "model": "a scripted policy",
                "task_seeds": [str(i) for i in range(40)],
                "pairs": [
                    {"instance": f"task-{i:02d}", "filing": f"filing-{i:02d}",
                     "placebo": 0.4, "graded": 0.6, "oracle": 0.95, "ideal": 0.82}
                    for i in range(40)
                ],
                "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36,
                "min_oracle": 0.90, "min_learning_gap": 0.10,
                "floor": 0.0, "floor_rule": "drop",
                "candidates_screened": 1, "selection_note": "",
            }
        ),
        encoding="utf-8",
    )
    soundchange_review.attested(root, "a named reader")
    # The build verifies the bundle before it returns, and raises when it does not.
    built = bundle_mod.build(tmp_path / "bundles", GENERATOR, bank, screen, pack)
    assert built.root.is_dir()
