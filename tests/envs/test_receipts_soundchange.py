"""The sound change genre, the shared filing helpers, and the copy profile contract.

Each test states the failure it is here to catch, because a test whose failure
condition is "something changed" is a test nobody can act on.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import random
import shutil
import sys
import threading
import types
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import pytest

from shogym.cli import main
from shogym.envs.receipts import admission
from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import bundle as bundle_mod
from shogym.envs.receipts import review, soundchange_review
from shogym.envs.receipts import checks, copy_profiles
from shogym.envs.receipts.generators import ledger, soundchange
from shogym.envs.receipts.generators import soundchange_audit as audit
from shogym.envs.receipts.generators.vectors import VECTORS
from shogym.envs.receipts.protocol import option_mentions
from shogym.envs.receipts.protocol import (
    ConstructionExhausted,
    Instance,
    PublicTask,
    Task,
    draw,
    receipt_mask,
)
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.streams import digest
from shogym.envs.receipts.receipt_ast import (
    ReceiptAST,
    frozen_envelope,
    mask_slots,
    row_lines,
    serialize,
    slot_ranges,
)
from shogym.envs.receipts.registry import FIXTURES as FIXTURES_NAMES
from shogym.envs.receipts.registry import (
    BANK_DIR_VAR,
    GENRES,
    bank_path,
    load_generator,
    provenance_path,
)
from shogym.envs.receipts.env_v1 import ReceiptsV1Env
from shogym.envs.receipts.render import feedback_for, judge_cells

from shogym.serve import ServedEpisode

FIXTURES = Path(__file__).resolve().parents[1] / "_fixtures"


# ----- the ledger regression over the shared filing helpers -----------------


def _frozen_ledger() -> dict:
    return json.loads(
        (FIXTURES / "ledger_filing_before_extraction.json").read_text(encoding="utf-8")
    )


def test_moving_the_filing_helpers_out_changed_nothing_ledger_does() -> None:
    """The reading, the grades, the cell bytes and the legacy copy maxima, frozen.

    It fails if extracting the common line reading, printable normalization,
    identifier parser and equal-row scorer changed any ledger parser outcome, any
    canonical grade, any rendered cell byte or any legacy copy maximum. The
    expectations were taken from the module before the extraction, on two instances
    under a fixed key, over the seven registered filing classes and nine further
    readings the parser has to decide: a Python list, outer whitespace with the case
    changed, a repeated identifier, a comma-free filing at full length and at part
    length, prose, a mapping, an unknown identifier, an explicitly empty value and a
    value carrying characters outside printable ASCII.
    """
    frozen = _frozen_ledger()
    generator = ledger.GENERATOR
    master = bytes.fromhex(frozen["master"])
    for ordinal, want in sorted(frozen["instances"].items()):
        instance = draw(generator, master, int(ordinal))
        assert dict(instance.convention) == want["convention"]
        assert checks.copy_scores(generator, instance) == want["copy_scores"]
        assert checks.axis_leverage(generator, instance) == want["axis_leverage"]
        assert ledger.scope_sentence("A") == want["scope"]["A"]
        assert ledger.scope_sentence("B") == want["scope"]["B"]
        for side, side_want in sorted(want["sides"].items()):
            task = instance.side(side)
            identifiers = list(generator.row_identifiers(task.table))
            assert identifiers == side_want["identifiers"]
            assert list(task.key) == side_want["key"]
            assert (
                hashlib.sha256(task.text.encode()).hexdigest()
                == side_want["text_digest"]
            )
            for shape, expected in sorted(side_want["classes"].items()):
                raw = checks.filing_of(generator, instance, side, shape)
                canonical = generator.parse_and_canonicalize(task, raw)
                score, _ = generator.score(task, canonical)
                judged = judge_cells(
                    generator, task, canonical, instance.convention, instance.envelope
                )
                assert _reading(canonical, score) == _reading_of(expected), (
                    ordinal,
                    side,
                    shape,
                )
                assert {
                    kind: hashlib.sha256(payload).hexdigest()
                    for kind, payload in judged.payloads.items()
                } == expected["cells"], (ordinal, side, shape)
                assert list(judged.problems) == expected["problems"]
            for name, expected in sorted(side_want["odd"].items()):
                raw = _odd_filing(name, identifiers, list(task.key))
                canonical = generator.parse_and_canonicalize(task, raw)
                score, _ = generator.score(task, canonical)
                assert _reading(canonical, score) == _reading_of(expected), (
                    ordinal,
                    side,
                    name,
                )


def _odd_filing(name: str, identifiers: list[str], key: list[str]) -> object:
    """The nine readings the frozen fixture recorded, rebuilt from the instance."""
    tail = "é\ud800"
    if name == "python_list":
        return ["%s,%s" % (i, v) for i, v in zip(identifiers, key)]
    if name == "outer_space_and_case":
        return "\n".join(
            "  %s  ,  %s  " % (i.upper(), v.upper()) for i, v in zip(identifiers, key)
        )
    if name == "first_duplicate_wins":
        return "\n".join(
            ["%s,%s" % (identifiers[0], key[0]), "%s,not the answer" % identifiers[0]]
        )
    if name == "positional":
        return "\n".join(v if v else "-" for v in key)
    if name == "short_positional":
        return "\n".join(key[:5])
    if name == "prose":
        return "a paragraph about the schedule\nwith no identifier in it"
    if name == "mapping":
        return {"one": 1}
    if name == "unknown_identifier":
        return "NOT-AN-ID,Routine"
    if name == "empty_value":
        return "%s," % identifiers[0]
    if name == "unprintable_tail":
        return "\n".join("%s,%s%s" % (i, v, tail) for i, v in zip(identifiers, key))
    raise AssertionError(f"the fixture names a reading this test cannot rebuild: {name}")


def _reading(canonical, score: float) -> dict:
    return {
        "is_filing": canonical.is_filing,
        "reason": getattr(canonical, "reason", ""),
        "score": score,
        "values": list(getattr(canonical, "values", ())),
        "filed": list(getattr(canonical, "filed", ())),
        "filed_rows": getattr(canonical, "filed_rows", 0),
        "duplicates": list(getattr(canonical, "duplicates", ())),
        "extras": list(getattr(canonical, "extras", ())),
        "omissions": list(getattr(canonical, "omissions", ())),
    }


def _reading_of(expected: dict) -> dict:
    return {key: expected[key] for key in _READING_FIELDS}


_READING_FIELDS = (
    "is_filing",
    "reason",
    "score",
    "values",
    "filed",
    "filed_rows",
    "duplicates",
    "extras",
    "omissions",
)


# ----- the shared copying contract ------------------------------------------


class _Undeclared:
    """A generator-shaped object that never says which family prices it."""

    name = "undeclared"
    genre = "a generator with no copy profile"

    def answer_ranks(self, table) -> tuple[str, ...]:
        return ("one", "two")


class _Unregistered(_Undeclared):
    name = "unregistered"
    COPY_PROFILE = "whatever_this_build_does_not_know"


class _RanksWithoutAProfileForThem(_Undeclared):
    """Declares the word profile and hands back the daughters a draw realized."""

    name = "leaky"
    COPY_PROFILE = copy_profiles.SOUNDCHANGE_V1


class _NoRanksUnderTheTokenProfile(_Undeclared):
    name = "rankless"
    COPY_PROFILE = copy_profiles.ORDERED_TOKENS

    def answer_ranks(self, table) -> None:
        return None


def test_a_generator_without_a_declared_profile_is_refused_at_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy family is declared, and registration is where the declaration is read.

    It fails if an undeclared profile falls back to any family at all, if a profile
    name this build does not register is accepted, or if `load_generator` hands back a
    generator whose declaration was never read. Silence is the failure being refused:
    a family of words priced by maps between two published vocabularies reports the
    maximum of a transfer nobody can perform, and the bar is then read against a
    number that measured nothing.

    THE LAST PART REGISTERS ONE. Asking the helper directly and then loading the
    generators this build already declares leaves the rule stated and unenforced: both
    halves stay green with the reading removed from the registry, because everything on
    the roster declares a profile and nothing else ever reaches the door. A generator
    that declares none is put on the roster here, and the refusal is asked of the door.
    """
    for offered in (_Undeclared(), _Unregistered()):
        with pytest.raises(ValueError) as refusal:
            copy_profiles.profile_of(offered)
        assert "COPY_PROFILE" in str(refusal.value) or "profile" in str(refusal.value)
        with pytest.raises(ValueError):
            copy_profiles.require_profile(offered)
    for name in sorted(set(GENRES) | set(FIXTURES_NAMES)):
        assert copy_profiles.profile_of(load_generator(name)) in copy_profiles.PROFILES

    for genre, offered in (("undeclared", _Undeclared()), ("unregistered", _Unregistered())):
        stub = types.ModuleType(f"a_module_declaring_{genre}")
        stub.GENERATOR = offered
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
        monkeypatch.setitem(GENRES, genre, stub.__name__)
        assert GENRES[genre] == stub.__name__
        with pytest.raises(ValueError) as refused:
            load_generator(genre)
        assert "COPY_PROFILE" in str(refused.value) or "profile" in str(refused.value)

    # A gate vector reaches the same door by its own map, and is read the same way: the
    # vectors are exercised through the shipped commands, so an undeclared one would be
    # priced under whichever family the copy screen tried first.
    monkeypatch.setitem(VECTORS, "undeclared_vector", _Undeclared())
    with pytest.raises(ValueError) as vector_refused:
        load_generator("undeclared_vector")
    assert "COPY_PROFILE" in str(vector_refused.value) or "profile" in str(
        vector_refused.value
    )


def test_realized_answers_are_never_consumed_as_published_ranks() -> None:
    """The two halves of the declaration have to agree with each other.

    It fails if a family declaring the word profile may publish answer ranks anyway,
    which is how the daughters one hidden draw realized would reach the copy screen as
    though a reader could have read them off the task, and it fails if a family
    declaring the token profile may publish none, which is a bar read against maps
    that were never built.
    """
    with pytest.raises(ValueError) as leaked:
        copy_profiles.answer_ranks_for(_RanksWithoutAProfileForThem(), None)
    assert "ordered vocabulary" in str(leaked.value)
    with pytest.raises(ValueError) as missing:
        copy_profiles.answer_ranks_for(_NoRanksUnderTheTokenProfile(), None)
    assert "no answer ranks" in str(missing.value)


def test_the_character_maps_are_closed_under_composition() -> None:
    """The registered character family is a family, not a list of maps.

    It fails if composing two registered character maps leaves the registered set, if
    the identity is not in it, or if the count is not the direct product of the two
    permutation groups. A family that is not closed reports a maximum some composition
    of its own members exceeds, which is the failure that admitted two ledger draws.
    """
    maps = copy_profiles.character_maps()
    assert len(maps) == 36
    frozen = {tuple(sorted(table.items())) for table in maps}
    assert len(frozen) == 36
    identity = {c: c for c in copy_profiles.DELETABLE_VOWELS + copy_profiles.REPLACEMENT_PHONES}
    assert tuple(sorted(identity.items())) in frozen
    for one in maps:
        for other in maps:
            composed = {k: other.get(one[k], one[k]) for k in one}
            assert tuple(sorted(composed.items())) in frozen


def test_character_maps_and_row_moves_commute_and_close_the_product() -> None:
    """The product of the two commuting closed families is the family the bar reads.

    It fails if a character map applied after a row move differs from the same move
    applied after the map, if the identity filing is missing, or if either subfamily
    escapes the combined closure. A subfamily that exceeded the closure would mean the
    number the bar is read against is not a number any member reaches.
    """
    values = ["mbagtip", "pigtap", "pgatip", "kpitap"]
    width = len(values)
    for table in copy_profiles.character_maps():
        for move in copy_profiles.permutations(values, width):
            mapped_then_moved = [copy_profiles.apply_characters(table, v) for v in move]
            moved_then_mapped = copy_profiles.permutations(
                [copy_profiles.apply_characters(table, v) for v in values], width
            )
            assert mapped_then_moved in moved_then_mapped
    relabels = copy_profiles.character_relabellings(values, width)
    combined = copy_profiles.distinct(
        filing
        for relabelled in relabels
        for filing in copy_profiles.permutations(relabelled, width)
    )
    as_tuples = {tuple(f) for f in combined}
    assert tuple(values) in as_tuples
    assert {tuple(f) for f in relabels} <= as_tuples
    assert {tuple(f) for f in copy_profiles.permutations(values, width)} <= as_tuples
    assert len(combined) <= 36 * 2 * width


# ----- the cascade, against a second implementation --------------------------

#: The nine forms of the specification's compact illustration, and the daughter each
#: one takes under every draw. Nine rows separate all 36, which is what makes this a
#: usable hand check; a served batch is 24 rows and its own shuffle.
ILLUSTRATION = (
    "nbaketip", "piketap", "pekatip", "pikatep", "pekitap",
    "pakitep", "kepitap", "kipatep", "kapetip",
)

ILLUSTRATED_KEYS = """
g VV e RD mbagtip pigtap pgatip pigatp pgitap pagitp kpitap kipatp kaptip
g VV e DR mbaktip piktap pkatip pigatp pkitap pagitp kpitap kipatp kaptip
g VV i RD mbagetp pgetap pegatp pgatep pegtap pagtep keptap kpatep kapetp
g VV i DR mbagetp pketap pegatp pkatep pektap paktep keptap kpatep kapetp
g VV a RD mbgetip pigetp pegtip pigtep pegitp pgitep kepitp kiptep kpetip
g VV a DR mbketip pigetp pektip piktep pegitp pkitep kepitp kiptep kpetip
g _V e RD mbagtip pigtap pgatip pigatp pgitap pagitp gpitap gipatp gaptip
g _V e DR mbaktip piktap pgatip pigatp pgitap pagitp kpitap gipatp gaptip
g _V i RD mbagetp pgetap pegatp pgatep pegtap pagtep geptap gpatep gapetp
g _V i DR mbagetp pgetap pegatp pgatep pektap paktep geptap kpatep gapetp
g _V a RD mbgetip pigetp pegtip pigtep pegitp pgitep gepitp giptep gpetip
g _V a DR mbgetip pigetp pektip piktep pegitp pgitep gepitp giptep kpetip
x VV e RD mbaxtip pixtap pxatip pixatp pxitap paxitp kpitap kipatp kaptip
x VV e DR mbaktip piktap pkatip pixatp pkitap paxitp kpitap kipatp kaptip
x VV i RD mbaxetp pxetap pexatp pxatep pextap paxtep keptap kpatep kapetp
x VV i DR mbaxetp pketap pexatp pkatep pektap paktep keptap kpatep kapetp
x VV a RD mbxetip pixetp pextip pixtep pexitp pxitep kepitp kiptep kpetip
x VV a DR mbketip pixetp pektip piktep pexitp pkitep kepitp kiptep kpetip
x _V e RD mbaxtip pixtap pxatip pixatp pxitap paxitp xpitap xipatp xaptip
x _V e DR mbaktip piktap pxatip pixatp pxitap paxitp kpitap xipatp xaptip
x _V i RD mbaxetp pxetap pexatp pxatep pextap paxtep xeptap xpatep xapetp
x _V i DR mbaxetp pxetap pexatp pxatep pektap paktep xeptap kpatep xapetp
x _V a RD mbxetip pixetp pextip pixtep pexitp pxitep xepitp xiptep xpetip
x _V a DR mbxetip pixetp pektip piktep pexitp pxitep xepitp xiptep kpetip
s VV e RD mbastip pistap psatip pisatp psitap pasitp kpitap kipatp kaptip
s VV e DR mbaktip piktap pkatip pisatp pkitap pasitp kpitap kipatp kaptip
s VV i RD mbasetp psetap pesatp psatep pestap pastep keptap kpatep kapetp
s VV i DR mbasetp pketap pesatp pkatep pektap paktep keptap kpatep kapetp
s VV a RD mbsetip pisetp pestip pistep pesitp psitep kepitp kiptep kpetip
s VV a DR mbketip pisetp pektip piktep pesitp pkitep kepitp kiptep kpetip
s _V e RD mbastip pistap psatip pisatp psitap pasitp spitap sipatp saptip
s _V e DR mbaktip piktap psatip pisatp psitap pasitp kpitap sipatp saptip
s _V i RD mbasetp psetap pesatp psatep pestap pastep septap spatep sapetp
s _V i DR mbasetp psetap pesatp psatep pektap paktep septap kpatep sapetp
s _V a RD mbsetip pisetp pestip pistep pesitp psitep sepitp siptep spetip
s _V a DR mbsetip pisetp pektip piktep pesitp psitep sepitp siptep kpetip
"""

#: The specification's 24-row feasibility witness, an analytical fixture and not a bank.
FIXTURE_A = (
    "padepitokod", "popakited", "dekabepunikom", "pikekadom", "kodamekidebom",
    "tepikitapab", "natukidedimom", "nbapekim", "kibaked", "nikekapem",
    "nukikenad", "kebadukitimid", "nbekapanudid", "pababakikeb", "kakinbed",
    "kenonadibikib", "kekobatimom", "nbibekekab", "kupitakenom", "pupadikem",
    "biketabid", "manabiked", "kadekim", "nbokakibed",
)
FIXTURE_B = (
    "tutapiketakun", "kekudanbin", "pamidukepap", "dakomekip", "mubokapideken",
    "kinokapep", "mopotunikapen", "kanbukebin", "pabenbikumap", "bikadekun",
    "tamikep", "donakekip", "nbokibekat", "dimekakitip", "nadonobiket",
    "panipukep", "kemekabin", "mikebutokanen", "kenababikabet", "mimipobikaken",
    "dibumekekanban", "kekemapit", "tutikopekat", "bekikap",
)


def _illustrated() -> list[tuple[dict[str, str], tuple[str, ...]]]:
    out = []
    for line in ILLUSTRATED_KEYS.strip().splitlines():
        parts = line.split()
        out.append((
            {
                "reflex": {"g": "out_g", "x": "out_x", "s": "out_s"}[parts[0]],
                "environment": {"VV": "vowel_pair", "_V": "right_vowel"}[parts[1]],
                "loss": {"e": "erase_e", "i": "erase_i", "a": "erase_a"}[parts[2]],
                "sequence": {
                    "RD": "replace_then_erase", "DR": "erase_then_replace"
                }[parts[3]],
            },
            tuple(parts[4:]),
        ))
    return out


def test_the_production_cascade_and_an_independent_one_agree_everywhere() -> None:
    """Two implementations, every draw, and the strings most likely to separate them.

    It fails if any of the 36 cascades disagrees with the separate lookaround validator
    on any string of length zero through five over `aeiknb`, on the 57 displayed forms,
    or on the explicit o and u context cases, which are the vowels no draw deletes and
    which therefore decide whether a context is read as a class or as a letter.
    """
    letters = "aeiknb"
    strings = [""]
    for length in range(1, 6):
        strings.extend("".join(p) for p in itertools.product(letters, repeat=length))
    strings.extend(ILLUSTRATION)
    strings.extend(FIXTURE_A)
    strings.extend(FIXTURE_B)
    strings.extend(
        ["koka", "kuku", "okok", "ukuk", "boko", "buku", "tokot", "tukut",
         "nboko", "konbo", "okonbu", "kounbo", "aoukia", "kokokok"]
    )
    assert len(strings) == 1 + sum(len(letters) ** n for n in range(1, 6)) + 57 + 14
    for convention in soundchange.ALL_CONVENTIONS:
        for word in strings:
            assert soundchange.daughter(word, convention) == audit.audit_daughter(
                word, convention
            ), (word, convention)


def test_the_passes_read_their_own_input_and_the_nasal_pass_runs_once() -> None:
    """The four mistakes the task warns about, each as a case that separates them.

    It fails if an absent neighbor satisfies a vowel or consonant condition, if only
    one of several matching occurrences changes, if a later pass is tested against a
    partially rewritten string instead of the completed output of the previous pass, or
    if `neb` under e deletion becomes `mb` because the nasal pass was run again on a
    cluster the deletion created.
    """
    # An absent neighbor prevents a match, on both passes and at both ends.
    assert soundchange.replace_pass("ka", "g", "vowel_pair") == "ka"
    assert soundchange.replace_pass("ka", "g", "right_vowel") == "ga"
    assert soundchange.replace_pass("ak", "g", "right_vowel") == "ak"
    assert soundchange.replace_pass("ak", "g", "vowel_pair") == "ak"
    assert soundchange.delete_pass("eta", "e") == "eta"
    assert soundchange.delete_pass("ate", "e") == "ate"
    assert soundchange.delete_pass("tet", "e") == "tt"

    # Every matching occurrence changes, not just the first.
    assert soundchange.replace_pass("kaka", "g", "right_vowel") == "gaga"
    assert soundchange.replace_pass("kaka", "g", "vowel_pair") == "kaga"
    assert soundchange.delete_pass("bibib", "i") == "bbb"
    assert soundchange.nasal_pass("nbanb") == "mbamb"

    # A later pass sees the completed output of the previous one. Under replacement
    # first the deletion strands nothing; under deletion first the replacement loses the
    # vowel it needed, and the two orders separate.
    order_witness = {"reflex": "out_g", "environment": "right_vowel", "loss": "erase_i"}
    assert soundchange.daughter(
        "kika", dict(order_witness, sequence="replace_then_erase")
    ) == "gga"
    assert soundchange.daughter(
        "kika", dict(order_witness, sequence="erase_then_replace")
    ) == "kga"

    # The nasal pass is never run again, even on a cluster the deletion created.
    for sequence in ("replace_then_erase", "erase_then_replace"):
        assert soundchange.daughter(
            "neb",
            {"reflex": "out_g", "environment": "vowel_pair", "loss": "erase_e",
             "sequence": sequence},
        ) == "nb"
    assert soundchange.nasal_pass("neb") == "neb"


def test_the_support_is_thirty_six_and_every_worked_key_is_the_stated_one() -> None:
    """The declared support, and the 36 worked keys the specification prints.

    It fails if an unrestricted replacement environment is admitted alongside two order
    choices, which would put nine globally indistinguishable pairs of draws in the
    support, or if any of the 36 worked key vectors or their distinctness differs from
    the specification.
    """
    options = {axis.name: axis.options for axis in soundchange.AXES}
    assert options["environment"] == ("vowel_pair", "right_vowel")
    assert options["sequence"] == ("replace_then_erase", "erase_then_replace")
    assert len(soundchange.ALL_CONVENTIONS) == 36
    assert not hasattr(soundchange.GENERATOR, "SUPPORT")
    # An unconditional replacement commutes with the deletion on every word, which is
    # why it is not an option: the two orders would be the same function.
    for word in ("nbaketip", "kepitap", "kika", "kakinbed"):
        after = soundchange.nasal_pass(word)
        unconditional = after.replace("k", "g")
        for vowel in ("e", "i", "a"):
            assert soundchange.delete_pass(unconditional, vowel) == soundchange.delete_pass(
                after, vowel
            ).replace("k", "g")

    worked = _illustrated()
    assert len(worked) == 36
    seen = set()
    for convention, expected in worked:
        produced = tuple(soundchange.daughter(w, convention) for w in ILLUSTRATION)
        assert produced == expected, convention
        seen.add(produced)
    assert len(seen) == 36


# ----- the generated pair ----------------------------------------------------

MASTER = bytes(range(32))


def _table(label: str, protos: Sequence[str], ordinal: int = 0) -> soundchange.SoundTable:
    """A table over chosen forms, for asking a question of a pair nobody drew."""
    surface = soundchange.SURFACES[(label, ordinal % 2)]
    prefix = 0 if label == "A" else 1 << 20
    identifiers = ["F-%08x" % (prefix + n) for n in range(len(protos))]
    return soundchange.SoundTable(
        surface=surface.name,
        rows=tuple(
            soundchange.SoundRow(row_id=i, proto=p)
            for i, p in zip(identifiers, protos)
        ),
        body=soundchange.render_body(list(protos), identifiers, surface),
    )


def _made_instance(
    a_protos: Sequence[str],
    b_protos: Sequence[str],
    convention: Mapping[str, str] | None = None,
    ordinal: int = 0,
) -> Instance:
    """An instance over chosen forms, so a hand-made pair can be put to a check."""
    generator = soundchange.GENERATOR
    drawn = dict(convention or soundchange.ALL_CONVENTIONS[0])
    tasks = []
    for label, protos in (("A", a_protos), ("B", b_protos)):
        table = _table(label, protos, ordinal)
        key = soundchange.key_for(table, drawn)
        mask = receipt_mask(
            soundchange.GENERATOR.RECEIPT_POLICY, MASTER, generator.name, ordinal,
            label, len(key),
        )
        public = PublicTask(
            label=label, task_id="%016x" % (ordinal * 2 + (label == "B")),
            surface=table.surface, table=table, n_rows=len(key), mask=mask,
        )
        tasks.append(
            Task(
                label=label, task_id=public.task_id, surface=table.surface, table=table,
                text=generator.describe(public), key=key, mask=mask,
            )
        )
    return Instance(
        generator=generator.name, genre=generator.genre, ordinal=ordinal,
        convention=MappingProxyType(drawn), a=tasks[0], b=tasks[1],
        envelope=generator.build_envelope(MASTER, ordinal),
    )


def _drawn(ordinal: int = 0) -> Instance:
    return draw(soundchange.GENERATOR, MASTER, ordinal)


def test_a_generated_pair_is_disjoint_bounded_and_moves_on_every_axis() -> None:
    """What the construction filter promises about the two batches it accepted.

    It fails if either side repeats a form, if the two sides intersect, if a generated
    form is longer than 14 bytes or a generated daughter longer than the registered
    correction slot, or if any full-support movement minimum falls below four rows. The
    minima are taken over every draw and every alternative option, not under the draw
    this instance made.
    """
    for ordinal in range(3):
        instance = _drawn(ordinal)
        protos = {
            side: tuple(row.proto for row in instance.side(side).table.rows)
            for side in ("a", "b")
        }
        assert not set(protos["a"]) & set(protos["b"])
        for side, forms in protos.items():
            assert len(set(forms)) == len(forms) == 24
            assert max(len(f) for f in forms) <= 14
            assert min(len(f) for f in forms) >= 7
            assert all(set("aei") <= set(f) for f in forms)
            assert sum(1 for f in forms if "nb" in f) >= 4
            keys = audit.support_keys(forms)
            assert audit.distinct_keys(keys) == 36
            assert max(len(v) for key in keys for v in key) <= soundchange.MAX_DAUGHTER
            assert all(v for key in keys for v in key)
            for axis, (fewest, _) in audit.movement(keys).items():
                assert fewest >= audit.MIN_MOVED_ROWS, (ordinal, side, axis)
            assert audit.no_receipt_optimum(keys) <= audit.MAX_NO_RECEIPT
        assert tuple(row.proto for row in instance.a.table.rows) == protos["a"]


def test_the_construction_filter_never_reads_the_drawn_convention() -> None:
    """Acceptance is a function of the two batches, and of nothing the draw decided.

    It fails if changing only the selected convention changes the candidate acceptance
    decision, the task bytes, the printed identifiers, the row order or the committed
    filler. Acceptance is the load-bearing one: `describe` being invariant is not
    enough, because a filter that rejected candidates under some conventions would make
    the fact that this pair was served evidence about the draw.
    """
    generator = soundchange.GENERATOR
    for name in ("build_pair", "invent_word", "invent_batch"):
        parameters = set(inspect.signature(getattr(soundchange, name)).parameters)
        assert "convention" not in parameters, name
    assert "convention" not in set(inspect.signature(audit.pair_refusal).parameters)

    instance = _drawn(0)
    a_protos = tuple(row.proto for row in instance.a.table.rows)
    b_protos = tuple(row.proto for row in instance.b.table.rows)
    assert audit.pair_refusal(a_protos, b_protos) == ""
    first_texts = None
    first_record = None
    for convention in soundchange.ALL_CONVENTIONS:
        made = _made_instance(a_protos, b_protos, convention)
        texts = (made.a.text, made.b.text)
        record = (
            generator.table_record(made.a.table),
            generator.table_record(made.b.table),
            generator.row_identifiers(made.a.table),
            generator.row_identifiers(made.b.table),
            made.envelope.filler,
            {k: tuple(v) for k, v in made.envelope.neutral.items()},
        )
        # The acceptance decision, recomputed with this convention live.
        assert audit.pair_refusal(a_protos, b_protos) == ""
        if first_texts is None:
            first_texts, first_record = texts, record
        assert texts == first_texts
        assert record == first_record


def test_the_reading_is_the_registered_one_on_this_genre_too() -> None:
    """Case, whitespace, duplicates, omissions, prose and characters nobody can print.

    It fails if case or outer whitespace equivalents receive different grades, if a
    later duplicate replaces the first value, if an omitted row is credited, if prose
    gets a positional reading without exactly one line per printed row, or if
    non-ASCII, control or surrogate input raises out of the seal instead of folding.
    """
    generator = soundchange.GENERATOR
    instance = _drawn(0)
    task = instance.a
    identifiers = list(generator.row_identifiers(task.table))
    key = list(task.key)

    canonical = "\n".join("%s,%s" % kv for kv in zip(identifiers, key))
    assert generator.score(task, generator.parse_and_canonicalize(task, canonical))[0] == 1.0
    shouted = "\n".join(
        "  %s  ,  %s  " % (i.upper(), v.upper()) for i, v in zip(identifiers, key)
    )
    assert generator.score(task, generator.parse_and_canonicalize(task, shouted))[0] == 1.0

    first_wins = "\n".join(
        ["%s,%s" % (identifiers[0], key[0]), "%s,%s" % (identifiers[0], "bbbb")]
        + ["%s,%s" % kv for kv in zip(identifiers[1:], key[1:])]
    )
    read = generator.parse_and_canonicalize(task, first_wins)
    assert read.values[0] == key[0]
    assert read.duplicates == (identifiers[0],)
    assert generator.score(task, read)[0] == 1.0

    omitted = "\n".join("%s,%s" % kv for kv in zip(identifiers[:-1], key[:-1]))
    read = generator.parse_and_canonicalize(task, omitted)
    assert read.filed[-1] is False
    assert read.omissions == (identifiers[-1],)
    assert generator.score(task, read)[0] == round(23 / 24, 6)

    assert generator.score(task, generator.parse_and_canonicalize(task, "\n".join(key)))[0] == 1.0
    short = generator.parse_and_canonicalize(task, "\n".join(key[:5]))
    assert not short.is_filing and short.reason == "no_known_identifier"
    prose = generator.parse_and_canonicalize(
        task, "a paragraph about the batch\nwith no identifier in it"
    )
    assert not prose.is_filing and prose.reason == "no_known_identifier"

    awkward = "\n".join(
        "%s,%s%s" % (i, v, "é\ud800\x1b[7m\x00") for i, v in zip(identifiers, key)
    )
    read = generator.parse_and_canonicalize(task, awkward)
    assert read.is_filing
    assert generator.score(task, read)[0] == 0.0
    judged = judge_cells(
        soundchange.GENERATOR, task, read, instance.convention, instance.envelope
    )
    assert not judged.problems
    assert all(payload.isascii() for payload in judged.payloads.values())


def test_a_thirteen_byte_daughter_survives_serialize_and_read_back() -> None:
    """The correction slot is sixteen because the family's longest answer is thirteen.

    It fails if a canonical correction is truncated or changed when the cell is
    serialized and read back out of the bytes. The witness is the specification's own:
    `nbekapimotokud` under the g, both-vowels, e, replacement-first draw produces
    `mbgapimotogud`, thirteen bytes, which ledger's twelve-byte correction slot would
    truncate.
    """
    witness = "nbekapimotokud"
    convention = {
        "reflex": "out_g", "environment": "vowel_pair", "loss": "erase_e",
        "sequence": "replace_then_erase",
    }
    answer = soundchange.daughter(witness, convention)
    assert answer == "mbgapimotogud"
    assert len(answer) == 13 > ledger.CORRECTION_WIDTH
    assert soundchange.CORRECTION_WIDTH == 16

    instance = _drawn(0)
    task = instance.a
    identifiers = list(generator_of().row_identifiers(task.table))
    longest = max(task.key, key=len)
    blank = "\n".join("%s," % identifier for identifier in identifiers)
    read = generator_of().parse_and_canonicalize(task, blank)
    envelope = frozen_envelope(instance.envelope)
    ast = generator_of().render_receipt(
        task, read, task.key, feedback_for(generator_of(), task, envelope)
    )
    payload = serialize(ast, envelope)
    low, high = envelope.slot_span("correction")
    printed = [
        line[low:high].decode("ascii").strip()
        for line in row_lines(payload, ast, envelope)
    ]
    assert printed == list(task.key)
    assert longest in printed
    # The same value in ledger's narrower slot is a different, shorter string, which is
    # what makes the wider registration load bearing rather than tidy.
    assert len(longest) <= soundchange.CORRECTION_WIDTH
    truncated = longest[: ledger.CORRECTION_WIDTH - 1] + "~"
    assert len(longest) <= ledger.CORRECTION_WIDTH or truncated != longest


def generator_of():
    """The registered sound change generator, named once for the tests below."""
    return soundchange.GENERATOR


def test_every_cell_is_congruent_under_every_filing_class_and_every_draw() -> None:
    """The three cells, over seven filing shapes, 36 conventions and both siblings.

    It fails if any byte outside the two registered slots moves, if the graded and
    placebo cells lose row alignment, if their JSON encoded lengths differ, or if any
    printed correction is anything but that row's own truth. The encoded length is
    checked as well as the raw one because the serving contract carries the cells as
    JSON strings, and two cells of one size can encode to two sizes if one of them
    holds a character the encoder escapes.
    """
    generator = soundchange.GENERATOR
    instance = _drawn(0)
    envelope = frozen_envelope(instance.envelope)
    judged_count = 0
    for side in ("a", "b"):
        task = instance.side(side)
        for shape in checks.FILING_CLASSES:
            raw = checks.filing_of(generator, instance, side, shape)
            reference_mask = None
            reference_placebo = None
            for convention in soundchange.ALL_CONVENTIONS:
                truth = tuple(generator.key_for(task.table, convention))
                retasked = Task(
                    label=task.label, task_id=task.task_id, surface=task.surface,
                    table=task.table, text=task.text, key=truth, mask=task.mask,
                )
                canonical = generator.parse_and_canonicalize(retasked, raw)
                judged = judge_cells(
                    generator, retasked, canonical, convention, envelope
                )
                assert not judged.problems, (side, shape, judged.problems)
                judged_count += 1
                graded, placebo = judged.payloads["graded"], judged.payloads["placebo"]
                ranges = slot_ranges(judged.asts["graded"], envelope)
                masked = mask_slots(graded, ranges)
                if reference_mask is None:
                    reference_mask, reference_placebo = masked, placebo
                assert masked == reference_mask
                assert mask_slots(placebo, ranges) == masked
                assert placebo == reference_placebo
                assert len(json.dumps(graded.decode("ascii"))) == len(
                    json.dumps(placebo.decode("ascii"))
                )
                alignment = tuple(r.identifier for r in judged.asts["graded"].rows)
                assert alignment == generator.row_identifiers(task.table)
                for row, outcome in zip(judged.asts["graded"].rows, judged.outcomes):
                    printed = {slot.name: slot.value for slot in row.slots}
                    assert printed["verdict"] == ("PASS" if outcome.matched else "FAIL")
                    assert printed["correction"] == (
                        "" if outcome.matched else outcome.correct
                    )
    assert judged_count == 2 * len(checks.FILING_CLASSES) * 36


class _BrokenOracle(soundchange.SoundChangeGenerator):
    """The registered generator with one thing wrong with the cell it renders."""

    def __init__(self, breakage: str) -> None:
        self.breakage = breakage

    def render_oracle(self, task_id, convention, row_count: int = 0):
        ast = super().render_oracle(task_id, convention, row_count)
        if self.breakage == "wrong_scope":
            body = ("CASCADE FOR THE FIELD TRANSCRIPTION BATCH",) + ast.body[1:]
            return ReceiptAST(kind=ast.kind, task_id=ast.task_id,
                              row_count=ast.row_count, body=body)
        if self.breakage == "wrong_wrapper":
            return ReceiptAST(kind=ast.kind, task_id="0" * 16,
                              row_count=ast.row_count, body=ast.body)
        if self.breakage == "wrong_row_count":
            return ReceiptAST(kind=ast.kind, task_id=ast.task_id,
                              row_count=ast.row_count + 1, body=ast.body)
        if self.breakage == "coaching":
            return ReceiptAST(
                kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
                body=ast.body + ("  Work the k pass first and you will find it easier.",),
            )
        if self.breakage == "missing_option":
            body = tuple(line for line in ast.body if "deletion pass, remove" not in line)
            return ReceiptAST(kind=ast.kind, task_id=ast.task_id,
                              row_count=ast.row_count, body=body)
        if self.breakage == "two_options":
            return ReceiptAST(
                kind=ast.kind, task_id=ast.task_id, row_count=ast.row_count,
                body=ast.body + ("  5. And also remove the vowel a between consonants.",),
            )
        raise AssertionError(self.breakage)


def test_the_oracle_is_the_registered_cell_and_states_exactly_the_drawn_rule() -> None:
    """The oracle arm is the denominator of the room, so what it says is established.

    It fails if a phrase contained in another phrase is accepted by the template, if a
    body stating two options on an axis or none survives the parse, or if a wrong scope
    line, a wrong wrapper, a wrong row count or an extra coaching line survives the
    comparison with the cell the registered template renders.
    """
    generator = soundchange.GENERATOR
    instance = _drawn(0)
    assert checks.check_oracle(generator, instance).passed
    for convention in soundchange.ALL_CONVENTIONS:
        ast = generator.render_oracle(instance.a.task_id, convention, 24)
        assert generator.parse_oracle(ast) == convention
        assert ast.row_count == 24 and not ast.rows
        rendered = "\n".join(ast.body)
        assert len(rendered.encode("ascii")) < soundchange.ORACLE_BODY_ALLOWANCE
        for axis in soundchange.AXES:
            phrases = soundchange.ORACLE_TEMPLATE.phrases[axis.name]
            present = [o for o, p in phrases.items() if _flat(p) in _flat(rendered)]
            assert present == [convention[axis.name]], (axis.name, present)
    assert generator.parse_oracle(
        generator.render_oracle("x" * 16, soundchange.ALL_CONVENTIONS[0], 24)
    ) == {
        "reflex": "out_g", "environment": "vowel_pair",
        "loss": "erase_e", "sequence": "replace_then_erase",
    }

    with pytest.raises(ValueError):
        OracleTemplate(
            head=("A RULE", ""),
            sentences={"reflex": "use {}."},
            phrases={"reflex": {"out_g": "the phone g", "out_x": "the phone g and x"}},
        )

    for breakage in (
        "wrong_scope", "wrong_wrapper", "wrong_row_count", "coaching",
        "missing_option", "two_options",
    ):
        broken = _BrokenOracle(breakage)
        result = checks.check_oracle(broken, instance)
        assert not result.passed, breakage
        judged = judge_cells(
            broken, instance.a,
            broken.parse_and_canonicalize(instance.a, ""),
            instance.convention, instance.envelope,
        )
        assert judged.problems, breakage


def _flat(text: str) -> str:
    return " ".join(text.split())


# ----- the two checks the profile adds beyond the eleven ---------------------


def _indicator_batch(label: str, seed: int) -> tuple[str, ...]:
    """Forms whose daughters state the reflex and the lost vowel and nothing else.

    The single k sits between o and u, which no draw deletes and which are vowels on
    both sides, so the replacement fires under either environment and under either
    order, and the two remaining axes move no row at all. A reader handed the reflex
    and the lost vowel then knows every answer, which is what the augmented floor is
    there to notice.
    """
    picker = random.Random(seed)
    inert = "bdmpt" if label == "A" else "bdmpt"
    finals = soundchange.FINAL_A if label == "A" else soundchange.FINAL_B
    out: list[str] = []
    while len(out) < 24:
        vowels = ["o", "u"] + picker.sample(["a", "e", "i"], 3)
        cells = [picker.choice(inert), "k"] + [picker.choice(inert) for _ in range(3)]
        word = "".join(c + v for c, v in zip(cells, vowels)) + picker.choice(finals)
        if word not in out:
            out.append(word)
    return tuple(out)


def test_giving_the_readable_axes_away_has_to_leave_something_to_infer() -> None:
    """The stronger lookup, and a fixture that clears H while leaving nothing.

    It fails if a batch whose daughters are direct indicators of the reflex and the
    lost vowel clears `phone_lookup` even though nothing is left above that lookup: its
    ordinary headroom is ample, because the whole corrected key still resolves what the
    receipt's own labels do not, and every point of what the receipt adds beyond the two
    readable phones is zero.
    """
    generator = soundchange.GENERATOR
    indicators = _made_instance(_indicator_batch("A", 11), _indicator_batch("B", 22))
    a_codes = audit.answer_codes(audit.support_keys(_forms(indicators, "a")))
    b_codes = audit.answer_codes(audit.support_keys(_forms(indicators, "b")))
    measured = audit.room(a_codes, b_codes)
    assert measured.ceiling == 1.0
    assert measured.ceiling - measured.worst_floor > audit.MIN_LOOKUP_ROOM
    assert measured.worst_augmented == 1.0
    refused = audit.check_phone_lookup(generator, indicators)
    assert not refused.passed
    assert "puts the floor at" in refused.detail or "left above" in refused.detail

    admitted = audit.check_phone_lookup(generator, _drawn(0))
    assert admitted.passed, admitted.detail


def _forms(instance: Instance, side: str) -> tuple[str, ...]:
    return tuple(row.proto for row in instance.side(side).table.rows)


def test_a_sibling_made_by_renaming_inert_consonants_is_refused() -> None:
    """The disjoint final consonants are not the obstacle, and are not evidence.

    It fails if a B built from A by renaming only the consonants the rule never looks at
    clears `analogy`. Those forms keep every skeleton, so A's own edits transfer to them
    position for position and a reader who has A's corrected key has B's; the registered
    character copy family scores zero on exactly that pair, which is why that family is
    not by itself evidence of anything.
    """
    generator = soundchange.GENERATOR
    original = _forms(_drawn(0), "a")
    renamed = tuple(
        "".join({"d": "m", "m": "d", "p": "t", "t": "p"}.get(c, c) for c in form)
        for form in original
    )
    assert {audit.skeleton(f) for f in renamed} == {audit.skeleton(f) for f in original}
    disguised = _made_instance(original, renamed)
    refused = audit.check_analogy(generator, disguised)
    assert not refused.passed
    assert "skeleton transfer" in refused.detail

    a_keys = audit.support_keys(original)
    b_keys = audit.support_keys(renamed)
    copied = max(audit.character_copy_maximum(a_keys[n], b_keys[n]) for n in range(36))
    assert copied <= 0.05 < audit.MAX_TRANSFER
    assert audit.check_analogy(generator, _drawn(0)).passed


# ----- the bank, its provenance, and what a bounded search that runs out is ---


def test_the_key_is_recorded_before_the_bank_is_built_and_kept_when_it_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed materialization used to leave nothing behind, and now leaves its key.

    It fails if the key's commitment is not written before construction begins, if a
    key is lost when the bank does not fill, or if a second attempt under the same
    evidence directory rolls another key without being told to by name. The whole gate
    universe is a function of the master key and no bank record counts rolls, so a
    reroll for a friendlier bank is the one thing an operator can do that no hash sees.
    """
    monkeypatch.setenv(BANK_DIR_VAR, str(tmp_path))
    record = provenance_path("soundchange")

    # A construction that cannot finish: the attempt is refused by name, the key is kept,
    # and the command exits nonzero with a sentence rather than a traceback.
    soundchange.build_pair.cache_clear()
    monkeypatch.setattr(soundchange, "MAX_ATTEMPTS", 0)
    assert _cli(["receipts", "materialize", "soundchange", "--size", "1"]) == 1
    said = capsys.readouterr().out
    assert "could not be constructed" in said and "ordinal 0" in said
    attempts = bank_mod.read_provenance(record)["attempts"]
    assert len(attempts) == 1
    failed = attempts[0]
    assert failed["outcome"] == bank_mod.FAILED
    assert failed["note"] == "first attempt"
    kept = bytes.fromhex(failed["master"])
    assert len(kept) == 32
    assert failed["commitment"] == bank_mod.key_commitment(kept)

    # A second attempt under the same record is refused until the reroll is named.
    monkeypatch.undo()
    monkeypatch.setenv(BANK_DIR_VAR, str(tmp_path))
    assert _cli(["receipts", "materialize", "soundchange", "--size", "1"]) == 1
    assert "--reroll" in capsys.readouterr().out
    assert len(bank_mod.read_provenance(record)["attempts"]) == 1

    assert _cli([
        "receipts", "materialize", "soundchange", "--size", "1",
        "--reroll", "the first attempt could not construct an instance",
    ]) == 0
    filled = capsys.readouterr().out
    assert "key commitment" in filled
    attempts = bank_mod.read_provenance(record)["attempts"]
    assert len(attempts) == 2
    assert attempts[0] == failed
    assert attempts[1]["outcome"] == bank_mod.FILLED
    assert attempts[1]["note"] == "the first attempt could not construct an instance"
    second = bytes.fromhex(attempts[1]["master"])
    assert second != kept
    assert bank_mod.load_bank(bank_path("soundchange")).master == second


def test_two_attempts_at_once_keep_both_keys_and_neither_is_an_unnamed_reroll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The refusal is a read of the record and a write of it, with an attempt between them.

    It fails if a second materialization can start while the first one is constructing,
    if two attempts under one evidence directory are recorded as one, or if either key
    is lost when both constructions run out. Both keys drew a gate universe by the time
    either was written down, so a record holding one of them says a reroll that happened
    did not happen, which is the whole of what the record is kept for.
    """
    monkeypatch.setenv(BANK_DIR_VAR, str(tmp_path))
    record = provenance_path("soundchange")
    constructing = threading.Event()
    give_up = threading.Event()

    def held(*_: object, **__: object) -> None:
        """A construction that is under way until this test lets it run out."""
        constructing.set()
        assert give_up.wait(30)
        raise ConstructionExhausted("ordinal 0 could not be constructed, holding 0 of 1")

    monkeypatch.setattr(bank_mod, "materialized", held)
    monkeypatch.setattr(bank_mod, "ATTEMPT_WAIT_SECONDS", 0.2)
    codes: list[int] = []
    first = threading.Thread(
        target=lambda: codes.append(
            _cli(["receipts", "materialize", "soundchange", "--size", "1"])
        )
    )
    first.start()
    try:
        assert constructing.wait(30)
        # The first attempt has its key written down and is building. A second command
        # arriving now waits for the claim and then says what it is waiting for, rather
        # than reading the same empty record and rolling a second key beside it.
        assert _cli([
            "receipts", "materialize", "soundchange", "--size", "1",
            "--reroll", "a second operator at the same moment",
        ]) == 1
        assert "one at a time" in capsys.readouterr().out
        assert len(bank_mod.read_provenance(record)["attempts"]) == 1
    finally:
        give_up.set()
        first.join(30)
    assert codes == [1]

    # And sequentially, which is the case the claim leaves: two exhausted attempts, two
    # keys, each with the commitment it was recorded under.
    assert _cli([
        "receipts", "materialize", "soundchange", "--size", "1",
        "--reroll", "the first attempt could not construct an instance",
    ]) == 1
    attempts = bank_mod.read_provenance(record)["attempts"]
    assert [entry["outcome"] for entry in attempts] == [bank_mod.FAILED, bank_mod.FAILED]
    keys = [bytes.fromhex(entry["master"]) for entry in attempts]
    assert len(set(keys)) == 2
    for entry, key in zip(attempts, keys):
        assert entry["commitment"] == bank_mod.key_commitment(key)


def test_a_construction_that_runs_out_is_a_named_whole_bank_failure() -> None:
    """Exhaustion is not a skipped ordinal and not an unnamed exception.

    It fails if a bounded construction that cannot produce a pair is reported as an
    ordinary rejected instance, if it reaches a caller as an exception with no name of
    its own, or if bundle verification lets it through as something other than a bundle
    that does not verify. Treating it as a rejection would move the population: the bank
    would fill from later ordinals and the passing fraction it printed would be measured
    against a rule it did not use.
    """
    generator = soundchange.GENERATOR
    master = bytes(range(1, 33))
    with pytest.raises(ConstructionExhausted) as exhausted:
        with _attempts(0):
            soundchange.build_pair(master, 7)
    assert "ordinal 7" in str(exhausted.value)

    bank = bank_mod.Bank(
        generator=generator.name, genre=generator.genre,
        renderer=bank_mod.RENDERER_CONFIGURATION, master=master, size=1,
    )
    with pytest.raises(ConstructionExhausted) as named:
        with _attempts(0):
            bank_mod.population(bank, generator)
    assert "ordinal 0" in str(named.value)
    assert "holding 0 of 1" in str(named.value)
    assert not isinstance(named.value, ValueError)


def _cli(argv: list[str]) -> int:
    """The command line as an exit status, which is what a caller of it reads."""
    try:
        main(argv)
    except SystemExit as exit_code:
        return int(exit_code.code or 0)
    return 0


class _attempts:
    """Hold the construction budget at a chosen value, and clear the memoized pairs."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.held = soundchange.MAX_ATTEMPTS

    def __enter__(self) -> None:
        soundchange.build_pair.cache_clear()
        soundchange.MAX_ATTEMPTS = self.budget

    def __exit__(self, *_: object) -> None:
        soundchange.MAX_ATTEMPTS = self.held
        soundchange.build_pair.cache_clear()


# ----- a frozen bank, its review pack, its bundle and what it serves ---------


def _screen_artifact(pairs: int = 40) -> dict:
    """A recorded room screen for this family, structurally valid and not measured here.

    The numbers stand in for a pilot nobody ran in a test. What is being exercised is
    the path that recomputes everything mechanical about a bundle, not the screen.
    """
    return {
        "family": soundchange.GENERATOR.name,
        "model": "a scripted policy",
        "task_seeds": [str(i) for i in range(pairs)],
        "pairs": [
            {"instance": f"task-{i:02d}", "filing": f"filing-{i:02d}",
             "placebo": 0.4, "graded": 0.6, "oracle": 0.95}
            for i in range(pairs)
        ],
        "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36,
        "floor": 0.0, "floor_rule": "drop",
        "candidates_screened": 1, "selection_note": "",
    }


#: The key this module's bank is built under. FIXED, because a pack carries all seven
#: worksheet cases and a bank of two instances need not exhibit all seven: under a fresh
#: key every run, the pack these tests export was a different pack and one key in a few
#: hundred held no form whose daughter carries a cluster the deletion created. The bank
#: under this key does, so what is being exercised here is the exporter rather than the
#: draw, and the bank that exhibits six of the seven has a test of its own.
PACK_MASTER = hashlib.sha256(b"soundchange-review-pack").digest()


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory):
    """One small bank of this genre, its exported pack, and a bundle that verifies."""
    room = tmp_path_factory.mktemp("soundchange")
    bank, held = bank_mod.materialized(soundchange.GENERATOR, PACK_MASTER, 2)
    outcomes = room / "screen.json"
    outcomes.write_text(json.dumps(_screen_artifact()), encoding="utf-8")
    pack_root = room / "pack"
    pack = soundchange_review.export(bank, held, pack_root)
    # The exporter leaves the attestation unset, so the bundle refuses the pack until a
    # person has put their name to it. That refusal is the point of leaving it unset.
    with pytest.raises(ValueError):
        bundle_mod.build(room / "bundles", soundchange.GENERATOR, bank, outcomes, pack)
    soundchange_review.attested(pack_root, "a named reader")
    built = bundle_mod.build(
        room / "bundles", soundchange.GENERATOR, bank, outcomes, pack
    )
    assert bundle_mod.verify(built, soundchange.GENERATOR).problems == ()
    return bank, held, built


def test_the_review_pack_covers_the_family_and_names_no_reviewer(
    frozen, tmp_path: Path
) -> None:
    """What the exported pack has to contain, and the one thing it must not.

    It fails if the pack misses a surface template, an option of any axis, a registered
    filing class, the row count the bank holds or a counterfactual render, if the 36
    oracle cells are not all there, if a counterfactual is missing for any axis on any
    surface, if the worksheets carry fewer than the seven cases or carry them in another
    order, or if the exporter names a reviewer. It also fails if the trace worksheets
    are labelled as renders: they are explanatory material for the roster release, they
    are not served tasks, and a bundle that carried them as evidence of what was served
    would be saying something nobody checked.
    """
    bank, held, _ = frozen
    room = tmp_path / "again"
    pack = soundchange_review.export(bank, held, room)
    manifest = json.loads(pack.read_text(encoding="utf-8"))
    assert set(manifest) == set(review.REQUIRED_FIELDS)
    assert manifest["reviewer"] is None
    assert manifest["family"] == "soundchange"
    assert manifest["bank"] == bank_mod.bank_identity(bank)
    assert manifest["seeds"] == list(held.ordinals)

    coverage = review.required_coverage(
        soundchange.GENERATOR, checks.FILING_CLASSES, [soundchange.ROWS]
    )
    seen = [(e["category"], e["key"]) for e in manifest["renders"]]
    assert coverage.missing(seen) == []
    oracles = {e["path"] for e in manifest["renders"] if e["path"].startswith("renders/oracle-")}
    assert len(oracles) == 36
    for surface in soundchange.GENERATOR.surface_templates():
        for axis in soundchange.AXES:
            assert any(
                key == f"{surface} {axis.name}=" + option
                for _, key in seen
                for option in axis.options
            ), (surface, axis.name)
    for entry in manifest["renders"]:
        assert not entry["path"].startswith(soundchange_review.WORKSHEETS)
        assert (room / entry["path"]).is_file()

    index = json.loads((room / soundchange_review.WORKSHEET_INDEX).read_text("utf-8"))
    assert index["cases"] == list(soundchange_review.WORKSHEET_CASES)
    assert index["files"]
    for name, hashed in index["files"].items():
        assert digest((room / name).read_bytes()) == hashed
    sheet = json.loads(
        (room / soundchange_review.WORKSHEETS / "cases.json").read_text("utf-8")
    )
    assert len(sheet) == len(soundchange_review.WORKSHEET_CASES)
    assert [case["case"] for case in sheet] == list(soundchange_review.WORKSHEET_CASES)
    for case in sheet:
        assert set(case) >= {
            "case", "proto", "after_nasal", "daughter", "skeleton", "oracle",
        }

    # The exported pack is the same pack twice, so two readers read one document.
    twice = tmp_path / "twice"
    soundchange_review.export(bank, held, twice)
    assert (twice / soundchange_review.PACK).read_bytes() == pack.read_bytes()


def test_a_bank_missing_a_worksheet_case_is_refused_and_no_pack_is_written(
    tmp_path: Path,
) -> None:
    """A pack is all seven cases, and a small bank does not always hold all seven.

    It fails if an export drops a worksheet case the bank does not exhibit, which is how
    a pack reached a reader with no example of the cluster the deletion creates while
    the manifest said nothing was missing, and it fails if the refusal does not name the
    case or leaves a half written pack behind for somebody to read as one. The key here
    is a bank of two whose rows exhibit six of the seven.
    """
    master = hashlib.sha256(b"soundchange-review-missing-case:15").digest()
    bank, held = bank_mod.materialized(soundchange.GENERATOR, master, 2)
    assert held.ordinals == (0, 1)
    room = tmp_path / "pack"
    with pytest.raises(ValueError) as refused:
        soundchange_review.export(bank, held, room)
    said = str(refused.value)
    assert soundchange_review.WORKSHEET_CASES[6] in said
    assert "exhibits 6" in said
    assert not room.exists()


def test_a_frozen_bank_rebuilds_and_a_failed_new_check_is_not_dealable(frozen) -> None:
    """Replay, rebuild, and what happens when one of the added checks says no.

    It fails if a small valid frozen bank cannot be rebuilt byte-identically, if an
    instance that fails one of the three checks this profile adds is still admitted, or
    if replaying a committed fork changes its cells or answers one filing with another
    filing's feedback.
    """
    bank, held, built = frozen
    generator = soundchange.GENERATOR
    again = bank_mod.population(bank, generator)
    assert again.ordinals == held.ordinals
    for first, second in zip(held.instances, again.instances):
        assert bank_mod.instance_digest(first, generator) == bank_mod.instance_digest(
            second, generator
        )

    instance = held.instances[0]
    assert admission.report(generator, instance, bank.master, admission.Thresholds()).admitted
    refused = checks.CheckResult("analogy", False, "a fixture refusal")
    with _stub(audit, "check_analogy", lambda *_: refused):
        report = admission.report(
            generator, instance, bank.master, admission.Thresholds()
        )
        assert "analogy" in report.failed_checks
        assert not report.admitted
        with pytest.raises(ValueError):
            bank_mod.population(bank, generator)

    room = Path(built.root).parent / "forks"
    source = built.digest
    task = instance.a
    identifiers = list(generator.row_identifiers(task.table))
    perfect = "\n".join("%s,%s" % kv for kv in zip(identifiers, task.key))
    wrong = "\n".join("%s,%s" % (i, "bbb") for i in identifiers)
    keyed = bank_mod.filing_digest(perfect)
    assert bank_mod.load_fork(room, task.task_id, keyed, source) is None
    first = bank_mod.fork_for(generator, instance, "a", perfect, room, source)
    assert bank_mod.load_fork(room, task.task_id, keyed, source) is not None
    replayed = bank_mod.fork_for(generator, instance, "a", perfect, room, source)
    assert replayed.replayed
    assert (replayed.graded, replayed.placebo, replayed.oracle) == (
        first.graded, first.placebo, first.oracle
    )
    assert replayed.component_score == first.component_score == 1.0
    other = bank_mod.fork_for(generator, instance, "a", wrong, room, source)
    assert other.graded != first.graded
    assert other.component_score == 0.0
    assert other.filing_digest != first.filing_digest


class _stub:
    """Put one attribute back where it was, whatever the block did."""

    def __init__(self, module, name: str, value) -> None:
        self.module, self.name, self.value = module, name, value
        self.held = getattr(module, name)

    def __enter__(self) -> None:
        setattr(self.module, self.name, self.value)

    def __exit__(self, *_: object) -> None:
        setattr(self.module, self.name, self.held)


async def test_a_filing_an_independent_validator_believes_seals_at_one(
    frozen, tmp_path: Path
) -> None:
    """What the served environment does with a correct filing, an empty one and a grade.

    It fails if a filing computed by the second implementation of the cascade does not
    seal at 1, if an empty filing is credited, or if either channel of the terminal
    result carries a grade: the content the agent reads, or the metadata sidecar its own
    process reads. What one graded receipt is worth is the quantity this environment
    exists to measure, so a terminal that handed the grade back would put a receipt in
    every arm including the one that is meant to carry none.
    """
    _, _, built = frozen
    private = tmp_path / "served"
    private.mkdir()
    opened = private / built.root.name
    shutil.copytree(built.root, opened)
    config = {"genre": "soundchange", "bundle": str(opened), "side": "a"}
    env = ReceiptsV1Env(**config)
    ordinal = env._ordinals[0]
    instance = env.instance(ordinal)
    task = instance.a
    protos = tuple(row.proto for row in task.table.rows)
    independent = audit.audit_key(protos, instance.convention)
    assert independent == tuple(task.key)
    filing = "\n".join(
        "%s,%s" % kv
        for kv in zip(soundchange.GENERATOR.row_identifiers(task.table), independent)
    )

    episode = await ServedEpisode.start(
        "receipts_v1", task=0, env_config=config, trace_path=tmp_path / "run.jsonl"
    )
    try:
        spec = episode.describe()
        assert "BATCH (" in spec.instructions
        assert option_mentions(soundchange.AXES, json.dumps(spec.model_dump())) == []
        result = await episode.call("submit_filing", {"filing": filing})
        assert result.terminated
        content = json.loads(result.content)
        assert set(content) == {"filed", "rows", "finalize_error"}
        assert "1.0" not in json.dumps(content)
        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["component_score"] == 1.0
        assert feedback["rows_omitted"] == 0.0
    finally:
        await episode.close()

    # A filing that names every row and answers none of them. No correct daughter is
    # ever the empty string, and a row that was filed empty is still a filed row, so
    # this is a scorable zero rather than an absent filing.
    blank = "\n".join(
        "%s," % identifier
        for identifier in soundchange.GENERATOR.row_identifiers(task.table)
    )
    empty = await ServedEpisode.start(
        "receipts_v1", task=0, env_config=config, trace_path=tmp_path / "empty.jsonl"
    )
    try:
        await empty.call("submit_filing", {"filing": blank})
        feedback = {item["name"]: item["value"] for item in empty.terminal_feedback}
        assert feedback["component_score"] == 0.0
        assert feedback["rows_filed"] == float(soundchange.ROWS)
        assert "no_filing" not in feedback
    finally:
        await empty.close()

    # And prose, which names no row at all, is reason coded rather than scored low.
    unread = await ServedEpisode.start(
        "receipts_v1", task=0, env_config=config, trace_path=tmp_path / "prose.jsonl"
    )
    try:
        await unread.call("submit_filing", {"filing": "I could not work out the cascade"})
        feedback = {item["name"]: item["value"] for item in unread.terminal_feedback}
        assert feedback["component_score"] == 0.0
        assert feedback["no_filing"] == "no_known_identifier"
    finally:
        await unread.close()
