"""The sound change genre, the shared filing helpers, and the copy profile contract.

Each test states the failure it is here to catch, because a test whose failure
condition is "something changed" is a test nobody can act on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shogym.envs.receipts import checks, copy_profiles
from shogym.envs.receipts.generators import ledger
from shogym.envs.receipts.protocol import draw
from shogym.envs.receipts.registry import FIXTURES as FIXTURES_NAMES
from shogym.envs.receipts.registry import GENRES, load_generator
from shogym.envs.receipts.render import judge_cells

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


def test_a_generator_without_a_declared_profile_is_refused_at_registration() -> None:
    """A copy family is declared, and registration is where the declaration is read.

    It fails if an undeclared profile falls back to any family at all, if a profile
    name this build does not register is accepted, or if `load_generator` hands back a
    generator whose declaration was never read. Silence is the failure being refused:
    a family of words priced by maps between two published vocabularies reports the
    maximum of a transfer nobody can perform, and the bar is then read against a
    number that measured nothing.
    """
    for offered in (_Undeclared(), _Unregistered()):
        with pytest.raises(ValueError) as refusal:
            copy_profiles.profile_of(offered)
        assert "COPY_PROFILE" in str(refusal.value) or "profile" in str(refusal.value)
        with pytest.raises(ValueError):
            copy_profiles.require_profile(offered)
    for name in sorted(set(GENRES) | set(FIXTURES_NAMES)):
        assert copy_profiles.profile_of(load_generator(name)) in copy_profiles.PROFILES


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
