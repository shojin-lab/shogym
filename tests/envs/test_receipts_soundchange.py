"""The sound change genre, the shared filing helpers, and the copy profile contract.

Each test states the failure it is here to catch, because a test whose failure
condition is "something changed" is a test nobody can act on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shogym.envs.receipts import checks
from shogym.envs.receipts.generators import ledger
from shogym.envs.receipts.protocol import draw
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
