"""Scoring the filing log: the drawn key, the per-item verdicts, and the fractions read off them.

Out of process, after the seal, over the world's end state, with no model call anywhere and the
key frozen before the agent acts. Everything here is a pure function of the end state and the
key, so a verdict is reproducible from the record rather than from a rerun.

**The denominator is the whole item set.** A request the agent never filed a line for is scored
incorrect, not dropped, because dropping it would make the score an average over whatever the
agent chose to attempt and a contrast between two arms could then be carried entirely by how many
lines each arm bothered to write. The filing rate is reported separately for exactly that reason:
a difference that lives in ``exercise_fraction`` is an agent learning a chore, and it must be
visible rather than folded into the headline.

**The parser is lenient and says so.** A line counts if it names one request and one band, in any
order, with any separator and in any case. Two bands on one line is a wrong answer rather than a
parse failure, because the agent acted; a line naming no request at all is a shape error, which
is reported as a shape error and never graded as though the agent had answered.
"""

from __future__ import annotations

import re
import zlib
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from shogym.envs.appworld.ledger import BANDS, CONVENTIONS, Backlog, Convention
from shogym.envs.appworld.world import SLOTS, Filing

# ----- what the receipt prints in place of a value -----
#
# A closed ASCII vocabulary, so no world data reaches a payload. A cell the agent left unset, one
# it filled with something outside the option set, and one the renderer cannot resolve are three
# different states and each has its own token; none of them is a value the world supplied.

NOT_FILED = "not filed"
NOT_SET = "not set"
OTHER = "other"
AMBIGUOUS = "ambiguous"
NOT_DETERMINED = "not determined"
OK = "ok"
NOT_OK = "not ok"

#: Every token that may appear in a rendered ``observed`` column, which is what makes the width
#: of that column a constant and the payload's length independent of the world.
OBSERVED_VOCABULARY: Tuple[str, ...] = (
    *BANDS,
    *(option for slot in SLOTS for option in slot.options),
    NOT_FILED,
    NOT_SET,
    OTHER,
    AMBIGUOUS,
    NOT_DETERMINED,
    OK,
    NOT_OK,
)

LEDGER = "ledger"
PINNED = "pinned"
ASSERTION = "assertion"


class Key(NamedTuple):
    """What the world was scored against: the convention over the ledger, and the four slots."""

    convention: Convention
    slots: Tuple[str, ...]


def draw_key(task_id: str, pulse: int) -> Key:
    """The key for one task under one draw.

    Uniform over the 64 conventions and over each slot's own options, and a deterministic
    function of the task and the pulse alone. It never sees the agent, the feedback regime or the
    run, so the same task drawn twice is the same key and two arms of one task are graded against
    one rule."""
    draw = stream(f"{pulse}|{task_id}")
    convention = CONVENTIONS[next(draw) % len(CONVENTIONS)]
    slots = tuple(slot.options[next(draw) % len(slot.options)] for slot in SLOTS)
    return Key(convention=convention, slots=slots)


def stream(seed: str):
    """An endless stream of non-negative integers from a string seed.

    A checksum chain rather than ``random``: it is reproducible across processes and Python
    versions without touching the global generator that the world itself draws from."""
    state = zlib.crc32(seed.encode())
    while True:
        state = zlib.crc32(b"|", state * 2654435761 & 0xFFFFFFFF)
        yield state


class Item(NamedTuple):
    """One scored item: what it is, what the agent did, and whether that was right.

    A receipt prints the first three columns and the verdict, and nothing else: no expected
    value, no rule statement, and nothing naming which choice the item turns on. ``dated`` is not
    printed. It says whether this is one of the requests that record dates, which the sampler
    behind a drawn receipt needs and a reader of a real one can already see in the world."""

    kind: str
    check_id: str
    observed: str
    passed: bool
    dated: bool = False


class Verdicts(NamedTuple):
    """Every scored item, in canonical order, plus the fractions read off them."""

    items: Tuple[Item, ...]
    ledger_fraction: float
    pinned_fraction: float
    exercise_fraction: float
    parse_fraction: float
    distinct_bands: int
    filing_rows: int

    @property
    def assertions(self) -> int:
        return sum(1 for item in self.items if item.kind == ASSERTION)


def score(
    *,
    backlog: Backlog,
    key: Key,
    filing: Filing,
    assertions: Sequence[Tuple[str, Optional[bool]]],
) -> Verdicts:
    """Score one sealed episode.

    ``assertions`` is the base task's own checks in their canonical order, each a check id and
    whether it passed, with ``None`` for a check whose outcome the evaluator did not report.
    They are carried so the receipt's row count is the task's own check count, and they are never
    mixed into the ledger's fraction."""
    reported, parsed, lines = _read_lines(filing.lines, backlog)
    expected = backlog.key(key.convention)

    items: List[Item] = []
    for check_id, passed in assertions:
        observed = NOT_DETERMINED if passed is None else (OK if passed else NOT_OK)
        items.append(Item(ASSERTION, check_id, observed, bool(passed)))

    correct_ledger = 0
    exercised = 0
    for position, request in enumerate(backlog.requests):
        observed = reported.get(request.reference, NOT_FILED)
        if observed != NOT_FILED:
            exercised += 1
        passed = observed == expected[position]
        correct_ledger += passed
        items.append(Item(LEDGER, request.reference, observed, passed, request.dates is not None))

    correct_pinned = 0
    for slot, drawn, filled in zip(SLOTS, key.slots, _slot_values(filing)):
        if filled is None or filled == slot.default:
            observed = NOT_SET
        else:
            observed = filled if filled in slot.options else OTHER
        passed = filled is not None and filled == drawn
        correct_pinned += passed
        items.append(Item(PINNED, slot.check_id, observed, passed))

    bands = {b for b in reported.values() if b in BANDS}
    total = len(backlog.requests)
    return Verdicts(
        items=tuple(items),
        ledger_fraction=correct_ledger / total,
        pinned_fraction=correct_pinned / len(SLOTS),
        exercise_fraction=exercised / total,
        parse_fraction=(parsed / lines) if lines else 0.0,
        distinct_bands=len(bands),
        filing_rows=filing.rows,
    )


def _slot_values(filing: Filing) -> Tuple[Optional[str], ...]:
    """The four stored slots off the filing, in the order :data:`SLOTS` lists them."""
    return (filing.section, filing.color, filing.unit, filing.priority)


def _read_lines(
    lines: Sequence[str], backlog: Backlog
) -> Tuple[Dict[str, str], int, int]:
    """What each request was given, how many lines parsed, and how many lines there were.

    The first line that names a request wins. A request named twice is one answer plus one
    duplicate rather than two answers, so a second line cannot revise the first: revising after
    the fact is what the seal exists to prevent, and a parser that allowed it here would let it
    back in through the description."""
    references = [r.reference for r in backlog.requests]
    reported: Dict[str, str] = {}
    parsed = 0
    counted = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        counted += 1
        named = [ref for ref in references if _mentions(line, ref)]
        if len(named) != 1:
            continue
        parsed += 1
        reference = named[0]
        if reference in reported:
            continue
        bands = [band for band in BANDS if _mentions(line, band)]
        if len(bands) == 1:
            reported[reference] = bands[0]
        elif bands:
            reported[reference] = AMBIGUOUS
        else:
            reported[reference] = OTHER
    return reported, parsed, counted


def _mentions(line: str, token: str) -> bool:
    """Whether ``line`` names ``token``, ignoring case and whatever separates it from the rest."""
    return re.search(rf"(?<![0-9A-Za-z]){re.escape(token)}(?![0-9A-Za-z])", line, re.I) is not None


__all__ = [
    "AMBIGUOUS",
    "ASSERTION",
    "LEDGER",
    "NOT_DETERMINED",
    "NOT_FILED",
    "NOT_OK",
    "NOT_SET",
    "OBSERVED_VOCABULARY",
    "OK",
    "OTHER",
    "PINNED",
    "Item",
    "Key",
    "Verdicts",
    "draw_key",
    "score",
    "stream",
]
