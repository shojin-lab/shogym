"""The three things a sealed episode can be handed back, all the same size on the wire.

- **The receipt** lists every scored item in canonical order with ``PASS`` or ``FAIL`` beside it,
  and nothing else. No expected value, no rule statement, no per-axis error count, and no naming
  of which choice an item involves. The verdicts are the only place reference information appears,
  and they are per item rather than per axis, so a reader has to work out which conventions
  account for the pattern rather than read the answer off a label.
- **The digest** is a deterministic function of the task identity and the agent's own submission.
  Both are already in the agent's own transcript, so it carries no information about the drawn
  key by construction, and no inspection of its wording is needed to establish that.
- **The drawn receipt** looks exactly like the receipt and states verdicts that were sampled
  rather than computed. Its number of passing requests comes from the roster's own distribution
  of passing counts, so it is not separable from a real receipt by that number either, and the
  sample is keyed by the task and the class alone, never by the real verdicts. It is not inert
  and it is not a control: it states false verdicts, and a reader that acts on them can finish
  worse than one handed nothing.

**All three are the same length on the wire, by construction rather than by padding at the end.**
Every column is fixed width and every cell comes from a closed ASCII vocabulary, so world data
never reaches a payload and the corpus's own non-ASCII cells cannot change the byte count. The
two words that genuinely differ, the status and the last column's heading, are laid into fields
whose widths are constants of this module. Length matching is what makes two arms one comparison:
an agent that reads a short answer in one arm and a long one in the other has been told which arm
it is in before reading a word of either.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from shogym.envs.appworld.scorer import (
    ASSERTION,
    LEDGER,
    OBSERVED_VOCABULARY,
    PINNED,
    Item,
    Verdicts,
    stream,
)
from shogym.envs.appworld.world import SLOTS

#: The frozen table of passing-request counts, enumerated over the whole served roster crossed
#: with all 64 conventions before any episode was served, and committed beside the code. A drawn
#: receipt takes its count from here, so it is not separable from a real one by its count.
PASS_COUNTS_FILE = Path(__file__).with_name("pass_counts.txt")

GRADED = "graded"
#: What the drawn arm's visible vector is keyed by, beside the task.
#:
#: A committed constant rather than the pulse. The pulse is the hidden parameter the real key is
#: drawn from, so keying the visible vector by it too would make the vector a sample of the key's
#: own cause. Nothing that produces a key may appear here. Changing this value re-rolls every
#: drawn payload, so it is frozen like the count table beside it.
DRAWN_BASIS = "shogym/appworld/drawn/v1"

DIGEST = "digest"
DRAWN = "drawn"

_STATUS: Dict[str, str] = {
    GRADED: "graded",
    DIGEST: "recorded, batch review pending",
    DRAWN: "graded",
}
_LAST_HEADING: Dict[str, str] = {GRADED: "verdict", DIGEST: "digest", DRAWN: "verdict"}

# Column widths, taken from the vocabularies that can appear under them rather than chosen. A
# width read off a vocabulary cannot be outgrown by a value, which is what keeps a wide cell from
# pushing a row out of shape and the payload out of its length.
_INDEX_WIDTH = 3
_KIND_WIDTH = max(len(kind) for kind in (ASSERTION, LEDGER, PINNED))
_ID_WIDTH = max(len("fr.label.color"), len("RQ-0000"), len("aw.000")) + 2
_OBSERVED_WIDTH = max(len(token) for token in OBSERVED_VOCABULARY)
_LAST_WIDTH = max(len(heading) for heading in _LAST_HEADING.values())
_STATUS_WIDTH = max(len(status) for status in _STATUS.values())

PASS = "PASS"
FAIL = "FAIL"


def render(
    *, task_id: str, verdicts: Verdicts, cell: str, pulse: int = 0
) -> str:
    """One payload for one sealed episode, in the class ``cell`` names.

    The row set, the row order and every column but the last are the same in all three classes,
    and the last column is four characters wide in all three."""
    if cell not in _STATUS:
        raise ValueError(f"unknown payload class {cell!r}; expected one of {sorted(_STATUS)}")
    marks = _marks(task_id=task_id, verdicts=verdicts, cell=cell, pulse=pulse)
    counts = _counts(verdicts)
    header = (
        f"SUBMISSION RECEIPT   task {task_id}\n"
        f"status: {_STATUS[cell]:<{_STATUS_WIDTH}}\n"
        f"checks: {len(verdicts.items)}   ({counts})\n"
        "\n"
        f"{'#':>{_INDEX_WIDTH}}   {'kind':<{_KIND_WIDTH}}   {'check id':<{_ID_WIDTH}}   "
        f"{'observed':<{_OBSERVED_WIDTH}}   {_LAST_HEADING[cell]:<{_LAST_WIDTH}}\n"
    )
    rows = "".join(
        f"{position:>{_INDEX_WIDTH}}   {item.kind:<{_KIND_WIDTH}}   "
        f"{item.check_id:<{_ID_WIDTH}}   {item.observed:<{_OBSERVED_WIDTH}}   "
        f"{mark:<{_LAST_WIDTH}}\n"
        for position, (item, mark) in enumerate(zip(verdicts.items, marks), start=1)
    )
    return header + rows


def _counts(verdicts: Verdicts) -> str:
    """The header's breakdown of the row count by kind.

    A task constant, fixed before the agent acts and the same for every possible submission, so
    it carries nothing about the score."""
    ledger = sum(1 for item in verdicts.items if item.kind == LEDGER)
    pinned = sum(1 for item in verdicts.items if item.kind == PINNED)
    return f"assertions {verdicts.assertions}, ledger {ledger}, pinned {pinned}"


def _marks(*, task_id: str, verdicts: Verdicts, cell: str, pulse: int) -> Tuple[str, ...]:
    """The last column's values, one per row, all four characters wide."""
    if cell == DIGEST:
        return tuple(_digest(task_id, item) for item in verdicts.items)
    if cell == DRAWN:
        return tuple(
            PASS if passed else FAIL
            for passed in _drawn_verdicts(task_id=task_id, verdicts=verdicts)
        )
    return tuple(PASS if item.passed else FAIL for item in verdicts.items)


def _digest(task_id: str, item: Item) -> str:
    """Four hex characters over the task identity, the item and what the agent put there.

    Deterministic rather than random: a random column is a channel nobody controls and nobody can
    audit, and the point of this payload is that its content is recomputable from things the
    agent already holds."""
    material = (task_id + item.check_id + item.observed).encode()
    return hashlib.sha256(material).hexdigest()[:4]


def _drawn_verdicts(*, task_id: str, verdicts: Verdicts) -> Tuple[bool, ...]:
    """A verdict per row that carries nothing about the drawn key.

    The dated requests get a number of passes drawn from the roster's own distribution
    (:func:`pass_counts`), spread over them uniformly at random. The undated request and
    the four stored slots get an independent draw at their own option count, which is the rate a
    real receipt shows on them. The base task's own checks keep their real verdicts, because an
    assertion says something about the base task and nothing about the key.

    **Not keyed by the pulse, and that is the point.** The real convention is itself a
    deterministic function of the pulse, so a visible vector keyed by it too would be an
    agent-visible sample of the hidden parameter: independence of the *verdicts* is not
    independence of the *key*. Two things sharing a hidden cause is a leak whether or not either
    reads the other. The stream is keyed by :data:`DRAWN_BASIS`, a committed constant that the key
    never touches, so the only thing the visible vector varies with is the task. The pulse is not
    a parameter of this function at all, which is the form of that claim that cannot quietly stop
    being true."""
    draw = stream(f"{DRAWN_BASIS}|{task_id}|drawn")
    dated = [position for position, item in enumerate(verdicts.items) if item.dated]
    passing = set(_choose(draw, dated, _draw_count(draw)))
    arities = {slot.check_id: len(slot.options) for slot in SLOTS}
    dated_positions = set(dated)

    out: List[bool] = []
    for position, item in enumerate(verdicts.items):
        if item.kind == ASSERTION:
            out.append(item.passed)
        elif position in dated_positions:
            out.append(position in passing)
        elif item.kind == PINNED:
            out.append(next(draw) % arities[item.check_id] == 0)
        else:
            out.append(next(draw) % 2 == 0)
    return tuple(out)


def _draw_count(draw: Iterator[int]) -> int:
    """One number of passing requests, drawn from the roster's frozen distribution."""
    counts = pass_counts()
    total = sum(counts)
    target = next(draw) % total
    running = 0
    for count, weight in enumerate(counts):
        running += weight
        if target < running:
            return count
    return len(counts) - 1


def _choose(draw: Iterator[int], population: Sequence[int], size: int) -> List[int]:
    """``size`` members of ``population``, uniformly and without replacement."""
    pool = list(population)
    picked: List[int] = []
    for _ in range(min(size, len(pool))):
        picked.append(pool.pop(next(draw) % len(pool)))
    return picked


@lru_cache(maxsize=1)
def pass_counts() -> Tuple[int, ...]:
    """The frozen distribution, one weight per possible number of passing dated requests.

    Read from the committed table rather than computed, because a distribution recomputed per
    process is a distribution that can move under a generator change without anybody noticing,
    and a drawn receipt whose count came from a different roster than the real ones is separable
    from them by exactly the thing this payload exists to destroy."""
    weights = tuple(int(line) for line in PASS_COUNTS_FILE.read_text().split() if line)
    if not weights or not sum(weights):
        raise ValueError(f"the passing-count table at {PASS_COUNTS_FILE} is empty")
    return weights


__all__ = [
    "DIGEST",
    "DRAWN_BASIS",
    "DRAWN",
    "FAIL",
    "GRADED",
    "PASS",
    "PASS_COUNTS_FILE",
    "pass_counts",
    "render",
]
