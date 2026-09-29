"""Shared machinery for row-banded families that serve a full receipt.

Three families in this package (`scorecard`, `standing` and `allocation`) have the same
outer shape: a printed table of twenty four records, one answer per record drawn from a
small ordered vocabulary the task itself publishes, a receipt that reports a verdict and
a correction on every record, and a hidden rule made of several named decisions. What
differs between them is how a record's answer is computed. Everything else lives here,
once, so the three families read a filing, score it and build their cells the same way
ledger does, and so the perfect reader of a receipt is one function rather than three.

WHAT IS HERE.

  the envelope   registered widths, slots, column titles and the committed neutral
                 tokens, built exactly as the ledger builds its own.
  the family     `BandedRowsFamily`, the half of the generator protocol that does not
                 depend on the genre: reading, scoring, the three cells, the oracle.
  the design     helpers a `build_table` uses to demand that a table separates every
                 pair of rules and that every single decision moves enough records.
  the reader     `perfect_reader`, the ideal reader of one delivered receipt: uniform
                 over the whole rule set, conditioned on what the receipt shows, filing
                 the majority answer per record with a fixed tie rule. It is exact: it
                 enumerates the rule set rather than sampling it.

None of this module reads a drawn convention except `perfect_reader`, which is the
controller-side ideal reader and is never shown to an agent.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from shogym.envs.receipts import filing as shared_filing
from shogym.envs.receipts import streams
from shogym.envs.receipts.copy_profiles import ORDERED_TOKENS
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.oracle import parse as parse_oracle_cell
from shogym.envs.receipts.oracle import render as render_oracle_cell
from shogym.envs.receipts.protocol import (
    FULL_RECEIPT,
    ROW_ADDITIVE_EQUAL_WEIGHT,
    Axis,
    Filing,
    Instance,
    PublicTask,
    ReceiptPolicy,
    RowOutcome,
    Task,
    conventions,
    support_of,
)
from shogym.envs.receipts.receipt_ast import (
    Envelope,
    ReceiptAST,
    SlotSpec,
    envelope_size_for,
)
from shogym.envs.receipts.render import (
    CORRECTION_SLOT,
    FAIL_TOKEN,
    PASS_TOKEN,
    VERDICT_SLOT,
    Feedback,
    graded_receipt,
    placebo_receipt,
)
from shogym.receipts import ROW_LABEL

# ----- the registered envelope constants -------------------------------------------
# The same widths and grammar the ledger registers. None is read off a draw.

IDENTIFIER_WIDTH = 12
OBSERVED_WIDTH = 16
VERDICT_WIDTH = 4
CORRECTION_WIDTH = 12
#: The longest answer any of these families may publish. A correction is printed in a
#: slot of this width, and a longer one would be truncated in the bytes a reader reads.
MAX_ANSWER = CORRECTION_WIDTH
#: What the receipt prints for a row filed with an empty value, and for a row not filed
#: at all. Two acts, two tokens, as in the ledger.
BLANK_TOKEN = "(empty)"
UNFILED_TOKEN = "(none)"
VERDICT_TOKENS = (PASS_TOKEN, FAIL_TOKEN)
SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec(VERDICT_SLOT, VERDICT_WIDTH, vocabulary=VERDICT_TOKENS),
    SlotSpec(
        CORRECTION_SLOT, CORRECTION_WIDTH,
        vocabulary=(BLANK_TOKEN,), allows_answers=True, allows_empty=True,
    ),
)
#: Neutral titles shared by the graded and placebo cells, as in the ledger.
COLUMN_TITLES = ("mark", "note")
#: Digits only, so no neutral token can ever spell a verdict or an answer word.
FILLER_ALPHABET = "0123456789"


def envelope_size(rows: int, body_allowance: int) -> int:
    """The registered cell size for a family of `rows` records and an oracle allowance."""
    return envelope_size_for(
        max_rows=rows,
        identifier_width=IDENTIFIER_WIDTH,
        observed_width=OBSERVED_WIDTH,
        slots=SLOTS,
        body_allowance=body_allowance,
    )


def committed_envelope(
    master: bytes, family: str, ordinal: int, size: int, rows: int
) -> Envelope:
    """The registered envelope with this instance's committed filler and neutral tokens.

    Drawn from the filler stream under the family's own name, so two families under one
    key never share a placebo token, and from nothing that has seen a convention.
    """
    neutral: dict[str, tuple[str, ...]] = {}
    for spec in SLOTS:
        neutral[spec.name] = tuple(
            streams.filler_stream(
                master, FILLER_ALPHABET, spec.width, family, ordinal, "neutral",
                spec.name, row,
            )
            for row in range(rows)
        )
    return Envelope(
        size=size,
        identifier_width=IDENTIFIER_WIDTH,
        observed_width=OBSERVED_WIDTH,
        slots=SLOTS,
        filler=streams.filler_stream(master, FILLER_ALPHABET, size, family, ordinal, "pad"),
        column_titles=COLUMN_TITLES,
        neutral=neutral,
    )


# --------------------------------------------------------------------------
# the genre-independent half of the protocol
# --------------------------------------------------------------------------


class BandedRowsFamily:
    """What every row-banded family here does the same way.

    A subclass supplies the genre: its surfaces, how a table is built, how a record's
    answer follows from a rule, the task text and the oracle's phrase table. This class
    supplies the rest, which is the ledger's reading, scoring and cell building with
    nothing changed, so a filing is read and a receipt is built by the shared rules.
    """

    name: str = ""
    genre: str = ""
    AXES: tuple[Axis, ...] = ()
    SCORING: str = ROW_ADDITIVE_EQUAL_WEIGHT
    #: Every answer is one token of a complete ordered vocabulary each task prints, and
    #: the two siblings never share a token, so the registered transfer is a dictionary
    #: between two published orders exactly as it is for the ledger.
    COPY_PROFILE: str = ORDERED_TOKENS
    #: A verdict and a correction on every record. See each family's docstring for why
    #: the full receipt is the right instrument for it.
    RECEIPT_POLICY: ReceiptPolicy = FULL_RECEIPT
    BLANK_TOKEN = BLANK_TOKEN
    UNFILED_TOKEN = UNFILED_TOKEN
    ORACLE: OracleTemplate
    ENVELOPE_SIZE: int = 0
    ROWS: int = 24

    # ----- supplied by the genre -----

    def key_for(self, table: Any, convention: Mapping[str, str]) -> tuple[str, ...]:
        raise NotImplementedError

    def row_identifiers(self, table: Any) -> tuple[str, ...]:
        raise NotImplementedError

    # ----- the envelope -----

    def build_envelope(self, master: bytes, ordinal: int) -> Envelope:
        return committed_envelope(master, self.name, ordinal, self.ENVELOPE_SIZE, self.ROWS)

    # ----- reading and scoring -----

    def normalize_answer(self, value: str) -> str:
        # The function the scorer compares with, so the two cannot drift.
        return shared_filing.fold(value)

    def row_label(self, table: Any) -> tuple[str, ...]:
        # Every printed row names the record it grades; nothing names a decision.
        return (ROW_LABEL,) * len(self.row_identifiers(table))

    def parse_and_canonicalize(self, task: Task, raw: object) -> Filing:
        """The shared reading: identifier, comma, answer; first line for a record wins."""
        return shared_filing.parse_rows(self.row_identifiers(task.table), raw)

    def score(self, task: Task, canonical: Filing) -> tuple[float, tuple[RowOutcome, ...]]:
        """Equal weight per record, exact match after the fold, filed rows only."""
        identifiers = self.row_identifiers(task.table)
        truth = task.key if task.key else ("",) * len(identifiers)
        return shared_filing.score_rows(identifiers, truth, canonical, self.normalize_answer)

    # ----- the three cells -----

    def render_receipt(
        self, task: Task, canonical: Filing, truth: Sequence[str], feedback: Feedback
    ) -> ReceiptAST:
        """The shared grader's cell, built from the scorer's own outcomes."""
        graded = Task(
            label=task.label, task_id=task.task_id, surface=task.surface, table=task.table,
            text=task.text, key=tuple(truth), mask=task.mask,
        )
        _, outcomes = self.score(graded, canonical)
        return graded_receipt(task.task_id, outcomes, BLANK_TOKEN, UNFILED_TOKEN, feedback)

    def render_placebo(
        self, task: PublicTask, canonical: Filing, envelope: Envelope
    ) -> ReceiptAST:
        """The inert cell, from the public task: it has no rule to look at."""
        blind = Task(
            label=task.label, task_id=task.task_id, surface=task.surface, table=task.table,
            text="", key=(), mask=task.mask,
        )
        _, outcomes = self.score(blind, canonical)
        return placebo_receipt(task.task_id, outcomes, envelope, BLANK_TOKEN, UNFILED_TOKEN)

    def render_oracle(
        self, task_id: str, convention: Mapping[str, str], row_count: int = 0
    ) -> ReceiptAST:
        return render_oracle_cell(self.ORACLE, task_id, convention, row_count)

    def parse_oracle(self, ast: ReceiptAST) -> dict[str, str]:
        return parse_oracle_cell(self.ORACLE, ast)


def oracle_allowance(template: OracleTemplate, axes: Sequence[Axis], margin: int = 64) -> int:
    """The body allowance an oracle needs, as the longest statement over the whole space.

    A family maximum rather than a draw's, so the envelope does not move with the rule.
    """
    from shogym.envs.receipts.oracle import render_body

    longest = max(
        len("\n".join(render_body(template, convention))) for convention in conventions(axes)
    )
    return longest + margin


# --------------------------------------------------------------------------
# rendering a table as the task prints it
# --------------------------------------------------------------------------

FORMAT_NOTE = {
    "csv": "comma-separated",
    "tsv": "tab-separated",
    "semicsv": "semicolon-separated, quoted",
    "pipe": "pipe-delimited table",
    "fixed": "fixed-width columns",
    "jsonl": "one JSON object per line",
    "ini": "one bracketed block per record",
    "yaml": "a YAML list",
}


def render_records(header: Sequence[str], records: Sequence[Sequence[str]], fmt: str) -> str:
    """Records as one of the registered text formats. The first column is the identifier."""
    rows = [[str(v) for v in record] for record in records]
    if fmt == "csv":
        return "\n".join([",".join(header)] + [",".join(r) for r in rows])
    if fmt == "tsv":
        return "\n".join(["\t".join(header)] + ["\t".join(r) for r in rows])
    if fmt == "semicsv":
        return "\n".join(
            [";".join('"%s"' % h for h in header)]
            + [";".join('"%s"' % v for v in r) for r in rows]
        )
    if fmt == "pipe":
        widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(header)]
        out = [
            "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
            "|" + "|".join("-" * (w + 2) for w in widths) + "|",
        ]
        out += [
            "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(r)) + " |" for r in rows
        ]
        return "\n".join(out)
    if fmt == "fixed":
        widths = [
            max(len(h), max((len(r[i]) for r in rows), default=0)) + 2
            for i, h in enumerate(header)
        ]
        out = ["".join(h.ljust(widths[i]) for i, h in enumerate(header)).rstrip()]
        out += ["".join(v.ljust(widths[i]) for i, v in enumerate(r)).rstrip() for r in rows]
        return "\n".join(out)
    if fmt == "jsonl":
        return "\n".join(json.dumps(dict(zip(header, r))) for r in rows)
    if fmt == "ini":
        out: list[str] = []
        for r in rows:
            out.append("[%s]" % r[0])
            out.extend("%s = %s" % (h, v) for h, v in zip(header[1:], r[1:]))
            out.append("")
        return "\n".join(out).rstrip()
    if fmt == "yaml":
        out = []
        for r in rows:
            out.append("- %s: %s" % (header[0], r[0]))
            out.extend("  %s: %s" % (h, v) for h, v in zip(header[1:], r[1:]))
        return "\n".join(out)
    raise ValueError(f"no registered format {fmt!r}")


# --------------------------------------------------------------------------
# the design: what a table has to do before it is served
# --------------------------------------------------------------------------


def single_axis_pairs(axes: Sequence[Axis]) -> list[tuple[int, int, str]]:
    """Every unordered pair of rules that differ in exactly one decision.

    Indices are positions in `conventions(axes)`, which is the enumeration every helper
    here and every check in the package uses.
    """
    combos = conventions(axes)
    names = [axis.name for axis in axes]
    index = {tuple(c[n] for n in names): i for i, c in enumerate(combos)}
    out: list[tuple[int, int, str]] = []
    for i, combo in enumerate(combos):
        for axis in axes:
            here = axis.options.index(combo[axis.name])
            for later in axis.options[here + 1:]:
                other = dict(combo)
                other[axis.name] = later
                out.append((i, index[tuple(other[n] for n in names)], axis.name))
    return out


def coded(keys: Sequence[Sequence[str]], fold: Callable[[str], str]) -> np.ndarray:
    """Answer keys as integer codes, (rules, rows), after the scorer's own fold."""
    if not keys:
        return np.zeros((0, 0), dtype=np.int64)
    width = len(keys[0])
    out = np.zeros((len(keys), width), dtype=np.int64)
    vocabulary: dict[str, int] = {}
    for r, key in enumerate(keys):
        for c, value in enumerate(key):
            out[r, c] = vocabulary.setdefault(fold(value), len(vocabulary))
    return out


@dataclass(frozen=True)
class Separation:
    """What one table does across the whole rule set."""

    #: Whether every pair of rules gives this table a different answer key.
    distinct: bool
    #: Per decision, the fewest records any single change of that decision moves.
    fewest: Mapping[str, int]
    #: The share of records holding the commonest answer, at its worst over the rules.
    crowding: float

    def admits(self, need: int, max_crowding: float = 1.0) -> bool:
        return (
            self.distinct
            and all(v >= need for v in self.fewest.values())
            and self.crowding <= max_crowding
        )


def separation(
    codes: np.ndarray, pairs: Sequence[tuple[int, int, str]], axes: Sequence[Axis]
) -> Separation:
    """How well one table's answer codes separate the rules, from the codes alone."""
    distinct = len({row.tobytes() for row in codes}) == codes.shape[0]
    fewest = {axis.name: codes.shape[1] for axis in axes}
    if pairs:
        left = np.array([p[0] for p in pairs])
        right = np.array([p[1] for p in pairs])
        moved = (codes[left] != codes[right]).sum(axis=1)
        for (_, _, axis), count in zip(pairs, moved.tolist()):
            if count < fewest[axis]:
                fewest[axis] = int(count)
    crowding = 0.0
    width = codes.shape[1]
    if width:
        for row in codes:
            crowding = max(crowding, float(np.bincount(row).max()) / width)
    return Separation(distinct=distinct, fewest=fewest, crowding=crowding)


def shortfall(
    codes: np.ndarray,
    pairs: Sequence[tuple[int, int, str]],
    need: int,
    max_crowding: float,
) -> float:
    """How far one table is from admissible, as one number a local search can lower.

    Zero exactly when every pair of rules gives the table a different key, every change
    of one decision moves at least `need` records, and no rule puts more than
    `max_crowding` of the records in one answer. The terms are weighted so that a merged
    pair of rules costs more than any shortfall in movement, which costs more than
    crowding.
    """
    rules, width = codes.shape
    merged = rules - len({row.tobytes() for row in codes})
    deficit = 0
    if pairs:
        left = np.array([p[0] for p in pairs])
        right = np.array([p[1] for p in pairs])
        moved = (codes[left] != codes[right]).sum(axis=1)
        deficit = int(np.maximum(need - moved, 0).sum())
    allowed = int(max_crowding * width)
    crowded = 0
    for row in codes:
        crowded += max(int(np.bincount(row).max()) - allowed, 0)
    return 1000.0 * merged + 10.0 * deficit + 1.0 * crowded


def no_receipt_level(codes: np.ndarray) -> float:
    """The level a reader with no receipt reaches: the mean majority share per record."""
    rules, width = codes.shape
    if not rules or not width:
        return 0.0
    return float(np.mean([np.bincount(codes[:, c]).max() / rules for c in range(width)]))


def all_pairs(axes: Sequence[Axis], need_single: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Every unordered pair of rules, and how many records have to separate each.

    A pair that differs in one decision needs `need_single` separating records, which
    is what makes every decision move enough records; any other pair needs one, which
    is what makes the whole rule set distinguishable.
    """
    count = len(conventions(axes))
    single = {(i, j) for i, j, _ in single_axis_pairs(axes)}
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    need = np.array([need_single if p in single else 1 for p in pairs], dtype=np.int64)
    return pairs, need


def greedy_rows(
    candidates: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    need: np.ndarray,
    count: int,
    tiebreak: Sequence[float],
) -> list[int]:
    """Pick `count` candidate records that between them separate the pairs of rules.

    `candidates` is (candidates, rules) of answer codes. A record helps a pair of rules
    when it gives them different answers, and a pair stops needing help once `need`
    chosen records separate it. Each step takes the record that helps most, with the
    committed `tiebreak` values breaking ties, so the choice is a function of the
    stream that drew the candidates and of nothing else.
    """
    left = np.array([p[0] for p in pairs])
    right = np.array([p[1] for p in pairs])
    splits = (candidates[:, left] != candidates[:, right]).astype(np.float64)
    cover = np.zeros(len(pairs), dtype=np.int64)
    chosen: list[int] = []
    available = np.ones(candidates.shape[0], dtype=bool)
    jitter = np.asarray(tiebreak, dtype=float)
    for _ in range(count):
        want = np.maximum(need - cover, 0).astype(float)
        gain = splits @ want + 1e-3 * jitter
        gain[~available] = -np.inf
        best = int(np.argmax(gain))
        chosen.append(best)
        available[best] = False
        cover += splits[best].astype(np.int64)
    return chosen


def rebalance_rows(
    candidates: np.ndarray,
    chosen: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    need: np.ndarray,
    rng: Any,
    attempts: int = 3000,
    peak_weight: float = 0.0,
) -> list[int]:
    """Swap chosen records for others while every pair stays separated, to spread answers.

    A table whose answers crowd into one band under some rule is a table a lookup can
    copy: a relabel of the other sibling's answers lands on the crowded band often
    enough to score. So after the greedy choice has separated every pair, single swaps
    are tried, each kept only when every pair is still separated `need` times and the
    sum over rules of the squared band counts falls. The swaps are drawn from `rng`,
    the table's own surface stream, so the result is a function of that stream alone.

    `peak_weight` adds, per chosen record, how many rules agree on its commonest answer.
    A record every rule answers alike is a record a reader with no receipt at all gets
    right, so weighting it keeps the no-receipt level down while the bands spread.
    """
    left = np.array([p[0] for p in pairs])
    right = np.array([p[1] for p in pairs])
    splits = (candidates[:, left] != candidates[:, right]).astype(np.int64)
    picked = list(chosen)
    cover = splits[picked].sum(axis=0)
    width = int(candidates.max()) + 1 if candidates.size else 1
    rules = candidates.shape[1]
    counts = np.zeros((rules, width), dtype=np.int64)
    columns = np.arange(rules)
    for row in picked:
        counts[columns, candidates[row]] += 1
    peaks = np.array(
        [np.bincount(candidates[n], minlength=width).max() for n in range(candidates.shape[0])],
        dtype=np.float64,
    )

    def objective(table: np.ndarray, rows: Sequence[int]) -> float:
        return float((table ** 2).sum()) + peak_weight * float(peaks[list(rows)].sum())

    spread = objective(counts, picked)
    outside = [n for n in range(candidates.shape[0]) if n not in set(picked)]
    for _ in range(attempts):
        if not outside:
            break
        slot = rng.randrange(len(picked))
        choice = rng.randrange(len(outside))
        old, new = picked[slot], outside[choice]
        trial = cover - splits[old] + splits[new]
        if (trial < need).any():
            continue
        moved = counts.copy()
        moved[columns, candidates[old]] -= 1
        moved[columns, candidates[new]] += 1
        trial_rows = list(picked)
        trial_rows[slot] = new
        value = objective(moved, trial_rows)
        if value >= spread:
            continue
        picked[slot], outside[choice] = new, old
        cover, counts, spread = trial, moved, value
    return picked


# --------------------------------------------------------------------------
# the perfect reader of a delivered receipt
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PerfectReading:
    """What the ideal reader of one receipt files on the sibling task, and what it earns."""

    #: One answer per record of the sibling task, in printed order.
    filing: tuple[str, ...]
    #: How many rules of the whole set are consistent with what the receipt showed.
    consistent: int
    #: The reader's expected score under its own posterior: the mean majority share.
    expected: float
    #: The score its filing earns against the sibling's actual answer key.
    score: float


def receipt_evidence(receipt: ReceiptAST, blank_token: str = BLANK_TOKEN) -> dict[int, str]:
    """What a graded receipt shows about the rule: the correct answer on each reported row.

    A PASS row shows that the filed value was the answer, and a FAIL row prints the
    answer as its correction, so every reported row reveals that row's correct answer
    and nothing else. A row carrying neutral tokens was not reported and shows nothing.
    Keys are zero-based printed positions.
    """
    out: dict[int, str] = {}
    for row in receipt.rows:
        slots = {slot.name: slot.value for slot in row.slots}
        verdict = slots.get(VERDICT_SLOT)
        if verdict == PASS_TOKEN:
            value = row.observed
        elif verdict == FAIL_TOKEN:
            value = slots.get(CORRECTION_SLOT, "")
        else:
            continue
        out[row.ordinal - 1] = "" if value == blank_token else value
    return out


def consistent_rules(
    generator: Any, table: Any, evidence: Mapping[int, str]
) -> list[dict[str, str]]:
    """Every rule of the family's whole set whose answers agree with the evidence."""
    fold = generator.normalize_answer
    wanted = {row: fold(value) for row, value in evidence.items()}
    kept: list[dict[str, str]] = []
    for rule in support_of(generator):
        key = generator.key_for(table, rule)
        if all(fold(key[row]) == value for row, value in wanted.items()):
            kept.append(dict(rule))
    return kept


def majority_filing(
    generator: Any, table: Any, rules: Sequence[Mapping[str, str]]
) -> tuple[tuple[str, ...], float]:
    """The majority answer per record over `rules`, and the mean majority share.

    THE TIE RULE IS FIXED: among answers tied for the most rules, the one the task's
    published answer order lists first, and among answers it does not list, the
    alphabetically first after the scorer's fold.
    """
    if not rules:
        raise ValueError("no rule is consistent with the receipt, so it cannot be honest")
    fold = generator.normalize_answer
    keys = [generator.key_for(table, rule) for rule in rules]
    ranks = generator.answer_ranks(table) or ()
    order = {fold(token): n for n, token in enumerate(ranks)}
    filing: list[str] = []
    share = 0.0
    for row in range(len(keys[0])):
        counts: Counter[str] = Counter()
        spelled: dict[str, str] = {}
        for key in keys:
            value = fold(key[row])
            counts[value] += 1
            spelled.setdefault(value, key[row])
        top = max(counts.values())
        tied = [value for value, n in counts.items() if n == top]
        pick = min(tied, key=lambda value: (order.get(value, len(order)), value))
        filing.append(spelled[pick])
        share += top / float(len(keys))
    return tuple(filing), share / max(len(keys[0]), 1)


def perfect_reader(
    generator: Any, instance: Instance, receipt: ReceiptAST, side: str = "a"
) -> PerfectReading:
    """The ideal reader of this receipt, computed exactly over the whole rule set.

    Uniform prior over every rule the family can draw, conditioned on the rows the
    receipt reports, filing the per-record majority on the sibling task with the fixed
    tie rule of `majority_filing`. `side` is the task the receipt grades.
    """
    graded = instance.side(side)
    sibling = instance.side("b" if side.strip().lower() == "a" else "a")
    evidence = receipt_evidence(receipt, generator.BLANK_TOKEN)
    rules = consistent_rules(generator, graded.table, evidence)
    filing, expected = majority_filing(generator, sibling.table, rules)
    fold = generator.normalize_answer
    right = sum(1 for got, want in zip(filing, sibling.key) if fold(got) == fold(want))
    return PerfectReading(
        filing=filing,
        consistent=len(rules),
        expected=round(expected, 12),
        score=round(right / float(len(sibling.key)), 6),
    )


__all__ = [
    "BLANK_TOKEN",
    "COLUMN_TITLES",
    "CORRECTION_WIDTH",
    "FILLER_ALPHABET",
    "FORMAT_NOTE",
    "IDENTIFIER_WIDTH",
    "MAX_ANSWER",
    "OBSERVED_WIDTH",
    "SLOTS",
    "UNFILED_TOKEN",
    "VERDICT_WIDTH",
    "BandedRowsFamily",
    "PerfectReading",
    "Separation",
    "all_pairs",
    "coded",
    "committed_envelope",
    "consistent_rules",
    "envelope_size",
    "greedy_rows",
    "majority_filing",
    "no_receipt_level",
    "oracle_allowance",
    "perfect_reader",
    "rebalance_rows",
    "receipt_evidence",
    "render_records",
    "separation",
    "shortfall",
    "single_axis_pairs",
]
