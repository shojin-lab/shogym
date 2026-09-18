"""Reading a filing, scoring it by row, and saying which tasks share a convention.

Three things every generator of this shape needs, in one place rather than in each
of them.

THE READING. A line is a row identifier, a comma, and a value. Identifiers match
case-insensitively after whitespace is collapsed. The FIRST line for an identifier
wins and later ones are recorded as duplicates. A line naming no known identifier is
an extra. A row with no line is an omission and canonicalizes to empty. One
forgiving reading is registered and it is narrow: a filing with no commas anywhere
is read positionally only when it has exactly one line per printed row, so a
paragraph of prose cannot be read as an answer to the first rows of a table.

THE FOLD IS PART OF THE READING, not a renderer's afterthought. Every byte the agent
types reaches a cell through here, and the serializer refuses a field that is not
ASCII, so a value left exactly as typed lets one accented letter decide whether the
fork renders at all. Folding here means the scorer and both renderers see one value,
and a character that will not fold costs the agent its identifier match, which is a
reason-coded outcome rather than an exception out of the seal.

THE SCORE. The fraction of printed rows the filing got right: equal weight per row,
the denominator is the printed row count, rounded to six places. A row counts only if
it was FILED, because filing an empty value and filing nothing at all are different
acts and one family's correct answer can be the empty value.

WHICH TASKS SHARE A CONVENTION. One convention is drawn for a pair of sibling tasks,
and a reader that is never told so has no reason to carry to the second task anything
it worked out on the first. A run of 119 readers measured what that costs: 64 of them
held a correct statement of the rule and filed the second task under a different one,
because nothing they could see said the two were scored together. So the description
says which tasks share the conventions, by what the reader does with them rather than
by any name printed on them. The sentences are HERE rather than in one generator with
the other copying them, because two copies of a registered string are two strings the
moment one of them is edited.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

from shogym.envs.receipts.protocol import (
    Filing,
    NoFiling,
    ReceiptPolicy,
    RowOutcome,
    SealedSubmission,
)

#: WHICH TASKS SHARE A CONVENTION, SAID IN THE TASK ITSELF.
#: It states that two consecutive tasks share a rule, which is the design's own
#: premise, and nothing about what that rule is. It is a function of the sibling
#: label alone, so the same bytes reach every arm and no arm reads anything here
#: another cannot.
SCOPE_SENTENCE: dict[str, tuple[str, ...]] = {
    "A": (
        "This schedule and the next one you file are scored under the same house",
        "conventions; the pair after that is scored under conventions of its own.",
    ),
    "B": (
        "This schedule is scored under the same house conventions as the one you filed",
        "before it.",
    ),
}


def scope_sentence(label: str) -> str:
    """The sentence for one sibling, as the description prints it."""
    lines = SCOPE_SENTENCE.get(label.strip().upper())
    if lines is None:
        raise ValueError(f"a family has siblings A and B, not {label!r}")
    return "\n".join(lines)


def receipt_sentence(policy: ReceiptPolicy, sentence: Sequence[str]) -> str:
    """The genre's own sentence about what its receipt reports, as a description prints it.

    WHAT THE RECEIPT WILL SAY, SAID BEFORE THE WORK IS DONE. A receipt that reports two
    rows of twenty four is a receipt whose silence on the other twenty two means nothing,
    and a reader who was not told that has been handed twenty two rows of apparent
    evidence that the task was filed correctly. So the description says which of the two
    it is, and it names no selected row: the selection is drawn before any filing exists
    and a reader can see it in the receipt anyway, while a reader who could see it here
    could file the rest at random and lose nothing.

    THE RULE IS SHARED AND THE WORDS ARE THE GENRE'S. When the sentence is printed, that
    it is printed only under a policy that samples, and that it is the same bytes in
    every arm and under every convention are one registration for every family. What it
    calls a row and what it calls an answer is not one registration: a schedule has
    records and bands, a batch has forms and daughter forms, and one sentence over both
    would name neither. It is empty under the full receipt, which promises a verdict and
    a correction on every row and so has nothing to qualify.
    """
    if not policy.samples:
        return ""
    return "\n" + "\n".join(sentence) + "\n"


def normalize(value: object) -> str:
    """One filed value as a family reads it: collapsed whitespace, printable ASCII.

    Control bytes fold for the same reason the rest does. ESC, NUL and backspace are
    ASCII, so the serializer passes them, and the observed column is echoed into the
    placebo, where an escape sequence is highlighting in the arm that is meant to be
    inert.
    """
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return "".join(ch if " " <= ch <= "~" else "?" for ch in text)


def fold(value: object) -> str:
    """The same value with ASCII letter case removed, which is how answers compare."""
    return normalize(value).lower()


def lines_of(raw: object) -> list[str] | None:
    """The filing as lines, or None when there is nothing readable in it at all.

    A list or tuple is accepted for direct Python calls, with `None` entries standing
    for an empty line and scalar items converted; a structure of anything else is
    unreadable rather than stringified, because `str` of an object is a line nobody
    filed.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return [ln for ln in (line.strip() for line in raw.splitlines()) if ln]
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            if item is None:
                out.append("")
                continue
            if not isinstance(item, (str, int, float)):
                return None
            out.append(str(item).strip())
        return out
    return None


def parse_rows(identifiers: Sequence[str], raw: object) -> Filing:
    """One canonical value per printed row, by the registered reading rules.

    `identifiers` is the printed order. Everything else about the family, its
    columns, its answers and its normalization, is irrelevant to reading a line.
    """
    lines = lines_of(raw)
    if lines is None:
        return NoFiling("unreadable")
    if not any(line for line in lines):
        return NoFiling("empty")

    position = {fold(row_id): i for i, row_id in enumerate(identifiers)}
    values: list[str] = [""] * len(identifiers)
    seen: set[int] = set()
    duplicates: list[str] = []
    extras: list[str] = []
    filed = 0

    if not any("," in line for line in lines):
        if len(lines) != len(identifiers):
            return NoFiling("no_known_identifier")
        for i, line in enumerate(lines):
            if line:
                values[i] = normalize(line)
                seen.add(i)
                filed += 1
    else:
        for line in lines:
            if not line:
                continue
            head, _, tail = line.partition(",")
            index = position.get(fold(head))
            if index is None:
                extras.append(normalize(head))
                continue
            filed += 1
            if index in seen:
                duplicates.append(identifiers[index])
                continue
            values[index] = normalize(tail)
            seen.add(index)
        if not seen and not any(v for v in values):
            return NoFiling("no_known_identifier")

    omissions = tuple(identifiers[i] for i in range(len(identifiers)) if i not in seen)
    return SealedSubmission(
        values=tuple(values),
        filed=tuple(i in seen for i in range(len(identifiers))),
        filed_rows=filed,
        duplicates=tuple(duplicates),
        extras=tuple(extras),
        omissions=omissions,
    )


def score_rows(
    identifiers: Sequence[str],
    truth: Sequence[str],
    canonical: Filing,
    normalize_answer: Callable[[str], str],
) -> tuple[float, tuple[RowOutcome, ...]]:
    """The sealed scalar and what the filing did on every row.

    A NoFiling scores zero and still reports every row, so a receipt rendered against
    it has the same shape as any other. `normalize_answer` is the family's own, and it
    has to be the function the family's `normalize_answer` exposes, because gate H
    codes the sibling's legal actions with that one.
    """
    submitted = canonical if isinstance(canonical, SealedSubmission) else None
    outcomes: list[RowOutcome] = []
    for i, identifier in enumerate(identifiers):
        got = ""
        was_filed = False
        if submitted is not None:
            got = submitted.values[i] if i < len(submitted.values) else ""
            was_filed = submitted.filed[i] if i < len(submitted.filed) else False
        correct = truth[i] if i < len(truth) else ""
        outcomes.append(
            RowOutcome(
                ordinal=i + 1,
                identifier=identifier,
                filed=got,
                was_filed=was_filed,
                correct=correct,
                matched=was_filed
                and normalize_answer(got) == normalize_answer(correct),
            )
        )
    if not outcomes:
        return 0.0, ()
    matched = sum(1 for o in outcomes if o.matched)
    return round(matched / float(len(outcomes)), 6), tuple(outcomes)


__all__ = [
    "SCOPE_SENTENCE",
    "fold",
    "lines_of",
    "normalize",
    "parse_rows",
    "receipt_sentence",
    "score_rows",
    "scope_sentence",
]
