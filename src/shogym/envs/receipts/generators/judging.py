"""The judging genre: placings within heats under hidden judging conventions.

DESIGN NOTE

The task. Twenty four entries in a competition, run in four heats of six. Each entry has
three judges' scores out of 20, the first from the head judge, and a flag saying whether
it was entered late; other columns never matter. The steward files every entry's placing
within its heat, first to sixth. The competition's rules extract says that entries are
placed by their mark, that some entries are disqualified and placed below every
qualified entry of their heat, that late entries are penalised and that ties are broken
by a house rule. It does not say how the mark is made, who is disqualified, what the late
penalty is or how ties are broken.

The rule space. Four hidden decisions, drawn uniformly and independently:

    marking     4 options   the mark is the total of the three scores, the total of the
                            best two, the middle score, or the total with the head
                            judge's score counted twice
    bar         4 options   nobody is disqualified, or an entry is disqualified when any
                            judge scored it below 5, when the head judge scored it below
                            7, or when its three scores total less than 27
    late        2 options   a late entry loses 3 from its mark, or is placed below every
                            on-time entry among the qualified entries of its heat
    ties        2 options   equal marks are separated by the head judge's score and then
                            by running order, or by running order alone

That is 64 rules. Sibling tasks A and B are two different competitions, with their own
names, formats and placing words, scored under the same rule.

WHY INFERENCE FROM THE RECEIPT IS HARD. A placing is relational: it is decided by the
other five entries of the heat, so a correction on one entry is evidence only together
with its heat-mates' scores, and it moves when any of them moves. The decisions interact
through the ordering. A disqualification pushes one entry down and pulls every entry
below it up by one; a late penalty of 3 matters or not depending on the gaps the marking
leaves, and the gaps differ between a total, a best two and a middle score; the tie rule
only speaks where the marking produces ties, which the middle score does often and the
totals rarely. So the same corrected placing can come from several combinations, and the
rule is pinned only by reading all four heats together. `build_table` searches for scores
under which every pair of the 64 rules gives a different answer key (so the perfect
reader of the full receipt scores 1.0 on the sibling), every change of one decision moves
at least three entries on each side, and few entries are placed alike by every rule.
Because every heat's placings are a permutation of first to sixth, the answers are
spread evenly under every rule, which keeps a relabel of the other sibling's answers far
below the copy bar.

WHY AN AGENT COULD GET BETTER WITH EXPERIENCE. The four families of decision are fixed
across tasks. An agent that has read a few receipts and rule statements learns to rank
each heat under each candidate marking and compare orders rather than single entries, to
look at the entries that would be disqualified under some bar and at late entries whose
place depends on the penalty, and to use ties as the probe for the tie rule. That is the
reading the perfect reader does by enumeration.

WHY THE FULL RECEIPT. The difficulty is meant to be the inference, not missing
information: every entry is reported, the perfect reader identifies the rule on every
admitted instance, and what an agent loses against it is what it fails to infer.

AUTHORSHIP. Designed and built in one session by Claude Opus 5.5 through Claude Code, as
one of three families built one after another; no helper agent did any part of it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from shogym.envs.receipts import filing as shared_filing
from shogym.envs.receipts import streams
from shogym.envs.receipts.filing import scope_sentence
from shogym.envs.receipts.generators.banded_rows import (
    FORMAT_NOTE,
    MAX_ANSWER,
    BandedRowsFamily,
    envelope_size,
    no_receipt_level,
    oracle_allowance,
    prose,
    render_records,
    shortfall,
    single_axis_pairs,
)
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.protocol import (
    Axis,
    Column,
    ConstructionExhausted,
    PublicTask,
    Shape,
    conventions,
)

AXES: tuple[Axis, ...] = (
    Axis("marking", ("mark_sum3", "mark_best2", "mark_middle", "mark_head2x"),
         "how the three judges' scores make a mark"),
    Axis("bar", ("bar_none", "bar_any5", "bar_head7", "bar_total27"),
         "who is disqualified"),
    Axis("late", ("late_minus3", "late_last"),
         "how a late entry is penalised"),
    Axis("ties", ("tie_head", "tie_order"),
         "how equal marks are separated"),
)

HEATS = 4
PER_HEAT = 6
ROWS = HEATS * PER_HEAT
STEPS = 3000
NEED = 3
MAX_CROWDING = 0.5
#: The no-receipt level a draw may have: the share of entries a reader with no receipt
#: places correctly by taking each entry's commonest placing over the rules.
MAX_NO_RECEIPT = 0.5

SHAPE = Shape(
    columns=(
        Column("entry", "a surface prefix and a running number; lower runs earlier"),
        Column("heat", "one of four heats of six, printed in heat order"),
        Column("entrant", "a name, never read by the rule"),
        Column("head judge", "a whole number from 0 to 20"),
        Column("second judge", "a whole number from 0 to 20"),
        Column("third judge", "a whole number from 0 to 20"),
        Column("late", "yes or no"),
    ),
    rows=ROWS,
    case="one entry awaiting its placing within its heat",
    note="every heat's placings are a permutation of first to sixth under every rule",
)

# ----- surfaces ---------------------------------------------------------------------

#: Placing words. The two pools never share a word, so the two siblings of an instance
#: publish disjoint answer orders.
ORDINALS = ("1st", "2nd", "3rd", "4th", "5th", "6th")
WORDS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth")

SURFACES: dict[str, dict[str, Any]] = {
    "baking": dict(
        org="Wendlebury Show", title="bake-off heats", entry="bake", idpfx="BK",
        idstart=101, fmt="csv", heat_names=("Heat 1", "Heat 2", "Heat 3", "Heat 4"),
        cols=("entry", "heat", "baker", "head_judge", "judge_2", "judge_3", "late_entry"),
        people=("A. Moss", "R. Iqbal", "S. Fenn", "T. Oduya", "L. Graves", "P. Nair",
                "C. Holt", "J. Barros", "M. Keane", "E. Sato", "D. Lyle", "N. Afolabi"),
        manual="Show Schedule", places=ORDINALS),
    "choir": dict(
        org="Tamar Choral Festival", title="solo class heats", entry="singer",
        idpfx="SG", idstart=31, fmt="pipe",
        heat_names=("Morning A", "Morning B", "Afternoon A", "Afternoon B"),
        cols=("entry_no", "session", "singer", "adjudicator", "panel_2", "panel_3",
              "late"),
        people=("H. Bright", "O. Varga", "K. Mistry", "F. Lowry", "I. Castell", "G. Dunn",
                "Y. Park", "B. Achebe", "Z. Rowe", "Q. Tan", "U. Hale", "V. Rossi"),
        manual="Festival Rules", places=ORDINALS),
    "robotics": dict(
        org="Midland Schools Robotics", title="build challenge heats", entry="robot",
        idpfx="RB", idstart=240, fmt="jsonl",
        heat_names=("Pool W", "Pool X", "Pool Y", "Pool Z"),
        cols=("entry", "pool", "team", "lead_judge", "judge_b", "judge_c", "late_reg"),
        people=("Ashby", "Bramley", "Colwick", "Dunkirk", "Epperstone", "Farnsfield",
                "Gedling", "Hucknall", "Ilkeston", "Kimberley", "Lambley", "Mapperley"),
        manual="Challenge Handbook", places=ORDINALS),
    "photo": dict(
        org="Coldharbour Camera Club", title="print competition rounds", entry="print",
        idpfx="PR", idstart=11, fmt="tsv",
        heat_names=("Round 1", "Round 2", "Round 3", "Round 4"),
        cols=("print_no", "round", "author", "chief_judge", "judge_2", "judge_3",
              "late_hand_in"),
        people=("W. Pike", "A. Oyelaran", "M. Frost", "S. Kaur", "D. Hollis", "E. Brook",
                "R. Stone", "N. Vale", "T. Quaye", "L. Marsh", "J. Wren", "C. Ibe"),
        manual="Competition Rules", places=ORDINALS),
    "dogshow": dict(
        org="Kestle Open Dog Show", title="junior handling classes", entry="handler",
        idpfx="DH", idstart=501, fmt="fixed",
        heat_names=("Class A", "Class B", "Class C", "Class D"),
        cols=("ring_no", "class", "handler", "senior_judge", "judge_two", "judge_three",
              "late"),
        people=("Poppy", "Rufus", "Tilly", "Monty", "Bramble", "Hector", "Maisie",
                "Otis", "Willow", "Jasper", "Nell", "Barney"),
        manual="Show Regulations", places=WORDS),
    "dance": dict(
        org="Northgate Dance Festival", title="solo heats", entry="dancer",
        idpfx="DN", idstart=71, fmt="yaml",
        heat_names=("Heat A", "Heat B", "Heat C", "Heat D"),
        cols=("number", "heat", "dancer", "principal_judge", "judge_2", "judge_3",
              "late_entry"),
        people=("I. Adair", "K. Blake", "L. Chen", "M. Doyle", "N. Evans", "O. Fox",
                "P. Gill", "R. Hart", "S. Ince", "T. Jones", "U. Kaye", "V. Lord"),
        manual="Festival Syllabus", places=WORDS),
    "brass": dict(
        org="Pennine Brass Contest", title="solo slow melody heats", entry="soloist",
        idpfx="SM", idstart=301, fmt="ini",
        heat_names=("Heat I", "Heat II", "Heat III", "Heat IV"),
        cols=("draw_no", "heat", "soloist", "adjudicator", "assessor_2", "assessor_3",
              "late"),
        people=("Black Dyke", "Brighouse", "Cory", "Grimethorpe", "Fodens", "Leyland",
                "Hepworth", "Wingates", "Rothwell", "Tredegar", "Flowers", "Whitburn"),
        manual="Contest Rules", places=WORDS),
    "science": dict(
        org="Harbourside Science Fair", title="project judging rounds", entry="project",
        idpfx="SF", idstart=820, fmt="semicsv",
        heat_names=("Hall 1", "Hall 2", "Hall 3", "Hall 4"),
        cols=("project", "hall", "student", "lead_judge", "judge_2", "judge_3",
              "late_poster"),
        people=("E. Nwosu", "F. Ortiz", "G. Pratt", "H. Quinn", "I. Reyes", "J. Shaw",
                "K. Tully", "L. Umar", "M. Voss", "N. Webb", "O. Yates", "P. Zeller"),
        manual="Fair Rules", places=WORDS),
}

POOL_A = ("baking", "choir", "robotics", "photo")
POOL_B = ("dogshow", "dance", "brass", "science")


# --------------------------------------------------------------------------
# the table and the rule
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryRow:
    row_id: str
    heat: int
    order: int
    name: str
    scores: tuple[int, int, int]
    late: bool


@dataclass(frozen=True)
class JudgingTable:
    surface: str
    rows: tuple[EntryRow, ...]
    body: str

    @property
    def dom(self) -> dict[str, Any]:
        return SURFACES[self.surface]


def mark_of(scores: tuple[int, int, int], marking: str) -> int:
    head, second, third = scores
    if marking == "mark_sum3":
        return head + second + third
    if marking == "mark_best2":
        return head + second + third - min(scores)
    if marking == "mark_middle":
        return sorted(scores)[1]
    return 2 * head + second + third


def disqualified(scores: tuple[int, int, int], bar: str) -> bool:
    if bar == "bar_any5":
        return min(scores) < 5
    if bar == "bar_head7":
        return scores[0] < 7
    if bar == "bar_total27":
        return sum(scores) < 27
    return False


def heat_places(rows: list[EntryRow], convention: Mapping[str, str]) -> dict[int, int]:
    """Each entry's place in its heat, zero for first, keyed by position in `rows`."""
    keyed = []
    for position, row in enumerate(rows):
        mark = mark_of(row.scores, convention["marking"])
        out = 1 if disqualified(row.scores, convention["bar"]) else 0
        behind = 0
        if row.late:
            if convention["late"] == "late_minus3":
                mark -= 3
            elif not out:
                behind = 1
        head = row.scores[0] if convention["ties"] == "tie_head" else 0
        keyed.append(((out, behind, -mark, -head, row.order), position))
    return {position: place for place, (_, position) in enumerate(sorted(keyed))}


def key_for(table: JudgingTable, convention: Mapping[str, str]) -> tuple[str, ...]:
    """Each entry's placing in its heat, in printed order."""
    places = table.dom["places"]
    out: list[str] = [""] * len(table.rows)
    for heat in range(HEATS):
        members = [n for n, row in enumerate(table.rows) if row.heat == heat]
        placed = heat_places([table.rows[n] for n in members], convention)
        for position, place in placed.items():
            out[members[position]] = places[place]
    return tuple(out)


ALL_CONVENTIONS: tuple[dict[str, str], ...] = tuple(conventions(AXES))
PAIRS = single_axis_pairs(AXES)


def all_codes(rows: list[EntryRow]) -> np.ndarray:
    """(rules, entries) of places, zero for first, for every rule in `conventions` order."""
    out = np.zeros((len(ALL_CONVENTIONS), len(rows)), dtype=np.int64)
    heats = [[n for n, row in enumerate(rows) if row.heat == heat] for heat in range(HEATS)]
    for index, rule in enumerate(ALL_CONVENTIONS):
        for members in heats:
            placed = heat_places([rows[n] for n in members], rule)
            for position, place in placed.items():
                out[index, members[position]] = place
    return out


# --------------------------------------------------------------------------
# building heats that separate every rule
# --------------------------------------------------------------------------


#: The widest gap allowed between an entry's highest and lowest score. Judges disagree,
#: and the rule has to be read through their disagreement, but a panel that scores one
#: entry 1 and 18 is not a panel anyone would recognise.
MAX_SPREAD = 9


def _entry(rng: random.Random, heat: int, order: int) -> EntryRow:
    quality = rng.randint(5, 16)
    scores = tuple(max(0, min(20, quality + rng.randint(-3, 3))) for _ in range(3))
    if rng.random() < 0.2:
        low = rng.randrange(3)
        scores = tuple(
            max(0, min(scores) - rng.randint(1, 3)) if k == low else v
            for k, v in enumerate(scores)
        )
    return EntryRow(
        row_id="", heat=heat, order=order, name="",
        scores=(scores[0], scores[1], scores[2]), late=rng.random() < 0.3,
    )


def _layout(rng: random.Random) -> list[EntryRow]:
    return [
        _entry(rng, heat, heat * PER_HEAT + k) for heat in range(HEATS) for k in range(PER_HEAT)
    ]


def _mutate(rng: random.Random, rows: list[EntryRow]) -> list[EntryRow]:
    rows = list(rows)
    n = rng.randrange(len(rows))
    row = rows[n]
    if rng.random() < 0.2:
        rows[n] = EntryRow(row.row_id, row.heat, row.order, row.name, row.scores, not row.late)
        return rows
    scores = list(row.scores)
    k = rng.randrange(3)
    scores[k] = max(0, min(20, scores[k] + rng.choice((-3, -2, -1, 1, 2, 3))))
    if max(scores) - min(scores) > MAX_SPREAD:
        return rows
    rows[n] = EntryRow(
        row.row_id, row.heat, row.order, row.name, (scores[0], scores[1], scores[2]), row.late
    )
    return rows


def _objective(codes: np.ndarray) -> float:
    return shortfall(codes, PAIRS, NEED, MAX_CROWDING) + 20.0 * max(
        no_receipt_level(codes) - MAX_NO_RECEIPT, 0.0
    )


def build_heats(rng: random.Random) -> list[EntryRow] | None:
    """Four heats whose every rule gives a distinct key and every decision moves three."""
    rows = _layout(rng)
    best = _objective(all_codes(rows))
    for _ in range(STEPS):
        if best == 0.0:
            break
        trial = _mutate(rng, rows)
        value = _objective(all_codes(trial))
        if value <= best:
            rows, best = trial, value
    return rows if best == 0.0 else None


def render_body(rows: tuple[EntryRow, ...], dom: Mapping[str, Any]) -> str:
    records = [
        (
            row.row_id, dom["heat_names"][row.heat], row.name, str(row.scores[0]),
            str(row.scores[1]), str(row.scores[2]), "yes" if row.late else "no",
        )
        for row in rows
    ]
    return render_records(dom["cols"], records, dom["fmt"])


# --------------------------------------------------------------------------
# the task text
# --------------------------------------------------------------------------

TASK_TEMPLATE = """{org} {title}

You are the steward. Give every {entry} below its placing within its {heat_word},
using the rules extract that follows.

RULES EXTRACT ({manual})

{rule}

  The placings, in order, are: {placings}.

{scope}
{receipt}
ENTRIES ({fmtnote}; in {heat_word} order, running order within each)

{body}

File one line per {entry}: the {entry} number in column {c_id}, a comma, and its
placing, with no header and no other text. File every {entry}, in the order they
appear above.
"""

#: The extract's paragraph, filled to width once the surface's words are in.
RULE_PARAGRAPH = (
    "Every {entry} is scored out of 20 by three judges; the head judge's score is in "
    "column {c_head} and the other two are in columns {c_two} and {c_three}. Within each "
    "{heat_word} the {entry}s are placed from {first} to {last} by their mark, highest "
    "first. Some {entry}s are disqualified, and a disqualified {entry} is placed below "
    "every qualified {entry} of its {heat_word}, the disqualified being placed among "
    "themselves by mark in the same way. A late {entry} (column {c_late}) is penalised. "
    "Equal marks are separated by the house tie rule; running order means the {entry} "
    "number in column {c_id}, lower first. How the three scores make a mark, which "
    "{entry}s are disqualified, what the late penalty is and how ties are broken are set "
    "by the house rules, which this extract does not reproduce. Column {c_name} plays no "
    "part."
)

RECEIPT_SENTENCE: tuple[str, ...] = (
    "The receipt for these heats reports the verdict and the correct placing for",
    "selected entries only, and the lines for the other entries say nothing about",
    "whether they were right.",
)


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------

ORACLE_HEAD: tuple[str, ...] = (
    "HOUSE JUDGING RULES FOR THESE HEATS AND THE NEXT ONES YOU FILE (they",
    "govern the heats above and the next heats you are given, whichever",
    "competition they name; the heats after that are scored under rules of",
    "their own)",
    "",
)

ORACLE_TEMPLATE = OracleTemplate(
    head=ORACLE_HEAD,
    sentences={
        "marking": "The mark: {}.",
        "bar": "Disqualification: {}.",
        "late": "Late entries: {}.",
        "ties": "Ties: {}.",
    },
    phrases={
        "marking": {
            "mark_sum3": "the total of all three judges' scores",
            "mark_best2": "the total of the two highest of the three scores, the lowest "
                          "score being dropped",
            "mark_middle": "the middle one of the three scores, neither the highest nor "
                           "the lowest",
            "mark_head2x": "the head judge's score counted twice plus the other two scores",
        },
        "bar": {
            "bar_none": "no entry is disqualified",
            "bar_any5": "an entry is disqualified when any one judge scored it below 5",
            "bar_head7": "an entry is disqualified when the head judge scored it below 7",
            "bar_total27": "an entry is disqualified when its three scores add up to less "
                           "than 27",
        },
        "late": {
            "late_minus3": "a late entry, qualified or not, loses 3 from its mark and is "
                           "otherwise placed like any other",
            "late_last": "a late entry keeps its mark, and a late qualified entry is "
                         "placed below every on-time qualified entry of its heat, late "
                         "qualified entries being placed among themselves by mark; a late "
                         "disqualified entry is placed among the disqualified by its mark "
                         "as usual",
        },
        "ties": {
            "tie_head": "equal marks are separated by the head judge's score, higher "
                        "first, and then by running order",
            "tie_order": "equal marks are separated by running order alone, whatever "
                         "the judges scored",
        },
    },
)

ENVELOPE_SIZE = envelope_size(ROWS, oracle_allowance(ORACLE_TEMPLATE, AXES))


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class JudgingGenerator(BandedRowsFamily):
    """The judging genre, as the generator protocol wants it."""

    name = "judging"
    genre = "placings within heats under hidden judging rules"
    SHAPE = SHAPE
    AXES = AXES
    ORACLE = ORACLE_TEMPLATE
    ENVELOPE_SIZE = ENVELOPE_SIZE
    ROWS = ROWS

    def surface_for(self, ordinal: int, label: str) -> str:
        i = int(ordinal)
        if label.upper() == "A":
            return POOL_A[i % len(POOL_A)]
        return POOL_B[(i + i // len(POOL_B)) % len(POOL_B)]

    def surface_templates(self) -> tuple[str, ...]:
        return tuple(POOL_A) + tuple(POOL_B)

    def build_table(self, master: bytes, ordinal: int, label: str) -> JudgingTable:
        """Heats from this side's own surface stream that separate every rule."""
        surface = self.surface_for(ordinal, label)
        dom = SURFACES[surface]
        stream = streams.SURFACE_A if label.upper() == "A" else streams.SURFACE_B
        for attempt in range(40):
            rng = streams.rng(master, stream, self.name, ordinal, attempt)
            found = build_heats(rng)
            if found is None:
                continue
            names = list(dom["people"])
            rng.shuffle(names)
            rows = tuple(
                EntryRow(
                    row_id="%s-%d" % (dom["idpfx"], dom["idstart"] + row.order),
                    heat=row.heat, order=row.order,
                    name=names[row.order % len(names)] if row.order < len(names)
                    else names[row.order % len(names)] + " II",
                    scores=row.scores, late=row.late,
                )
                for row in found
            )
            return JudgingTable(surface=surface, rows=rows, body=render_body(rows, dom))
        raise ConstructionExhausted(
            "no judging heats for %s at instance %d separated every rule" % (surface, ordinal)
        )

    def table_record(self, table: JudgingTable) -> dict[str, Any]:
        return {
            "surface": table.surface,
            "rows": [
                {
                    "row_id": row.row_id,
                    "heat": row.heat,
                    "order": row.order,
                    "name": row.name,
                    "scores": list(row.scores),
                    "late": row.late,
                }
                for row in table.rows
            ],
            "body": table.body,
        }

    def key_for(self, table: JudgingTable, convention: Mapping[str, str]) -> tuple[str, ...]:
        return key_for(table, convention)

    def row_identifiers(self, table: JudgingTable) -> tuple[str, ...]:
        return tuple(row.row_id for row in table.rows)

    def row_classes(self, table: JudgingTable) -> tuple[str, ...]:
        return tuple("late" if row.late else "on time" for row in table.rows)

    def answer_ranks(self, table: JudgingTable) -> tuple[str, ...]:
        return tuple(table.dom["places"])

    def describe(self, task: PublicTask) -> str:
        table: JudgingTable = task.table
        dom = table.dom
        cols = dom["cols"]
        heat_word = cols[1]
        places = dom["places"]
        rule = RULE_PARAGRAPH.format(
            entry=dom["entry"], heat_word=heat_word, c_id=cols[0], c_name=cols[2],
            c_head=cols[3], c_two=cols[4], c_three=cols[5], c_late=cols[6],
            first=places[0], last=places[-1],
        )
        return TASK_TEMPLATE.format(
            org=dom["org"], title=dom["title"], entry=dom["entry"], manual=dom["manual"],
            heat_word=heat_word, c_id=cols[0], rule=prose(rule),
            placings=", ".join(places),
            scope=scope_sentence(task.label),
            receipt=shared_filing.receipt_sentence(self.RECEIPT_POLICY, RECEIPT_SENTENCE),
            fmtnote=FORMAT_NOTE[dom["fmt"]], body=table.body,
        )


def _check_vocabulary() -> None:
    """Placings fit the slot, and the two pools never share a placing word."""
    a_words = {shared_filing.fold(p) for s in POOL_A for p in SURFACES[s]["places"]}
    b_words = {shared_filing.fold(p) for s in POOL_B for p in SURFACES[s]["places"]}
    if a_words & b_words:
        raise ValueError(f"the two pools share placing words {sorted(a_words & b_words)}")
    for surface, dom in SURFACES.items():
        if len(dom["places"]) != PER_HEAT:
            raise ValueError(f"{surface} does not publish one placing per entry of a heat")
        if any(len(p) > MAX_ANSWER for p in dom["places"]):
            raise ValueError(f"a placing of {surface} is wider than the slot")


_check_vocabulary()

GENERATOR = JudgingGenerator()


__all__ = [
    "ALL_CONVENTIONS",
    "AXES",
    "ENVELOPE_SIZE",
    "GENERATOR",
    "ORACLE_TEMPLATE",
    "POOL_A",
    "POOL_B",
    "SHAPE",
    "SURFACES",
    "EntryRow",
    "JudgingGenerator",
    "JudgingTable",
    "all_codes",
    "disqualified",
    "heat_places",
    "key_for",
    "mark_of",
]
