"""The scorecard genre: priority banding under a hidden house points schedule.

DESIGN NOTE

The task. A list of twenty four open work orders, each with a hazard grade, the
criticality of the asset it concerns, how many people it affects, how many days it has
been open and whether it repeats an earlier fault, plus columns that never matter. A
scorecard extract tells the planner that each order's priority score is the sum of the
points it earns on four factors, that exposure is scored from the people affected or the
days open, and prints the table that turns a score into one of six bands. It does not
reproduce the points schedule. The planner files one band per order.

The rule space. Four hidden decisions, drawn uniformly and independently:

    hazard_points       4 options   what a minor and a serious hazard earn: 1 and 3,
                                    2 and 3, 1 and 4, or 2 and 4 points
    criticality_points  4 options   what a medium and a high criticality asset earn:
                                    1 and 2, 1 and 3, 2 and 2, or 0 and 3 points
    exposure_basis      3 options   one point per full 50 people affected, one point per
                                    full week open, or the larger of the two
    repeat_rule         2 options   a repeat fault earns two extra points, or is placed
                                    one band above the band its score gives

That is 96 rules. Sibling tasks A and B are two different organisations, with their own
column names, formats, band names and band cuts, scored under the same rule.

WHY INFERENCE FROM THE RECEIPT IS HARD. The receipt is complete and honest: every order
gets a verdict and, where the band was wrong, the correct band. Yet no single correction
names a decision. A band is an interval on a sum of three or four unknown point values,
so one corrected order is consistent with dozens of schedules, and the schedule is pinned
only by intersecting the intervals of many orders at once, which is a small integer
constraint problem rather than a lookup. The decisions also interact: whether exposure is
counted in people or in weeks decides which orders land on a band cut, which decides what
a corrected band says about the hazard points, and a repeat fault's correction reads
differently under "two extra points" and "one band up" depending on how near the cut its
score sits. `build_table` chooses the orders so that every pair of the 96 rules gives a
different answer key (the full receipt identifies the rule exactly, so the perfect reader
scores 1.0 on the sibling), so that every change of one decision moves at least three
orders on each side, and so that no rule puts more than half the orders in one band
while few orders are answered alike by every rule. The point options are chosen with
similar totals, so that no schedule crowds the orders into one band: a crowded band is
what a relabel of the other sibling's answers lands on often enough to score. An agent that anchors on the first explanation that fits a few rows,
or that never considers that exposure could be counted in time rather than people, files
the sibling under a wrong schedule that still agrees with most of its rows.

WHY AN AGENT COULD GET BETTER WITH EXPERIENCE. Successive tasks redraw the schedule from
the same four families of decision. An agent that has read several receipts and oracles
learns what kind of thing is hidden (point pairs per grade, a choice of exposure measure,
two treatments of repeats) and which rows are informative (orders sitting on a band cut,
repeat faults near a cut, orders whose hazard and criticality differ). That turns the
next receipt from an open-ended curve-fitting exercise into a search over a small, known
space, which is exactly the reading skill the perfect reader has from the start.

WHY THE FULL RECEIPT. The ledger moved to a sampled receipt because its full receipt made
the rule easy to read off. Here the difficulty is meant to lie in the inference, not in
the information: the perfect reader of the full receipt identifies the rule on every
admitted instance, and what an agent loses against it is what it fails to infer.

Construction, scoring and every cell follow the ledger through `banded_rows`: the public
task is built from the surface stream alone, only the key reads the rule, scoring is
equal-weight exact match after the shared fold, and the placebo prints committed neutral
tokens from the public task.
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
    all_pairs,
    envelope_size,
    greedy_rows,
    oracle_allowance,
    rebalance_rows,
    render_records,
    separation,
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
    Axis("hazard_points", ("h1s3", "h2s3", "h1s4", "h2s4"),
         "points for a minor and for a serious hazard"),
    Axis("criticality_points", ("m1x2", "m1x3", "m2x2", "m0x3"),
         "points for a medium and for a high criticality asset"),
    Axis("exposure_basis", ("people50", "days7", "larger_of"),
         "how exposure earns points"),
    Axis("repeat_rule", ("plus2points", "oneband_up"),
         "how a repeat fault is treated"),
)

#: Points by grade, none first. A grade of none and a criticality of low earn nothing,
#: and the task says so; the rest is the hidden schedule.
HAZARD = {
    "h1s3": (0, 1, 3),
    "h2s3": (0, 2, 3),
    "h1s4": (0, 1, 4),
    "h2s4": (0, 2, 4),
}
CRITICALITY = {
    "m1x2": (0, 1, 2),
    "m1x3": (0, 1, 3),
    "m2x2": (0, 2, 2),
    "m0x3": (0, 0, 3),
}
HAZARD_WORDS = ("none", "minor", "serious")
CRITICALITY_WORDS = ("low", "medium", "high")

ROWS = 24
#: How far a record's own columns range. People up to 349 and days up to 48, so every
#: exposure measure runs from nothing to six points.
MAX_PEOPLE = 349
MAX_DAYS = 48
#: How many random candidate orders a build draws before choosing its twenty four.
CANDIDATES = 2000
#: Every single change of one decision has to move at least this many orders.
NEED = 3
#: How many single swaps the rebalancing search tries, and how much it weighs a record
#: every rule answers alike against a band that crowds under some rule.
SWAPS = 6000
PEAK_WEIGHT = 20.0
#: No rule may put more than half the orders in one band. A crowded band is what a
#: relabel of the other sibling's answers lands on often enough to score.
MAX_CROWDING = 0.5

SHAPE = Shape(
    columns=(
        Column("order id", "a surface prefix and a running number"),
        Column("hazard grade", "none, minor or serious"),
        Column("asset criticality", "low, medium or high"),
        Column("people affected", "a whole number from 0 to 349"),
        Column("days open", "a whole number from 0 to 48"),
        Column("repeat fault", "yes or no"),
        Column("trade", "drawn from the surface's trades, never read by the rule"),
        Column("location", "drawn from the surface's locations, never read by the rule"),
    ),
    rows=ROWS,
    case="one open work order awaiting a priority band",
    note="rows are chosen so every single-decision change moves at least three orders",
)

# ----- surfaces ---------------------------------------------------------------------

SURFACES: dict[str, dict[str, Any]] = {
    "estates": dict(
        org="St Aldric's Hospital", title="estates work order list", entity="work order",
        idpfx="WO", idstart=4410, fmt="csv", manual="Estates Maintenance Policy",
        cols=("order_id", "hazard_grade", "asset_criticality", "people_affected",
              "days_open", "repeat_fault", "trade", "ward"),
        trades=("Electrical", "Plumbing", "HVAC", "Joinery", "Medical gas"),
        places=("Ward 3", "Ward 7", "Theatres", "Pharmacy", "Outpatients", "Kitchens"),
        bands=("Routine", "Planned", "Prompt", "Urgent", "Immediate", "Critical"),
        cuts=(2, 4, 6, 8, 11)),
    "campus": dict(
        org="Wexcombe University", title="facilities job queue", entity="job",
        idpfx="JB", idstart=1207, fmt="tsv", manual="Facilities Service Standard",
        cols=("job_ref", "hazard", "criticality", "occupants_affected", "days_waiting",
              "repeat", "trade", "building"),
        trades=("Electrical", "Roofing", "Lifts", "Plumbing", "Security"),
        places=("Library", "Chemistry", "Sports Hall", "Senate House", "Halls B"),
        bands=("Backlog", "Scheduled", "Soon", "Priority", "Pressing", "Crisis"),
        cuts=(1, 3, 5, 8, 10)),
    "depot": dict(
        org="Carrow Rail Depot", title="defect work list", entity="defect",
        idpfx="DF", idstart=7731, fmt="pipe", manual="Depot Engineering Instruction",
        cols=("defect_no", "hazard_grade", "asset_class", "passengers_affected",
              "days_logged", "recurring", "discipline", "road"),
        trades=("Traction", "Brakes", "Doors", "Signalling", "Bogies"),
        places=("Road 1", "Road 2", "Road 4", "Wash plant", "Shed 5"),
        bands=("Monitor", "Programme", "Expedite", "Escalate", "Withdraw", "Stop"),
        cuts=(2, 5, 7, 9, 12)),
    "housing": dict(
        org="Fenwick Homes", title="repairs register", entity="repair",
        idpfx="RP", idstart=3302, fmt="jsonl", manual="Responsive Repairs Code",
        cols=("repair_id", "hazard", "criticality", "residents_affected", "days_open",
              "repeat_visit", "trade", "estate"),
        trades=("Gas", "Damp", "Electrics", "Windows", "Drainage"),
        places=("Ashby Court", "Mill Lane", "Kestrel House", "Orchard Rise", "Tanners"),
        bands=("Minimal", "Standard", "Raised", "Severe", "Emergency", "Evacuate"),
        cuts=(1, 4, 6, 9, 11)),
    "airport": dict(
        org="Holt Regional Airport", title="maintenance request log", entity="request",
        idpfx="MR", idstart=5140, fmt="fixed", manual="Airside Maintenance Manual",
        cols=("request_no", "hazard_grade", "system_criticality", "passengers_affected",
              "days_open", "repeat_fault", "trade", "zone"),
        trades=("Baggage", "Lighting", "Jet bridge", "Fuel", "Fire"),
        places=("Stand 4", "Pier B", "Check-in", "Apron", "Gate 12"),
        bands=("Deferred", "Queued", "Elevated", "Hurried", "Blocking", "Grounding"),
        cuts=(2, 4, 7, 9, 12)),
    "schools": dict(
        org="Brackley Schools Trust", title="site issue log", entity="issue",
        idpfx="SI", idstart=2618, fmt="yaml", manual="Estates Handbook",
        cols=("issue_ref", "hazard", "asset_criticality", "pupils_affected",
              "days_reported", "repeat_issue", "trade", "school"),
        trades=("Heating", "Roofing", "Electrical", "Glazing", "Grounds"),
        places=("Ashcombe", "Brook Primary", "Hillside", "St Anne's", "Vale Academy"),
        bands=("Idle", "Termly", "Monthly", "Weekly", "Daily", "Closure"),
        cuts=(1, 3, 6, 8, 11)),
    "marina": dict(
        org="Pellow Marina", title="berth defect list", entity="defect",
        idpfx="BD", idstart=9025, fmt="ini", manual="Harbour Works Procedure",
        cols=("defect_ref", "hazard_grade", "asset_criticality", "berth_users_affected",
              "days_open", "repeat_defect", "trade", "pontoon"),
        trades=("Pontoon", "Electrical", "Water", "Fuel dock", "Security"),
        places=("A pontoon", "B pontoon", "C pontoon", "Fuel berth", "Visitor"),
        bands=("Watch", "Logged", "Attend", "Hasten", "Isolate", "Liftout"),
        cuts=(2, 5, 7, 10, 12)),
    "museum": dict(
        org="Garrick Museum", title="gallery fault register", entity="fault",
        idpfx="GF", idstart=6604, fmt="semicsv", manual="Collections Care Standard",
        cols=("fault_no", "hazard", "criticality", "visitors_affected", "days_open",
              "repeat", "trade", "gallery"),
        trades=("Climate", "Lighting", "Security", "Lifts", "Fabric"),
        places=("Gallery 2", "Gallery 9", "Atrium", "Print Room", "Stores"),
        bands=("Noted", "Diary", "Action", "Rush", "Closeroom", "Shutdown"),
        cuts=(1, 4, 6, 9, 11)),
}

POOL_A = ("estates", "campus", "depot", "housing")
POOL_B = ("airport", "schools", "marina", "museum")


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreRow:
    row_id: str
    hazard: int
    criticality: int
    people: int
    days: int
    repeat: bool
    trade: str
    place: str


@dataclass(frozen=True)
class ScoreTable:
    surface: str
    rows: tuple[ScoreRow, ...]
    body: str

    @property
    def dom(self) -> dict[str, Any]:
        return SURFACES[self.surface]


def exposure_points(row: ScoreRow, basis: str) -> int:
    people, days = row.people // 50, row.days // 7
    if basis == "people50":
        return people
    if basis == "days7":
        return days
    return max(people, days)


def band_index(score: int, cuts: tuple[int, ...]) -> int:
    """The band a score falls in: the first band runs from 0 to the first cut."""
    return sum(1 for cut in cuts if score > cut)


def row_answer(
    row: ScoreRow, dom: Mapping[str, Any], convention: Mapping[str, str]
) -> str:
    """One order's band under one rule."""
    score = HAZARD[convention["hazard_points"]][row.hazard]
    score += CRITICALITY[convention["criticality_points"]][row.criticality]
    score += exposure_points(row, convention["exposure_basis"])
    bands = dom["bands"]
    if row.repeat and convention["repeat_rule"] == "plus2points":
        score += 2
    index = band_index(score, dom["cuts"])
    if row.repeat and convention["repeat_rule"] == "oneband_up":
        index = min(index + 1, len(bands) - 1)
    return bands[index]


def key_for(table: ScoreTable, convention: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row_answer(row, table.dom, convention) for row in table.rows)


ALL_CONVENTIONS: tuple[dict[str, str], ...] = tuple(conventions(AXES))
PAIRS = single_axis_pairs(AXES)
EVERY_PAIR, PAIR_NEED = all_pairs(AXES, NEED)


# --------------------------------------------------------------------------
# building a table that separates every rule
# --------------------------------------------------------------------------


def _candidate(rng: random.Random, dom: Mapping[str, Any]) -> ScoreRow:
    return ScoreRow(
        row_id="",
        hazard=rng.randrange(3),
        criticality=rng.randrange(3),
        people=rng.randint(0, MAX_PEOPLE),
        days=rng.randint(0, MAX_DAYS),
        repeat=rng.random() < 0.4,
        trade=rng.choice(dom["trades"]),
        place=rng.choice(dom["places"]),
    )


def _answers(rows: list[ScoreRow], dom: Mapping[str, Any]) -> np.ndarray:
    """(rows, rules) of band indices, for the search."""
    bands = {band: n for n, band in enumerate(dom["bands"])}
    return np.array(
        [[bands[row_answer(row, dom, c)] for c in ALL_CONVENTIONS] for row in rows],
        dtype=np.int64,
    )


def render_body(table_rows: tuple[ScoreRow, ...], dom: Mapping[str, Any]) -> str:
    records = [
        (
            row.row_id,
            HAZARD_WORDS[row.hazard],
            CRITICALITY_WORDS[row.criticality],
            str(row.people),
            str(row.days),
            "yes" if row.repeat else "no",
            row.trade,
            row.place,
        )
        for row in table_rows
    ]
    return render_records(dom["cols"], records, dom["fmt"])


def build_rows(rng: random.Random, dom: Mapping[str, Any]) -> tuple[ScoreRow, ...] | None:
    """Twenty four orders chosen from random candidates to separate every rule."""
    pool = [_candidate(rng, dom) for _ in range(CANDIDATES)]
    answers = _answers(pool, dom)
    jitter = [rng.random() for _ in pool]
    chosen = greedy_rows(answers, EVERY_PAIR, PAIR_NEED, ROWS, jitter)
    chosen = rebalance_rows(
        answers, chosen, EVERY_PAIR, PAIR_NEED, rng, attempts=SWAPS, peak_weight=PEAK_WEIGHT
    )
    picked = [pool[i] for i in chosen]
    rng.shuffle(picked)
    rows = tuple(
        ScoreRow(
            row_id="%s-%d" % (dom["idpfx"], dom["idstart"] + n),
            hazard=row.hazard, criticality=row.criticality, people=row.people,
            days=row.days, repeat=row.repeat, trade=row.trade, place=row.place,
        )
        for n, row in enumerate(picked)
    )
    codes = _answers(list(rows), dom).T
    if not separation(codes, PAIRS, AXES).admits(NEED, max_crowding=MAX_CROWDING):
        return None
    return rows


# --------------------------------------------------------------------------
# the task text
# --------------------------------------------------------------------------


def band_table(dom: Mapping[str, Any]) -> str:
    cuts, bands = dom["cuts"], dom["bands"]
    left = ["0 to %d" % cuts[0]]
    for i in range(1, len(cuts)):
        left.append("%d to %d" % (cuts[i - 1] + 1, cuts[i]))
    left.append("%d and over" % (cuts[-1] + 1))
    width = max(len(x) for x in left + ["score"]) + 4
    lines = ["    %-*s%s" % (width, "score", "band")]
    lines += ["    %-*s%s" % (width, x, b) for x, b in zip(left, bands)]
    return "\n".join(lines)


TASK_TEMPLATE = """{org} {title}

You are the duty planner. Give every {entity} on the list below one priority band,
using the scorecard extract that follows.

SCORECARD EXTRACT ({manual})

  Every open {entity} is given a priority score, which is the sum of the points it
  earns on the house scorecard's four factors: its hazard grade (column {c_haz}), the
  criticality of the asset it concerns (column {c_crit}), its exposure, and whether it
  repeats an earlier fault (column {c_rep}). Exposure is scored either from the
  number of people affected (column {c_ppl}) or from the days the {entity} has been
  open (column {c_days}). The band is read off the score with this table:

{table}

  A hazard grade of none and an asset criticality of low earn no points. How many
  points each factor earns otherwise, how exposure is measured, and how a repeat
  fault is treated are set by the house scorecard, which this extract does not
  reproduce. The columns {c_trade} and {c_place} play no part in the score.

{scope}
{receipt}
LIST ({fmtnote})

{body}

File one line per {entity}: the id in column {c_id}, a comma, and the band, with no
header and no other text. File every {entity}, in the order they appear above.
"""


RECEIPT_SENTENCE: tuple[str, ...] = (
    "The receipt for this list reports the verdict and the correct band for selected",
    "orders only, and the lines for the other orders say nothing about whether they",
    "were right.",
)


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------

ORACLE_HEAD: tuple[str, ...] = (
    "HOUSE SCORECARD FOR THIS LIST AND THE NEXT ONE YOU FILE (it governs the",
    "list above and the next list you are given, whichever organisation it",
    "names; the list after that is scored under a scorecard of its own)",
    "",
)

ORACLE_TEMPLATE = OracleTemplate(
    head=ORACLE_HEAD,
    sentences={
        "hazard_points": "Hazard: {}.",
        "criticality_points": "Criticality: {}.",
        "exposure_basis": "Exposure: {}.",
        "repeat_rule": "Repeat faults: {}.",
    },
    phrases={
        "hazard_points": {
            "h1s3": "a minor hazard earns 1 point and a serious hazard earns 3 points",
            "h2s3": "a minor hazard earns 2 points and a serious hazard earns 3 points",
            "h1s4": "a minor hazard earns 1 point and a serious hazard earns 4 points",
            "h2s4": "a minor hazard earns 2 points and a serious hazard earns 4 points",
        },
        "criticality_points": {
            "m1x2": "a medium criticality asset earns 1 point and a high criticality "
                    "asset earns 2 points",
            "m1x3": "a medium criticality asset earns 1 point and a high criticality "
                    "asset earns 3 points",
            "m2x2": "a medium criticality asset earns 2 points and a high criticality "
                    "asset earns 2 points",
            "m0x3": "a medium criticality asset earns no points and a high criticality "
                    "asset earns 3 points",
        },
        "exposure_basis": {
            "people50": "one point for every full 50 people affected, and the days "
                        "open earn nothing",
            "days7": "one point for every full 7 days open, and the people affected "
                     "earn nothing",
            "larger_of": "the larger of one point for every full 50 people affected "
                         "and one point for every full 7 days open, never both",
        },
        "repeat_rule": {
            "plus2points": "a repeat fault earns 2 extra points, added to its score "
                           "before the band is read",
            "oneband_up": "a repeat fault earns no extra points and is placed one band "
                          "above the band its score gives, never above the last band",
        },
    },
)

ENVELOPE_SIZE = envelope_size(ROWS, oracle_allowance(ORACLE_TEMPLATE, AXES))


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class ScorecardGenerator(BandedRowsFamily):
    """The scorecard genre, as the generator protocol wants it."""

    name = "scorecard"
    genre = "priority banding under a hidden points schedule"
    SHAPE = SHAPE
    AXES = AXES
    ORACLE = ORACLE_TEMPLATE
    ENVELOPE_SIZE = ENVELOPE_SIZE
    ROWS = ROWS

    # ----- the instance -----

    def surface_for(self, ordinal: int, label: str) -> str:
        i = int(ordinal)
        if label.upper() == "A":
            return POOL_A[i % len(POOL_A)]
        return POOL_B[(i + i // len(POOL_B)) % len(POOL_B)]

    def surface_templates(self) -> tuple[str, ...]:
        return tuple(POOL_A) + tuple(POOL_B)

    def build_table(self, master: bytes, ordinal: int, label: str) -> ScoreTable:
        """A table from this side's own surface stream that separates every rule.

        The stream is keyed by the family, the ordinal, the side and the attempt, and
        nothing here can see the drawn rule: the table is the same whatever was drawn.
        """
        surface = self.surface_for(ordinal, label)
        dom = SURFACES[surface]
        stream = streams.SURFACE_A if label.upper() == "A" else streams.SURFACE_B
        for attempt in range(60):
            rng = streams.rng(master, stream, self.name, ordinal, attempt)
            rows = build_rows(rng, dom)
            if rows is None:
                continue
            return ScoreTable(surface=surface, rows=rows, body=render_body(rows, dom))
        raise ConstructionExhausted(
            "no scorecard table for %s at instance %d separated every rule" % (surface, ordinal)
        )

    def table_record(self, table: ScoreTable) -> dict[str, Any]:
        return {
            "surface": table.surface,
            "rows": [
                {
                    "row_id": row.row_id,
                    "hazard": row.hazard,
                    "criticality": row.criticality,
                    "people": row.people,
                    "days": row.days,
                    "repeat": row.repeat,
                    "trade": row.trade,
                    "place": row.place,
                }
                for row in table.rows
            ],
            "body": table.body,
        }

    def key_for(self, table: ScoreTable, convention: Mapping[str, str]) -> tuple[str, ...]:
        return key_for(table, convention)

    def row_identifiers(self, table: ScoreTable) -> tuple[str, ...]:
        return tuple(row.row_id for row in table.rows)

    def row_classes(self, table: ScoreTable) -> tuple[str, ...]:
        return tuple("repeat" if row.repeat else "first" for row in table.rows)

    def answer_ranks(self, table: ScoreTable) -> tuple[str, ...]:
        # The band table the task prints, in the order it prints it.
        return tuple(table.dom["bands"])

    # ----- the task text -----

    def describe(self, task: PublicTask) -> str:
        table: ScoreTable = task.table
        dom = table.dom
        cols = dom["cols"]
        return TASK_TEMPLATE.format(
            org=dom["org"], title=dom["title"], entity=dom["entity"], manual=dom["manual"],
            c_id=cols[0], c_haz=cols[1], c_crit=cols[2], c_ppl=cols[3], c_days=cols[4],
            c_rep=cols[5], c_trade=cols[6], c_place=cols[7],
            table=band_table(dom), scope=scope_sentence(task.label),
            receipt=shared_filing.receipt_sentence(self.RECEIPT_POLICY, RECEIPT_SENTENCE),
            fmtnote=FORMAT_NOTE[dom["fmt"]], body=table.body,
        )


def _check_vocabulary() -> None:
    """Every band fits the correction slot and no two surfaces share a band word."""
    seen: dict[str, str] = {}
    for surface, dom in SURFACES.items():
        for band in dom["bands"]:
            if len(band) > MAX_ANSWER:
                raise ValueError(f"band {band!r} of {surface} is wider than the slot")
            folded = shared_filing.fold(band)
            if folded in seen:
                raise ValueError(f"band {band!r} is used by {seen[folded]} and {surface}")
            seen[folded] = surface


_check_vocabulary()

GENERATOR = ScorecardGenerator()


__all__ = [
    "ALL_CONVENTIONS",
    "AXES",
    "CRITICALITY",
    "ENVELOPE_SIZE",
    "GENERATOR",
    "HAZARD",
    "ORACLE_TEMPLATE",
    "POOL_A",
    "POOL_B",
    "SHAPE",
    "SURFACES",
    "ScoreRow",
    "ScoreTable",
    "ScorecardGenerator",
    "band_index",
    "exposure_points",
    "key_for",
    "row_answer",
]
