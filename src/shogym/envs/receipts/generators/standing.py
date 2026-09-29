"""The standing genre: a member's standing after each infraction, under hidden house rules.

DESIGN NOTE

The task. A log of twenty four infractions by four or five members of a club, a library
or a building, in date order. Each infraction is of a named type, and the task prints how
many points each type carries and the table that turns a member's live points into one of
six standings. The clerk files, for every entry, the member's standing immediately after
it. What the log never says is which earlier infractions are still live, how a repeat is
counted, whether a first offence counts and whether a standing can fall back.

The rule space. Four hidden decisions, drawn uniformly and independently:

    lapse           4 options   an earlier infraction stays live for 60 or 90 days, only
                                a member's latest three infractions are live, or
                                everything stays live until a clean spell of 45 days
                                wipes it
    repeat_count    4 options   an infraction of the same type as the member's previous
                                one counts its points once, twice, or with one or two
                                points added
    first_offence   2 options   a member's first infraction counts in full, or is a
                                warning worth no points
    falling         2 options   a standing is read afresh from the live points every
                                time, or never falls below the member's last standing

That is 64 rules. Sibling tasks A and B are two different organisations, with their own
infraction types, standing names and cuts, scored under the same rule.

WHY INFERENCE FROM THE RECEIPT IS HARD. Every entry's corrected standing is a sum over the
member's own history, so the evidence for a decision is spread across rows: an entry is
explained only by the entries of the same member that came before it, weeks earlier and
many printed lines away. The four decisions confound one another. A standing that is
higher than expected can mean a longer lapse, a heavier repeat count, a first offence that
counted, or a standing that did not fall; a standing that is lower can mean the opposite
of any of them. Two lapse options count time and two count events, and on most entries
they agree, so they part only on the members whose gaps straddle a window or a clean
spell. Each correction therefore constrains the rule only jointly with the other entries
of that member, and the decisions separate only when several members' histories are read
together. `build_table` searches for a log in which every pair of the 64 rules gives a
different answer key (so the perfect reader of the full receipt scores 1.0 on the
sibling), every change of one decision moves at least three entries on each side, no rule
puts more than half the entries at one standing, and the standings are spread enough that
neither a reader with no receipt nor a relabel of the other sibling's answers earns much.

WHY AN AGENT COULD GET BETTER WITH EXPERIENCE. The decisions are the same four families
on every task. An agent that has seen a few receipts and rule statements learns to keep a
per-member ledger, to look first at members whose infractions straddle a lapse boundary,
at repeats of one type, and at the moments a standing would fall, and to treat the
decisions as a small grid to test rather than a story to guess. That is the reading the
perfect reader does by enumeration.

WHY THE FULL RECEIPT. The difficulty is meant to be the inference, not missing
information: every entry is reported, the perfect reader identifies the rule on every
admitted instance, and what an agent loses against it is what it fails to infer.

AUTHORSHIP. Designed and built in one session by Claude Opus 5.5 through Claude Code, as
one of three families built one after another; no helper agent did any part of it.
"""

from __future__ import annotations

import datetime as dt
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
    coincidence,
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
    Axis("lapse", ("live60days", "live90days", "latest3only", "clean45reset"),
         "which earlier infractions still count"),
    Axis("repeat_count", ("rep_x1", "rep_x2", "rep_plus1", "rep_plus2"),
         "how a repeat of the member's previous infraction type counts"),
    Axis("first_offence", ("first_full", "first_free"),
         "whether a member's first infraction counts"),
    Axis("falling", ("level_fresh", "level_kept"),
         "whether a standing can fall back"),
)

WINDOWS = {"live60days": 60, "live90days": 90}
#: Under the clean-spell option, a gap of this many days or more between two of a
#: member's infractions wipes everything before it.
CLEAN_SPELL = 45

ROWS = 24
#: A table search stops once it reaches no shortfall; this bounds how long it tries.
STEPS = 2500
NEED = 3
MAX_CROWDING = 0.5
#: The no-receipt level and the chance agreement a log may have, averaged over rules.
#: Above them a reader with no receipt, or a relabel of the other sibling's answers,
#: earns too much of the log for free.
MAX_NO_RECEIPT = 0.55
MAX_COINCIDENCE = 0.21

SHAPE = Shape(
    columns=(
        Column("entry id", "a surface prefix and a running number, in date order"),
        Column("date", "an ISO date; each member's infractions are days to months apart"),
        Column("member", "one of four or five member identifiers"),
        Column("infraction", "one of the surface's four named types"),
        Column("recorded by", "a staff name, never read by the rule"),
        Column("location", "a place name, never read by the rule"),
    ),
    rows=ROWS,
    case="one logged infraction awaiting the member's standing after it",
    note="the log starts clean: no member has an infraction before its first entry",
)

# ----- surfaces ---------------------------------------------------------------------

SURFACES: dict[str, dict[str, Any]] = {
    "toollib": dict(
        org="Harrowgate Tool Library", title="borrower infraction log", member="borrower",
        mpfx="BR", epfx="TL", estart=2041, fmt="csv", start=dt.date(2026, 1, 5),
        manual="Borrowing Rules", cols=("entry_id", "date", "borrower", "infraction",
                                         "logged_by", "branch"),
        types=(("late return", 1), ("unclean return", 1), ("missing part", 2),
               ("damaged tool", 3)),
        levels=("Clear", "Noted", "Warned", "Restricted", "Suspended", "Barred"), cuts=(1, 3, 5, 7, 9),
        staff=("J. Okoro", "M. Hale", "S. Varga"), places=("Eastside", "Riverside", "Hub")),
    "league": dict(
        org="Kingsmere Sunday League", title="disciplinary register", member="player",
        mpfx="PL", epfx="DR", estart=517, fmt="pipe", start=dt.date(2025, 9, 7),
        manual="League Disciplinary Code", cols=("case_no", "match_date", "player",
                                                  "offence", "referee", "ground"),
        types=(("dissent", 1), ("time wasting", 1), ("reckless tackle", 2),
               ("violent conduct", 3)),
        levels=("Eligible", "Booked", "Cautioned", "Benched", "Sidelined", "Banned"), cuts=(2, 3, 5, 7, 9),
        staff=("R. Adeyemi", "T. Lowe", "K. Brandt"), places=("Mill Park", "Cross Lane",
                                                              "Stanton Rec")),
    "cowork": dict(
        org="Lanyard Coworking", title="house rules log", member="member",
        mpfx="MB", epfx="HR", estart=3310, fmt="jsonl", start=dt.date(2026, 2, 2),
        manual="House Rules", cols=("log_id", "date", "member_id", "breach", "host",
                                    "floor"),
        types=(("noise complaint", 1), ("no-show booking", 1), ("unsigned guest", 2),
               ("property damage", 3)),
        levels=("Welcome", "Reminded", "Formal", "Probation", "Paused", "Revoked"), cuts=(1, 2, 3, 5, 7),
        staff=("A. Mensah", "L. Rossi", "P. Doyle"), places=("Ground", "Mezzanine", "Roof")),
    "allot": dict(
        org="Fernside Allotments", title="committee breach log", member="plot holder",
        mpfx="PH", epfx="CB", estart=145, fmt="tsv", start=dt.date(2026, 3, 1),
        manual="Tenancy Agreement", cols=("breach_no", "date", "plot_holder", "breach",
                                          "reported_by", "site"),
        types=(("untidy plot", 1), ("late fees", 1), ("unlit bonfire", 2),
               ("boundary damage", 3)),
        levels=("Standing", "Advised", "Notice", "Final", "Forfeit", "Evicted"), cuts=(1, 3, 4, 6, 8),
        staff=("G. Pryce", "H. Nakamura", "D. Flint"), places=("North", "South", "Orchard")),
    "gym": dict(
        org="Forge Fitness", title="conduct log", member="member",
        mpfx="GM", epfx="CL", estart=880, fmt="fixed", start=dt.date(2026, 1, 12),
        manual="Membership Terms", cols=("entry", "date", "member", "incident", "staff",
                                         "area"),
        types=(("kit left out", 1), ("late cancel", 1), ("misused machine", 2),
               ("abusive language", 3)),
        levels=("Active", "Flagged", "Warning", "Supervised", "Frozen", "Expelled"), cuts=(2, 3, 5, 7, 9),
        staff=("C. Obi", "V. Lind", "N. Shah"), places=("Weights", "Studio", "Pool")),
    "library": dict(
        org="Oakham Library", title="reader record", member="reader",
        mpfx="RD", epfx="LR", estart=6120, fmt="yaml", start=dt.date(2025, 10, 6),
        manual="Library Regulations", cols=("record_id", "date", "reader", "infraction",
                                            "librarian", "branch"),
        types=(("overdue item", 1), ("noise", 1), ("damaged book", 2), ("lost item", 3)),
        levels=("Normal", "Reminder", "Caution", "Limited", "Held", "Blocked"), cuts=(1, 3, 5, 7, 9),
        staff=("E. Ward", "F. Ansah", "B. Kerr"), places=("Central", "Westgate", "Mobile")),
    "flats": dict(
        org="Quayside Flats", title="tenancy breach log", member="tenant",
        mpfx="TN", epfx="TB", estart=402, fmt="ini", start=dt.date(2026, 4, 6),
        manual="Estate Management Policy", cols=("breach_ref", "date", "tenant", "breach",
                                                 "officer", "block"),
        types=(("bins left out", 1), ("noise", 1), ("unauthorised pet", 2),
               ("fire door propped", 3)),
        levels=("Compliant", "Letter", "Breach", "Ultimatum", "Court", "Possession"),
        cuts=(1, 3, 4, 6, 8),
        staff=("I. Patel", "O. Grant", "W. Chen"), places=("Block A", "Block B", "Block C")),
    "rowing": dict(
        org="Riverside Rowing Club", title="boathouse log", member="rower",
        mpfx="RW", epfx="BH", estart=77, fmt="semicsv", start=dt.date(2026, 3, 16),
        manual="Boathouse Rules", cols=("log_no", "date", "rower", "offence", "captain",
                                        "bay"),
        types=(("boat back late", 1), ("wet kit left", 1), ("unsigned outing", 2),
               ("hull damage", 3)),
        levels=("Member", "Minuted", "Reprimand", "Grounded", "Beached", "Dismissed"),
        cuts=(1, 2, 3, 5, 7),
        staff=("Q. Hart", "U. Bello", "Y. Moss"), places=("Bay 1", "Bay 2", "Pontoon")),
}

POOL_A = ("toollib", "league", "cowork", "allot")
POOL_B = ("gym", "library", "flats", "rowing")


# --------------------------------------------------------------------------
# the table and the rule
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    row_id: str
    day: int
    member: str
    kind: int
    staff: str
    place: str


@dataclass(frozen=True)
class StandingTable:
    surface: str
    rows: tuple[Entry, ...]
    body: str

    @property
    def dom(self) -> dict[str, Any]:
        return SURFACES[self.surface]


def entry_points(kind: int, repeat: bool, first: bool, convention: Mapping[str, str],
                 points: tuple[int, ...]) -> int:
    """What one infraction carries under one rule."""
    if first and convention["first_offence"] == "first_free":
        return 0
    base = points[kind]
    if not repeat:
        return base
    rule = convention["repeat_count"]
    if rule == "rep_x2":
        return 2 * base
    if rule == "rep_plus1":
        return base + 1
    if rule == "rep_plus2":
        return base + 2
    return base


def level_of(total: int, cuts: tuple[int, ...]) -> int:
    return sum(1 for cut in cuts if total > cut)


def key_for(table: StandingTable, convention: Mapping[str, str]) -> tuple[str, ...]:
    """Each entry's standing just after it, entry by entry in printed order.

    The reference implementation, written the way the oracle states the rule. The
    table search uses `all_codes`, which computes every rule at once, and a test holds
    the two equal on every rule.
    """
    dom = table.dom
    points = tuple(p for _, p in dom["types"])
    history: dict[str, list[tuple[int, int, int]]] = {}
    last_level: dict[str, int] = {}
    out: list[str] = []
    for entry in table.rows:
        past = history.setdefault(entry.member, [])
        repeat = bool(past) and past[-1][1] == entry.kind
        carried = entry_points(entry.kind, repeat, not past, convention, points)
        past.append((entry.day, entry.kind, carried))
        lapse = convention["lapse"]
        if lapse == "latest3only":
            live = past[-3:]
        elif lapse == "clean45reset":
            start = 0
            for k in range(1, len(past)):
                if past[k][0] - past[k - 1][0] >= CLEAN_SPELL:
                    start = k
            live = past[start:]
        else:
            live = [item for item in past if entry.day - item[0] < WINDOWS[lapse]]
        level = level_of(sum(item[2] for item in live), dom["cuts"])
        if convention["falling"] == "level_kept":
            level = max(level, last_level.get(entry.member, 0))
        last_level[entry.member] = level
        out.append(dom["levels"][level])
    return tuple(out)


ALL_CONVENTIONS: tuple[dict[str, str], ...] = tuple(conventions(AXES))
PAIRS = single_axis_pairs(AXES)


def all_codes(
    days: np.ndarray, members: np.ndarray, kinds: np.ndarray, dom: Mapping[str, Any]
) -> np.ndarray:
    """(rules, entries) of standing indices for every rule at once, in `conventions` order.

    `days`, `members` and `kinds` are in printed order. The same arithmetic as
    `key_for`, vectorized over the rules so a table search can afford thousands of
    candidate logs.
    """
    n = len(days)
    points = np.array([p for _, p in dom["types"]], dtype=np.int64)[kinds]
    same = members[:, None] == members[None, :]
    earlier = np.arange(n)[None, :] <= np.arange(n)[:, None]
    mine = same & earlier
    first = np.array([not (mine[i, :i]).any() for i in range(n)])
    previous = np.full(n, -1)
    rank = np.zeros(n, dtype=np.int64)
    spell = np.zeros(n, dtype=np.int64)
    for i in range(n):
        before = np.nonzero(mine[i, :i])[0]
        if before.size:
            previous[i] = before[-1]
            gap_before = days[i] - days[before[-1]]
            spell[i] = rank[before[-1]] + 1 if gap_before >= CLEAN_SPELL else spell[before[-1]]
        rank[i] = before.size
    repeat = np.array([previous[i] >= 0 and kinds[previous[i]] == kinds[i] for i in range(n)])
    gap = days[:, None] - days[None, :]
    live = {
        name: mine & (gap < window) for name, window in WINDOWS.items()
    }
    live["latest3only"] = mine & (rank[:, None] - rank[None, :] < 3)
    live["clean45reset"] = mine & (rank[None, :] >= spell[:, None])
    carried = {}
    for rep in AXES[1].options:
        extra = {
            "rep_x1": points,
            "rep_x2": 2 * points,
            "rep_plus1": points + 1,
            "rep_plus2": points + 2,
        }[rep]
        base = np.where(repeat, extra, points)
        for first_rule in AXES[2].options:
            carried[(rep, first_rule)] = (
                np.where(first, 0, base) if first_rule == "first_free" else base
            )
    cuts = np.array(dom["cuts"], dtype=np.int64)
    member_rows = [np.nonzero(members == m)[0] for m in np.unique(members)]
    out = []
    for lapse in AXES[0].options:
        for rep in AXES[1].options:
            for first_rule in AXES[2].options:
                totals = live[lapse].astype(np.int64) @ carried[(rep, first_rule)]
                levels = (totals[:, None] > cuts[None, :]).sum(axis=1)
                for falling in AXES[3].options:
                    if falling == "level_kept":
                        kept = levels.copy()
                        for rows in member_rows:
                            kept[rows] = np.maximum.accumulate(levels[rows])
                        out.append(kept)
                    else:
                        out.append(levels)
    return np.array(out, dtype=np.int64)


# --------------------------------------------------------------------------
# building a log that separates every rule
# --------------------------------------------------------------------------

#: The gaps between one member's infractions are drawn from these ranges in days, so
#: that some fall inside and some outside each lapse window.
GAPS = ((4, 22), (23, 40), (41, 70), (71, 110))
#: Members join the log at staggered days, so first infractions, and the low standings
#: that come with them, are spread through the log rather than bunched at its head. A
#: log whose standings climb from top to bottom is a log whose answers a reader can copy
#: from the other sibling by position alone.
START_SPREAD = 150


def _layout(rng: random.Random) -> tuple[list[int], list[int], list[int]]:
    """A random log: per event its day, its member index and its type, unsorted."""
    count = rng.choice((4, 5))
    sizes = [ROWS // count] * count
    for i in range(ROWS - sum(sizes)):
        sizes[i] += 1
    days: list[int] = []
    members: list[int] = []
    kinds: list[int] = []
    for member, size in enumerate(sizes):
        day = rng.randint(0, START_SPREAD)
        for k in range(size):
            if k:
                low, high = rng.choice(GAPS)
                day += rng.randint(low, high)
            days.append(day)
            members.append(member)
            kinds.append(rng.randrange(4))
    return days, members, kinds


def _ordered(days: list[int], members: list[int], kinds: list[int]) -> list[int]:
    return sorted(range(len(days)), key=lambda i: (days[i], members[i]))


def _codes(days: list[int], members: list[int], kinds: list[int], dom: Mapping[str, Any]):
    order = _ordered(days, members, kinds)
    return all_codes(
        np.array([days[i] for i in order]),
        np.array([members[i] for i in order]),
        np.array([kinds[i] for i in order]),
        dom,
    )


def _mutate(
    rng: random.Random, days: list[int], members: list[int], kinds: list[int]
) -> tuple[list[int], list[int], list[int]]:
    days, kinds = list(days), list(kinds)
    i = rng.randrange(len(days))
    if rng.random() < 0.5:
        kinds[i] = rng.randrange(4)
        return days, members, kinds
    mine = sorted((d, j) for j, d in enumerate(days) if members[j] == members[i])
    position = [j for _, j in mine].index(i)
    low = mine[position - 1][0] + 1 if position else 0
    high = mine[position + 1][0] - 1 if position + 1 < len(mine) else days[i] + 60
    if high >= low:
        days[i] = rng.randint(low, high)
    return days, members, kinds


def _objective(codes: np.ndarray) -> float:
    return (
        shortfall(codes, PAIRS, NEED, MAX_CROWDING)
        + 20.0 * max(no_receipt_level(codes) - MAX_NO_RECEIPT, 0.0)
        + 50.0 * max(coincidence(codes) - MAX_COINCIDENCE, 0.0)
    )


def build_log(rng: random.Random, dom: Mapping[str, Any]):
    """A log whose every rule gives a distinct key and every decision moves three entries."""
    days, members, kinds = _layout(rng)
    best = _objective(_codes(days, members, kinds, dom))
    for _ in range(STEPS):
        if best == 0.0:
            break
        trial = _mutate(rng, days, members, kinds)
        value = _objective(_codes(*trial, dom))
        if value <= best:
            days, members, kinds = trial
            best = value
    if best != 0.0:
        return None
    return days, members, kinds


def _date(dom: Mapping[str, Any], day: int) -> str:
    return (dom["start"] + dt.timedelta(days=int(day))).isoformat()


def render_body(rows: tuple[Entry, ...], dom: Mapping[str, Any]) -> str:
    names = [name for name, _ in dom["types"]]
    records = [
        (row.row_id, _date(dom, row.day), row.member, names[row.kind], row.staff, row.place)
        for row in rows
    ]
    return render_records(dom["cols"], records, dom["fmt"])


# --------------------------------------------------------------------------
# the task text
# --------------------------------------------------------------------------


def points_table(dom: Mapping[str, Any]) -> str:
    width = max(len(name) for name, _ in dom["types"]) + 4
    lines = ["    %-*s%s" % (width, "infraction", "points")]
    lines += ["    %-*s%d" % (width, name, points) for name, points in dom["types"]]
    return "\n".join(lines)


def level_table(dom: Mapping[str, Any]) -> str:
    cuts, levels = dom["cuts"], dom["levels"]
    left = ["0" if cuts[0] == 0 else "0 to %d" % cuts[0]]
    for i in range(1, len(cuts)):
        low, high = cuts[i - 1] + 1, cuts[i]
        left.append("%d" % low if low == high else "%d to %d" % (low, high))
    left.append("%d and over" % (cuts[-1] + 1))
    width = max(len(x) for x in left + ["live points"]) + 4
    lines = ["    %-*s%s" % (width, "live points", "standing")]
    lines += ["    %-*s%s" % (width, x, b) for x, b in zip(left, levels)]
    return "\n".join(lines)


TASK_TEMPLATE = """{org} {title}

You are the membership clerk. For every entry in the log below, give the standing the
{member} concerned holds immediately after that entry, using the rules extract that
follows.

RULES EXTRACT ({manual})

  Each infraction carries points by its type:

{points}

{read}

{levels}

{rest}

{scope}
{receipt}
LOG ({fmtnote}; entries in date order)

{body}

File one line per entry: the id in column {c_id}, a comma, and the standing, with no
header and no other text. File every entry, in the order the entries appear above.
"""

#: The extract's paragraphs, filled to width once the surface's words are in.
READ_PARAGRAPH = (
    "A {member}'s standing is read from the points that are live at the moment, using "
    "this table:"
)
REST_PARAGRAPH = (
    "The log starts clean: no {member} has an infraction before their first entry "
    "below, and every {member} starts at the first standing. Which earlier infractions "
    "are still live, how an infraction of the same type as the {member}'s previous one "
    "is counted, whether a {member}'s first infraction counts, and whether a standing "
    "can fall back once the live points drop are set by the house rules, which this "
    "extract does not reproduce. The columns {c_staff} and {c_place} play no part."
)

RECEIPT_SENTENCE: tuple[str, ...] = (
    "The receipt for this log reports the verdict and the correct standing for selected",
    "entries only, and the lines for the other entries say nothing about whether they",
    "were right.",
)


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------

ORACLE_HEAD: tuple[str, ...] = (
    "HOUSE RULES FOR THIS LOG AND THE NEXT ONE YOU FILE (they govern the log",
    "above and the next log you are given, whichever organisation it names;",
    "the log after that is scored under house rules of its own)",
    "",
)

ORACLE_TEMPLATE = OracleTemplate(
    head=ORACLE_HEAD,
    sentences={
        "lapse": "Live points: {}.",
        "repeat_count": "Repeats: {}.",
        "first_offence": "First infractions: {}.",
        "falling": "Falling back: {}.",
    },
    phrases={
        "lapse": {
            "live60days": "an infraction stays live for 60 days, so at an entry the live "
                          "infractions are the member's infractions dated fewer than 60 "
                          "days before it, that entry included",
            "live90days": "an infraction stays live for 90 days, so at an entry the live "
                          "infractions are the member's infractions dated fewer than 90 "
                          "days before it, that entry included",
            "latest3only": "only the member's latest three infractions are live, that "
                           "entry and the two before it, however old they are",
            "clean45reset": "an infraction stays live until the member goes 45 days or "
                            "more without an infraction: a gap of 45 days or more between "
                            "two of the member's infractions wipes every infraction before "
                            "it, and until such a gap everything since the last one is live",
        },
        "repeat_count": {
            "rep_x1": "an infraction of the same type as the member's previous infraction "
                      "carries its usual points and nothing more",
            "rep_x2": "an infraction of the same type as the member's previous infraction "
                      "carries twice its usual points",
            "rep_plus1": "an infraction of the same type as the member's previous "
                         "infraction carries its usual points plus 1",
            "rep_plus2": "an infraction of the same type as the member's previous "
                         "infraction carries its usual points plus 2",
        },
        "first_offence": {
            "first_full": "a member's first infraction in the log carries its points like "
                          "any other",
            "first_free": "a member's first infraction in the log is a warning and carries "
                          "no points, now or later",
        },
        "falling": {
            "level_fresh": "the standing is read afresh from the live points at every "
                           "entry, so it falls when points stop being live",
            "level_kept": "a standing never falls: at every entry the member keeps the "
                          "higher of the standing the live points give and the standing "
                          "they held after their previous entry",
        },
    },
)

ENVELOPE_SIZE = envelope_size(ROWS, oracle_allowance(ORACLE_TEMPLATE, AXES))


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class StandingGenerator(BandedRowsFamily):
    """The standing genre, as the generator protocol wants it."""

    name = "standing"
    genre = "stateful standing under hidden house rules"
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

    def build_table(self, master: bytes, ordinal: int, label: str) -> StandingTable:
        """A log from this side's own surface stream that separates every rule."""
        surface = self.surface_for(ordinal, label)
        dom = SURFACES[surface]
        stream = streams.SURFACE_A if label.upper() == "A" else streams.SURFACE_B
        for attempt in range(40):
            rng = streams.rng(master, stream, self.name, ordinal, attempt)
            found = build_log(rng, dom)
            if found is None:
                continue
            days, members, kinds = found
            order = _ordered(days, members, kinds)
            ids = rng.sample(range(100, 1000), 6)
            rows = tuple(
                Entry(
                    row_id="%s-%d" % (dom["epfx"], dom["estart"] + n),
                    day=days[i],
                    member="%s-%d" % (dom["mpfx"], ids[members[i]]),
                    kind=kinds[i],
                    staff=rng.choice(dom["staff"]),
                    place=rng.choice(dom["places"]),
                )
                for n, i in enumerate(order)
            )
            return StandingTable(surface=surface, rows=rows, body=render_body(rows, dom))
        raise ConstructionExhausted(
            "no standing log for %s at instance %d separated every rule" % (surface, ordinal)
        )

    def table_record(self, table: StandingTable) -> dict[str, Any]:
        return {
            "surface": table.surface,
            "rows": [
                {
                    "row_id": row.row_id,
                    "day": row.day,
                    "member": row.member,
                    "kind": row.kind,
                    "staff": row.staff,
                    "place": row.place,
                }
                for row in table.rows
            ],
            "body": table.body,
        }

    def key_for(self, table: StandingTable, convention: Mapping[str, str]) -> tuple[str, ...]:
        return key_for(table, convention)

    def row_identifiers(self, table: StandingTable) -> tuple[str, ...]:
        return tuple(row.row_id for row in table.rows)

    def row_classes(self, table: StandingTable) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for row in table.rows:
            out.append("later" if row.member in seen else "first")
            seen.add(row.member)
        return tuple(out)

    def answer_ranks(self, table: StandingTable) -> tuple[str, ...]:
        return tuple(table.dom["levels"])

    def describe(self, task: PublicTask) -> str:
        table: StandingTable = task.table
        dom = table.dom
        cols = dom["cols"]
        words = dict(member=dom["member"], c_staff=cols[4], c_place=cols[5])
        return TASK_TEMPLATE.format(
            org=dom["org"], title=dom["title"], member=dom["member"], manual=dom["manual"],
            points=points_table(dom), levels=level_table(dom),
            read=prose(READ_PARAGRAPH.format(**words)),
            rest=prose(REST_PARAGRAPH.format(**words)), c_id=cols[0],
            scope=scope_sentence(task.label),
            receipt=shared_filing.receipt_sentence(self.RECEIPT_POLICY, RECEIPT_SENTENCE),
            fmtnote=FORMAT_NOTE[dom["fmt"]], body=table.body,
        )


def _check_vocabulary() -> None:
    seen: dict[str, str] = {}
    for surface, dom in SURFACES.items():
        for level in dom["levels"]:
            if len(level) > MAX_ANSWER:
                raise ValueError(f"standing {level!r} of {surface} is wider than the slot")
            folded = shared_filing.fold(level)
            if folded in seen:
                raise ValueError(f"standing {level!r} is used by {seen[folded]} and {surface}")
            seen[folded] = surface


_check_vocabulary()

GENERATOR = StandingGenerator()


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
    "Entry",
    "StandingGenerator",
    "StandingTable",
    "all_codes",
    "entry_points",
    "key_for",
    "level_of",
]
