"""The ledger genre: clerical date counting under a hidden house convention.

A table of records, each with three dates and some with none, and a policy extract
that tells the clerk to band every record by "days elapsed as at the reference
date". The extract is deliberately incomplete. Four decisions it never makes are
the hidden convention:

    anchor    3 options   which of the three dates the count runs from
    basis     3 options   calendar days, business days, or business days with the
                          listed public holidays taken out
    boundary  2 options   which band a count landing exactly on a printed figure
                          takes
    missing   4 options   what a record with no dates entered is given

That is 72 conventions, drawn uniformly and independently. Sibling tasks A and B
are two entirely different organisations, formats and band vocabularies, scored
under the same drawn convention.

WHY THE RECEIPT CAN CARRY SOMETHING. The scored column holds a value the rule
COMPUTES, a band, not the option the agent chose. Two anchor options that put a
record in the same band are indistinguishable on that record and distinguishable on
another, so the verdicts across records cut the option set into more than two pieces
and the agent can narrow the rule rather than only learn that it was wrong.
`build_table` searches for a table where that actually happens: it seeds records
whose counts land on the band cuts under several (anchor, basis) pairs, and rejects
a draw where varying an axis moves too few records.

WHAT THIS PORT CHANGED FROM THE INSTRUMENT'S GENERATOR.

The receipt carries no by-facet block. The instrument's graded receipt closed with a
per-facet error count, and its facets are the four axes under other names, so
printing them would have the receipt state its own interpretation.

The renderers return a structure, not text. Layout, widths, padding and the
envelope belong to the shared serializer, so the bytes are a function of the
structure and the structure is a faithful thing to gate.

Every stream is keyed and domain-separated. The instrument seeded one generator
from an integer and reused it; here surface A, surface B, the convention and the
filler are four independent streams under one controller-side key, so nothing about
one is recoverable from another.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from shogym.envs.receipts import streams
from shogym.envs.receipts.protocol import (
    Axis,
    Column,
    Filing,
    ROW_ADDITIVE_EQUAL_WEIGHT,
    NoFiling,
    PublicTask,
    RowOutcome,
    SealedSubmission,
    Shape,
    Task,
)
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.oracle import parse as parse_oracle_cell
from shogym.envs.receipts.oracle import render as render_oracle_cell
from shogym.envs.receipts.render import graded_receipt, placebo_receipt
from shogym.receipts import ROW_LABEL
from shogym.envs.receipts.receipt_ast import (
    Envelope,
    ReceiptAST,
    SlotSpec,
    envelope_size_for,
)

ROLES = ("event", "intake", "last_action")
PENDING_TOKEN = "PENDING"
#: What the receipt prints for a row the agent filed with an empty value. It is
#: deliberately not the word `blank`: that is the name of an option on the `missing`
#: axis, and a display token that collides with an option name is a word a reader
#: cannot tell from an interpretation of the rule.
BLANK_TOKEN = "(empty)"
#: What it prints for a row the agent filed nothing for. Two different acts, so
#: two different tokens: the receipt grades what the filing did.
UNFILED_TOKEN = "(none)"

AXES: tuple[Axis, ...] = (
    Axis("anchor", ("event", "intake", "last_action"),
         "which date column the count runs from"),
    Axis("basis", ("calendar", "business", "business_minus_holidays"),
         "which days are counted"),
    Axis("boundary", ("lower", "upper"),
         "a count landing exactly on a printed figure"),
    Axis("missing", ("lowest", "highest", "blank", "pending"),
         "a record with no dates entered"),
)

SHAPE = Shape(
    columns=(
        Column("record id", "a domain prefix and a running number from the domain's start"),
        Column("event date", "a day drawn inside the schedule window"),
        Column("intake date", "the event date plus nought to seven days"),
        Column("last action date", "the intake date plus nought to ten days"),
        Column("handler", "drawn from the domain's five names"),
        Column("amount", "a whole number drawn between 120 and 9800"),
    ),
    rows=24,
    case="one open record awaiting a band",
    note="20 rows carry all three dates and 4 carry none",
)

N_DATED = 20
N_MISSING = 4
SPAN = 46
N_DESIGNED = 8

# ----- the registered envelope constants -------------------------------------
# Every one of these is a family maximum or a fixed width. None is read off a
# particular draw, which is what makes the envelope size convention-independent.

IDENTIFIER_WIDTH = 12
OBSERVED_WIDTH = 16
VERDICT_WIDTH = 4
CORRECTION_WIDTH = 12
#: The two registered slots and the closed grammar each may print. A verdict is one
#: of two literals and nothing else; a correction is empty or the row's actual answer.
#: Anything outside these is refused at admission, whatever it looks like.
VERDICT_TOKENS = ("PASS", "FAIL")
SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec("verdict", VERDICT_WIDTH, vocabulary=VERDICT_TOKENS),
    SlotSpec(
        "correction", CORRECTION_WIDTH,
        vocabulary=(BLANK_TOKEN,), allows_answers=True, allows_empty=True,
    ),
)
#: One title per slot, shared by the graded and placebo cells. Neutral on purpose:
#: a header reading "expected" would tell a placebo reader what the column it
#: cannot see would have meant, and it would move bytes the slots do not cover.
COLUMN_TITLES = ("mark", "note")
#: The alphabet the placebo's neutral tokens and the padding are drawn from. Digits
#: only, and that is not decoration: with letters in it a neutral token eventually
#: spells a verdict, and a placebo that prints PASS on a row has delivered a
#: positive grade in the arm that is supposed to carry none. The `neutral` check
#: asserts the realized tokens rather than trusting the alphabet.
FILLER_ALPHABET = "0123456789"
ORACLE_BODY_ALLOWANCE = 1100
ENVELOPE_SIZE = envelope_size_for(
    max_rows=SHAPE.rows,
    identifier_width=IDENTIFIER_WIDTH,
    observed_width=OBSERVED_WIDTH,
    slots=SLOTS,
    body_allowance=ORACLE_BODY_ALLOWANCE,
)

# ----- domains: the surface data, and how its values are invented -------------

DOMAINS: dict[str, dict[str, Any]] = {
    "claims": dict(
        org="Meridian Mutual", title="claims triage schedule", entity="claim",
        idpfx="CLM", idstart=1043, fmt="csv",
        cols={"event": "date_of_loss", "intake": "date_received", "last_action": "last_contact"},
        extra=[("adjuster", ["R. Okafor", "T. Lindqvist", "M. Serrano", "D. Whitlock",
                             "P. Ferreira"]),
               ("reserve_gbp", None)],
        bands=["Routine", "Standard", "Priority", "Urgent", "Critical"],
        cuts=[4, 9, 16, 25], refdate=dt.date(2026, 6, 12),
        unit="office", manual="Claims Handling Manual",
        holnames=["Spring bank holiday", "Founders' Day", "Regional holiday"]),
    "permits": dict(
        org="Harlow County", title="building control permit backlog", entity="permit",
        idpfx="PMT", idstart=2210, fmt="pipe",
        cols={"event": "work_commenced", "intake": "application_filed",
              "last_action": "last_reviewed"},
        extra=[("officer", ["J. Abrahams", "S. Nkemelu", "L. Vantongeren", "C. Byrne",
                            "H. Salinas"]),
               ("valuation_gbp", None)],
        bands=["Green", "Amber", "Orange", "Red", "Black"],
        cuts=[3, 8, 14, 22], refdate=dt.date(2026, 5, 21),
        unit="registry", manual="Building Control Procedure",
        holnames=["May Day", "County Show holiday", "Charter Day"]),
    "warranty": dict(
        org="Kestrel Appliance", title="warranty ticket register", entity="ticket",
        idpfx="WT", idstart=5501, fmt="tsv",
        cols={"event": "fault_occurred", "intake": "logged_on",
              "last_action": "last_engineer_visit"},
        extra=[("technician", ["A. Prentice", "K. Osei", "V. Marchetti", "F. Duran",
                               "B. Halloway"]),
               ("model_code", None)],
        bands=["Watch", "Attend", "Expedite", "Escalate", "Recall"],
        cuts=[5, 11, 18, 28], refdate=dt.date(2026, 7, 9),
        unit="service desk", manual="Service Operations Handbook",
        holnames=["Summer holiday", "Works shutdown day", "Trade holiday"]),
    "grants": dict(
        org="Thornbury Trust", title="grant disbursement watchlist", entity="award",
        idpfx="GR", idstart=8802, fmt="jsonl",
        cols={"event": "project_start", "intake": "submitted_on",
              "last_action": "last_correspondence"},
        extra=[("programme_officer", ["N. Achterberg", "R. Iwuchukwu", "E. Solberg",
                                      "G. Mancini", "Y. Tadesse"]),
               ("award_gbp", None)],
        bands=["Nominal", "Monitor", "Review", "Intervene", "Suspend"],
        cuts=[6, 12, 20, 30], refdate=dt.date(2026, 4, 30),
        unit="grants office", manual="Disbursement Oversight Policy",
        holnames=["Easter Tuesday", "Trustees' holiday", "Spring holiday"]),
    "library": dict(
        org="Alderweir Libraries", title="recall queue", entity="loan",
        idpfx="LN", idstart=3374, fmt="fixed",
        cols={"event": "item_borrowed", "intake": "recall_raised",
              "last_action": "last_notice_sent"},
        extra=[("branch", ["Eastgate", "Marlow Road", "Cattermole", "Sixfields", "Priory"]),
               ("replacement_gbp", None)],
        bands=["Quiet", "Notice", "Chase", "Hold", "Bar"],
        cuts=[4, 10, 17, 26], refdate=dt.date(2026, 3, 19),
        unit="branch network", manual="Circulation Rules",
        holnames=["St Wendreda's Day", "Founders' holiday", "Spring closure"]),
    "fleet": dict(
        org="Draycott Haulage", title="vehicle defect log", entity="defect",
        idpfx="DF", idstart=4419, fmt="ini",
        cols={"event": "defect_arose", "intake": "entered_in_log",
              "last_action": "last_inspection"},
        extra=[("depot", ["Netherfield", "Cawsand", "Bewdley", "Longmoor", "Ardleigh"]),
               ("odometer_km", None)],
        bands=["Serviceable", "Caution", "Restricted", "Grounded", "Condemned"],
        cuts=[3, 7, 15, 24], refdate=dt.date(2026, 9, 4),
        unit="workshop", manual="Fleet Maintenance Standard",
        holnames=["August holiday", "Depot shutdown", "Harvest holiday"]),
    "tenancy": dict(
        org="Pelham Housing", title="repair request register", entity="request",
        idpfx="RQ", idstart=6620, fmt="yaml",
        cols={"event": "damage_noticed", "intake": "request_opened", "last_action": "last_visit"},
        extra=[("surveyor", ["I. Braithwaite", "O. Camara", "Z. Petrov", "W. Ndiaye",
                             "L. Trask"]),
               ("estimate_gbp", None)],
        bands=["Low", "Guided", "Firm", "Severe", "Statutory"],
        cuts=[5, 9, 19, 27], refdate=dt.date(2026, 2, 26),
        unit="repairs team", manual="Repairs Standard",
        holnames=["Winter holiday", "Borough holiday", "Founders' Day"]),
    "customs": dict(
        org="Varnhold Port", title="clearance backlog", entity="consignment",
        idpfx="CN", idstart=7715, fmt="semicsv",
        cols={"event": "goods_landed", "intake": "declaration_lodged",
              "last_action": "last_inspected"},
        extra=[("inspector", ["H. Rasmussen", "Q. Adeyemi", "T. Molnar", "S. Kavanagh",
                              "U. Bergstrom"]),
               ("declared_eur", None)],
        bands=["Clear", "Queue", "Hold", "Detain", "Seize"],
        cuts=[4, 8, 13, 21], refdate=dt.date(2026, 10, 15),
        unit="clearance hall", manual="Port Clearance Instruction",
        holnames=["Autumn holiday", "Harbour Day", "Trade fair holiday"]),
}

POOL_A = ("claims", "permits", "warranty", "grants")
POOL_B = ("library", "fleet", "tenancy", "customs")

FMT_NOTE = {
    "csv": "comma-separated",
    "tsv": "tab-separated",
    "semicsv": "semicolon-separated, quoted",
    "pipe": "pipe-delimited table",
    "fixed": "fixed-width columns",
    "jsonl": "one JSON object per line",
    "ini": "one bracketed block per record",
    "yaml": "a YAML list",
}


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerRow:
    row_id: str
    dates: dict[str, dt.date] | None


@dataclass(frozen=True)
class LedgerTable:
    domain: str
    rows: tuple[LedgerRow, ...]
    holidays: tuple[dt.date, ...]
    body: str

    @property
    def dom(self) -> dict[str, Any]:
        return DOMAINS[self.domain]


# --------------------------------------------------------------------------
# counting: the scoring function's arithmetic
# --------------------------------------------------------------------------


def daycount(anchor_date: dt.date, ref: dt.date, basis: str, holidays: Sequence[dt.date]) -> int:
    """Days after `anchor_date` up to and including `ref`, under the basis."""
    n = 0
    d = anchor_date + dt.timedelta(days=1)
    while d <= ref:
        if basis == "calendar":
            n += 1
        elif basis == "business":
            if d.weekday() < 5:
                n += 1
        else:
            if d.weekday() < 5 and d not in holidays:
                n += 1
        d += dt.timedelta(days=1)
    return n


def band_for(count: int, cuts: Sequence[int], bands: Sequence[str], boundary: str) -> str:
    if boundary == "lower":
        idx = sum(1 for c in cuts if count > c)
    else:
        idx = sum(1 for c in cuts if count >= c)
    return bands[idx]


def key_for(table: LedgerTable, convention: Mapping[str, str]) -> tuple[str, ...]:
    """The correct answer for every row under one convention: a band per row."""
    dom = table.dom
    out: list[str] = []
    for r in table.rows:
        if r.dates is None:
            m = convention["missing"]
            if m == "lowest":
                out.append(dom["bands"][0])
            elif m == "highest":
                out.append(dom["bands"][-1])
            elif m == "blank":
                out.append("")
            else:
                out.append(PENDING_TOKEN)
        else:
            n = daycount(
                r.dates[convention["anchor"]], dom["refdate"], convention["basis"],
                table.holidays,
            )
            out.append(band_for(n, dom["cuts"], dom["bands"], convention["boundary"]))
    return tuple(out)


ALL_CONVENTIONS: tuple[dict[str, str], ...] = tuple(
    {"anchor": a, "basis": b, "boundary": o, "missing": m}
    for a in AXES[0].options
    for b in AXES[1].options
    for o in AXES[2].options
    for m in AXES[3].options
)


# --------------------------------------------------------------------------
# building a table the readouts actually merge on
# --------------------------------------------------------------------------


def make_holidays(rng: random.Random, ref: dt.date, span: int) -> list[dt.date]:
    """One holiday near the reference date, one mid-window, one far back, so the
    business and business-minus-holidays options differ on most records."""
    hol: list[dt.date] = []
    for lo, hi in ((3, 9), (11, 22), (24, span - 4)):
        for _ in range(200):
            d = ref - dt.timedelta(days=rng.randint(lo, hi))
            if d.weekday() < 5 and d not in hol:
                hol.append(d)
                break
    return sorted(hol)


def place_dates_around(
    rng: random.Random, d: dt.date, ref: dt.date
) -> dict[str, dt.date] | None:
    """From the event date, the other two, keeping event <= intake <= last_action."""
    ev = d
    ik = ev + dt.timedelta(days=rng.randint(0, 7))
    la = ik + dt.timedelta(days=rng.randint(0, 10))
    if la >= ref or ev >= ref:
        return None
    return {"event": ev, "intake": ik, "last_action": la}


def build_rows(
    rng: random.Random, dom: dict[str, Any]
) -> tuple[list[LedgerRow], list[dt.date]]:
    ref = dom["refdate"]
    holidays = make_holidays(rng, ref, SPAN)
    anchor_opts, basis_opts = AXES[0].options, AXES[1].options
    counts: dict[tuple[str, dt.date], int] = {}
    for b in basis_opts:
        for k in range(1, SPAN + 1):
            d = ref - dt.timedelta(days=k)
            counts[(b, d)] = daycount(d, ref, b, holidays)
    settings = [(a, b) for a in anchor_opts for b in basis_opts]
    cuts = set(dom["cuts"])
    candidates: list[tuple[tuple[int, ...], dict[str, dt.date]]] = []
    for k in range(2, SPAN + 1):
        ev = ref - dt.timedelta(days=k)
        for gap_one in range(1, 8):
            ik = ev + dt.timedelta(days=gap_one)
            if ik >= ref:
                continue
            for gap_two in range(1, 11):
                la = ik + dt.timedelta(days=gap_two)
                if la >= ref:
                    continue
                trip = {"event": ev, "intake": ik, "last_action": la}
                vec = tuple(1 if counts[(b, trip[a])] in cuts else 0 for a, b in settings)
                if sum(vec):
                    candidates.append((vec, trip))
    rng.shuffle(candidates)
    cover = [0] * len(settings)
    chosen: list[dict[str, dt.date]] = []
    for _ in range(N_DESIGNED):
        best = None
        best_gain = -1.0
        for vec, trip in candidates:
            gain = sum(vec[j] * (3.0 / (1 + cover[j])) for j in range(len(settings)))
            if gain > best_gain:
                best_gain, best = gain, (vec, trip)
        if best is None:
            break
        vec, trip = best
        chosen.append(trip)
        cover = [cover[j] + vec[j] for j in range(len(settings))]
        candidates = [c for c in candidates if c[1] is not trip]

    dated: list[dict[str, dt.date] | None] = list(chosen)
    while len(dated) < N_DATED:
        ev = ref - dt.timedelta(days=rng.randint(2, SPAN))
        placed = place_dates_around(rng, ev, ref)
        if placed:
            dated.append(placed)
    entries: list[dict[str, dt.date] | None] = dated[:N_DATED] + [None] * N_MISSING
    rng.shuffle(entries)
    rows = [
        LedgerRow(row_id="%s-%d" % (dom["idpfx"], dom["idstart"] + i), dates=d)
        for i, d in enumerate(entries)
    ]
    return rows, holidays


def leverage(
    rows: Sequence[LedgerRow], domain: str, holidays: Sequence[dt.date]
) -> tuple[dict[str, tuple[int, float]], bool]:
    """How many rows move when one axis is varied and the rest are held.

    The minimum over conventions is what admits a table: an axis that moves one row
    on some convention cannot be narrowed from a receipt, and an axis that moves
    none is not in the task at all.
    """
    table = LedgerTable(domain=domain, rows=tuple(rows), holidays=tuple(holidays), body="")
    keys = {_combo(c): key_for(table, c) for c in ALL_CONVENTIONS}
    out: dict[str, tuple[int, float]] = {}
    for axis in AXES:
        moved: list[int] = []
        for c in ALL_CONVENTIONS:
            for alt in axis.options:
                if alt == c[axis.name]:
                    continue
                other = dict(c)
                other[axis.name] = alt
                first, second = keys[_combo(c)], keys[_combo(other)]
                moved.append(sum(1 for x, y in zip(first, second) if x != y))
        out[axis.name] = (min(moved), sum(moved) / len(moved))
    distinct = len({tuple(v) for v in keys.values()}) == len(keys)
    return out, distinct


def _combo(convention: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(convention[a.name] for a in AXES)


# --------------------------------------------------------------------------
# rendering the surface
# --------------------------------------------------------------------------


def _fmt_date(d: dt.date | None) -> str:
    return d.isoformat() if d else ""


def render_rows(rows: Sequence[LedgerRow], dom: dict[str, Any], rng: random.Random) -> str:
    cols, extra = dom["cols"], dom["extra"]
    names = [cols["event"], cols["intake"], cols["last_action"]]
    vals: list[dict[str, str]] = []
    for r in rows:
        rec = {"id": r.row_id}
        for role, nm in zip(ROLES, names):
            rec[nm] = _fmt_date(r.dates[role]) if r.dates else ""
        for nm, choices in extra:
            rec[nm] = str(rng.randint(120, 9800)) if choices is None else rng.choice(choices)
        vals.append(rec)
    idname = dom["entity"] + "_id"
    hdr = [idname] + names + [nm for nm, _ in extra]

    def cell(v: dict[str, str], h: str) -> str:
        return v["id"] if h == idname else v[h]

    fmt = dom["fmt"]
    if fmt == "csv":
        return "\n".join([",".join(hdr)] + [",".join(cell(v, h) for h in hdr) for v in vals])
    if fmt == "tsv":
        return "\n".join(["\t".join(hdr)] + ["\t".join(cell(v, h) for h in hdr) for v in vals])
    if fmt == "semicsv":
        return "\n".join(
            [";".join('"%s"' % h for h in hdr)]
            + [";".join('"%s"' % cell(v, h) for h in hdr) for v in vals]
        )
    if fmt == "pipe":
        w = [max(len(h), max((len(cell(v, h)) for v in vals), default=0)) for h in hdr]
        out = [
            "| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |",
            "|" + "|".join("-" * (w[i] + 2) for i in range(len(hdr))) + "|",
        ]
        for v in vals:
            out.append(
                "| " + " | ".join(cell(v, h).ljust(w[i]) for i, h in enumerate(hdr)) + " |"
            )
        return "\n".join(out)
    if fmt == "fixed":
        w = [max(len(h), max((len(cell(v, h)) for v in vals), default=0)) + 2 for h in hdr]
        out = ["".join(h.ljust(w[i]) for i, h in enumerate(hdr)).rstrip()]
        for v in vals:
            out.append("".join(cell(v, h).ljust(w[i]) for i, h in enumerate(hdr)).rstrip())
        return "\n".join(out)
    if fmt == "jsonl":
        return "\n".join(json.dumps({h: (cell(v, h) or None) for h in hdr}) for v in vals)
    if fmt == "ini":
        out = []
        for v in vals:
            out.append("[%s]" % v["id"])
            out.extend("%s = %s" % (h, v[h]) for h in hdr[1:])
            out.append("")
        return "\n".join(out).rstrip()
    if fmt == "yaml":
        out = []
        for v in vals:
            out.append("- %s: %s" % (idname, v["id"]))
            out.extend("  %s: %s" % (h, v[h] if v[h] != "" else "~") for h in hdr[1:])
        return "\n".join(out)
    raise ValueError(fmt)


def band_table(dom: dict[str, Any]) -> str:
    cuts, bands = dom["cuts"], dom["bands"]
    left = ["0 to %d" % cuts[0]]
    for i in range(1, 4):
        left.append("%d to %d" % (cuts[i - 1], cuts[i]))
    left.append("%d and over" % cuts[3])
    w = max(len(x) for x in left + ["days elapsed"]) + 4
    lines = ["    %-*s%s" % (w, "days elapsed", "band")]
    for x, b in zip(left, bands):
        lines.append("    %-*s%s" % (w, x, b))
    return "\n".join(lines)


TASK_TEMPLATE = """{org} {title}
Reference date: {ref}

You are the duty clerk. Place every {entity} on the schedule below into one band,
using the policy extract that follows.

POLICY EXTRACT ({manual}, section 4)

  Every open {entity} is placed in a band according to the number of days elapsed
  as at the reference date. Days elapsed are counted from the applicable date on
  the record: count the days falling after that date, up to and including the
  reference date.

{table}

  The {unit} operates Monday to Friday. Public holidays falling in the period
  covered by this schedule are:

{hol}

SCHEDULE ({fmtnote})

{body}

File one line per record: the record id, a comma, and the band, with no header and
no other text. File every record, in the order the records appear above.
"""


# --------------------------------------------------------------------------
# the oracle template, and reading it back
# --------------------------------------------------------------------------

ANCHOR_WORDS = {
    "event": "the date on which the underlying event occurred",
    "intake": "the date the record was opened with us",
    "last_action": "the date of the most recent action on the record",
}
BASIS_WORDS = {
    "calendar": "calendar days, weekends and public holidays included",
    "business": (
        "business days only, Monday to Friday, weekends excluded, public holidays counted "
        "as ordinary working days"
    ),
    "business_minus_holidays": (
        "business days only, Monday to Friday, excluding both weekends and the public "
        "holidays listed with the task"
    ),
}
BOUNDARY_WORDS = {
    "lower": "the LOWER of the two bands, the figure being the upper limit of the lower band",
    "upper": "the HIGHER of the two bands, the figure being the lower limit of the higher band",
}
MISSING_WORDS = {
    "lowest": "the first band in the table",
    "highest": "the last band in the table",
    "blank": "no band at all, the band being left empty",
    "pending": "the band %s, which is not in the table" % PENDING_TOKEN,
}
PHRASES = {
    "anchor": ANCHOR_WORDS,
    "basis": BASIS_WORDS,
    "boundary": BOUNDARY_WORDS,
    "missing": MISSING_WORDS,
}

ORACLE_HEAD: tuple[str, ...] = (
    "HOUSE CONVENTIONS (standing; they apply to every schedule handled in this",
    "engagement, not only to the one above)",
    "",
)

#: What each option means, in words. This is a declaration, not a renderer: the
#: package renders the oracle from it and reads one back with the same table, so the
#: sentence an oracle child reads and the rule an admission check believes it states
#: cannot come apart.
ORACLE_TEMPLATE = OracleTemplate(
    head=ORACLE_HEAD,
    sentences={
        "anchor": "The applicable date is {}.",
        "basis": "Days elapsed are counted as {}.",
        "boundary": (
            "Where a day count falls exactly on a figure printed in the band table, "
            "the record takes {}."
        ),
        "missing": "A record with no dates entered at all is given {}.",
    },
    phrases={
        "anchor": ANCHOR_WORDS,
        "basis": BASIS_WORDS,
        "boundary": BOUNDARY_WORDS,
        "missing": MISSING_WORDS,
    },
)


# --------------------------------------------------------------------------
# reading a filing
# --------------------------------------------------------------------------


def _norm(value: object) -> str:
    """One filed value as the family reads it: collapsed whitespace, printable ASCII.

    THE FOLD IS PART OF THE READING, not a renderer's afterthought. Every byte the
    agent types reaches a cell through here, and the serializer refuses a field that
    is not ASCII, so a value left exactly as typed lets one accented letter decide
    whether the fork renders at all. Folding here means the scorer and both renderers
    see one value, and a character that will not fold costs the agent its identifier
    match, which is a reason-coded outcome rather than an exception out of the seal.

    Control bytes fold for the same reason. ESC, NUL and backspace are ASCII, so the
    serializer passes them, and the observed column is echoed into the placebo, where
    an escape sequence is highlighting in the arm that is meant to be inert.
    """
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return "".join(ch if " " <= ch <= "~" else "?" for ch in text)


def _fold(value: object) -> str:
    return _norm(value).lower()


def _lines(raw: object) -> list[str] | None:
    """The filing as lines, or None when there is nothing readable in it at all."""
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


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class LedgerGenerator:
    """The ledger genre, as the generator protocol wants it."""

    name = "ledger"
    genre = "clerical date counting"
    SHAPE = SHAPE
    AXES = AXES
    SCORING: str = ROW_ADDITIVE_EQUAL_WEIGHT

    # ----- the instance -----

    def surface_for(self, ordinal: int, label: str) -> str:
        i = int(ordinal)
        if label.upper() == "A":
            return POOL_A[i % len(POOL_A)]
        return POOL_B[(i + i // len(POOL_B)) % len(POOL_B)]

    def surface_templates(self) -> tuple[str, ...]:
        return tuple(POOL_A) + tuple(POOL_B)

    def build_table(self, master: bytes, ordinal: int, label: str) -> LedgerTable:
        """A table this instance's own surface stream produced, and that moves on every axis."""
        surface = self.surface_for(ordinal, label)
        dom = DOMAINS[surface]
        stream = streams.SURFACE_A if label.upper() == "A" else streams.SURFACE_B
        for attempt in range(400):
            rng = streams.rng(master, stream, ordinal, attempt)
            rows, holidays = build_rows(rng, dom)
            moved, distinct = leverage(rows, surface, holidays)
            if not distinct:
                continue
            if moved["boundary"][0] < 3 or moved["anchor"][0] < 3 or moved["basis"][0] < 2:
                continue
            body = render_rows(rows, dom, streams.rng(master, stream, ordinal, attempt, "text"))
            return LedgerTable(
                domain=surface, rows=tuple(rows), holidays=tuple(holidays), body=body
            )
        raise RuntimeError(
            "no ledger table for %s at instance %d moved enough rows on every axis"
            % (surface, ordinal)
        )

    def table_record(self, table: LedgerTable) -> dict[str, Any]:
        """Every field of a ledger table, as one canonical value.

        The whole table, not the parts the bank happens to reach through some other
        field: the domain names the phrases and the band table, the rows carry each
        record's identifier and its three dates, the holidays decide the business
        basis, and the body is the schedule the agent reads. All five are read
        somewhere between describing the task and rendering the receipt, so all five
        are what the bank commits to.
        """
        return {
            "domain": table.domain,
            "rows": [
                {
                    "row_id": row.row_id,
                    "dates": (
                        None if row.dates is None
                        else {k: row.dates[k].isoformat() for k in sorted(row.dates)}
                    ),
                }
                for row in table.rows
            ],
            "holidays": [day.isoformat() for day in table.holidays],
            "body": table.body,
        }

    def build_envelope(self, master: bytes, ordinal: int) -> Envelope:
        """The registered envelope, with this instance's committed filler and neutrals."""
        neutral: dict[str, tuple[str, ...]] = {}
        for spec in SLOTS:
            neutral[spec.name] = tuple(
                streams.filler_stream(
                    master, FILLER_ALPHABET, spec.width, ordinal, "neutral", spec.name, row
                )
                for row in range(SHAPE.rows)
            )
        return Envelope(
            size=ENVELOPE_SIZE,
            identifier_width=IDENTIFIER_WIDTH,
            observed_width=OBSERVED_WIDTH,
            slots=SLOTS,
            filler=streams.filler_stream(
                master, FILLER_ALPHABET, ENVELOPE_SIZE, ordinal, "pad"
            ),
            column_titles=COLUMN_TITLES,
            neutral=neutral,
        )

    def key_for(self, table: LedgerTable, convention: Mapping[str, str]) -> tuple[str, ...]:
        return key_for(table, convention)

    def normalize_answer(self, value: str) -> str:
        # The same function the scorer compares with, so the two cannot drift.
        return _fold(value)

    def row_identifiers(self, table: LedgerTable) -> tuple[str, ...]:
        return tuple(r.row_id for r in table.rows)

    def row_classes(self, table: LedgerTable) -> tuple[str, ...]:
        return tuple("dated" if r.dates else "undated" for r in table.rows)

    def answer_ranks(self, table: LedgerTable) -> tuple[str, ...]:
        # The band table the task prints, in the order it prints it. Anyone holding
        # both schedules can read this off them.
        return tuple(table.dom["bands"])

    def row_label(self, table: LedgerTable) -> tuple[str, ...]:
        # Every row names the record it grades. Nothing on this receipt names an axis.
        return (ROW_LABEL,) * len(table.rows)

    # ----- reading the filing -----

    def parse_and_canonicalize(self, task: Task, raw: object) -> Filing:
        """One canonical value per printed row, by registered rules.

        The rules, fixed and the same for every ledger instance: a line is a record
        identifier, a comma, and a band. Identifiers match case-insensitively after
        whitespace is collapsed. The FIRST line for an identifier wins and later
        ones are recorded as duplicates. A line naming no known identifier is an
        extra. A row with no line is an omission and canonicalizes to empty.

        One forgiving reading is registered and it is narrow: a filing with no
        commas anywhere is read positionally ONLY when it has exactly one line per
        printed row. Any other comma-free filing names no identifier and is a
        reason-coded NoFiling, so a paragraph of prose cannot be read as an answer
        to the first rows of the schedule.
        """
        lines = _lines(raw)
        if lines is None:
            return NoFiling("unreadable")
        if not any(line for line in lines):
            return NoFiling("empty")

        identifiers = self.row_identifiers(task.table)
        position = {_fold(row_id): i for i, row_id in enumerate(identifiers)}
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
                    values[i] = _norm(line)
                    seen.add(i)
                    filed += 1
        else:
            for line in lines:
                if not line:
                    continue
                head, _, tail = line.partition(",")
                index = position.get(_fold(head))
                if index is None:
                    extras.append(_norm(head))
                    continue
                filed += 1
                if index in seen:
                    duplicates.append(identifiers[index])
                    continue
                values[index] = _norm(tail)
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

    def score(self, task: Task, canonical: Filing) -> tuple[float, tuple[RowOutcome, ...]]:
        """The sealed scalar and what the filing did on every row.

        The component score is the fraction of rows the filing got right: equal
        weight per row, the denominator is the printed row count, rounded to six
        places. A NoFiling scores zero and still reports every row, so a receipt
        rendered against it has the same shape as any other.

        A row counts only if it was FILED. One option on the `missing` axis is the
        empty band, so a row nobody filed would otherwise match it, and an agent
        that submitted nothing at all would collect every undated row for free
        whenever that option was drawn.
        """
        identifiers = self.row_identifiers(task.table)
        truth = task.key if task.key else ("",) * len(identifiers)
        submitted = canonical if isinstance(canonical, SealedSubmission) else None
        outcomes: list[RowOutcome] = []
        for i, identifier in enumerate(identifiers):
            got = ""
            was_filed = False
            if submitted is not None:
                got = submitted.values[i] if i < len(submitted.values) else ""
                was_filed = submitted.filed[i] if i < len(submitted.filed) else False
            outcomes.append(
                RowOutcome(
                    ordinal=i + 1,
                    identifier=identifier,
                    filed=got,
                    was_filed=was_filed,
                    correct=truth[i] if i < len(truth) else "",
                    matched=was_filed
                    and self.normalize_answer(got)
                    == self.normalize_answer(truth[i] if i < len(truth) else ""),
                )
            )
        if not outcomes:
            return 0.0, ()
        matched = sum(1 for o in outcomes if o.matched)
        return round(matched / float(len(outcomes)), 6), tuple(outcomes)

    # ----- the task text -----

    def describe(self, task: PublicTask) -> str:
        table: LedgerTable = task.table
        dom = table.dom
        hol = "\n".join(
            "    %s   %s" % (d.isoformat(), n) for d, n in zip(table.holidays, dom["holnames"])
        )
        return TASK_TEMPLATE.format(
            org=dom["org"], title=dom["title"], ref=dom["refdate"].isoformat(),
            entity=dom["entity"], manual=dom["manual"], table=band_table(dom),
            unit=dom["unit"], hol=hol, fmtnote=FMT_NOTE[dom["fmt"]], body=table.body,
        )

    # ----- the three cells -----

    def render_receipt(
        self, task: Task, canonical: Filing, truth: Sequence[str]
    ) -> ReceiptAST:
        """One verdict per record, on what the filing did.

        The rows are built by the shared grader from the scorer's own outcomes, so
        what a row says is not a choice this module gets to make. The receipt names
        records, never axes, and a correction is that row's own answer rather than a
        value picked for what a reader could deduce from it.
        """
        graded = Task(
            label=task.label, task_id=task.task_id, surface=task.surface, table=task.table,
            text=task.text, key=tuple(truth),
        )
        _, outcomes = self.score(graded, canonical)
        return graded_receipt(task.task_id, outcomes, BLANK_TOKEN, UNFILED_TOKEN)

    def render_placebo(
        self, task: PublicTask, canonical: Filing, envelope: Envelope
    ) -> ReceiptAST:
        """The inert cell: congruent with the graded one outside the registered slots.

        It takes the PUBLIC task and the envelope, so there is no argument here
        through which the hidden rule could reach it. Its slot values are this
        instance's committed neutral tokens, chosen before launch and by nothing that
        has seen a filing or a convention.
        """
        blind = Task(
            label=task.label, task_id=task.task_id, surface=task.surface, table=task.table,
            text="", key=(),
        )
        _, outcomes = self.score(blind, canonical)
        return placebo_receipt(
            task.task_id, outcomes, envelope, BLANK_TOKEN, UNFILED_TOKEN
        )

    #: The declared phrase table. Rendering and reading both go through it.
    ORACLE = ORACLE_TEMPLATE

    def render_oracle(
        self, task_id: str, convention: Mapping[str, str], row_count: int = 0
    ) -> ReceiptAST:
        """The drawn rule, from the declared phrases, with no rows to align.

        It carries its task's row count like its siblings do, so the outer wrapper
        is the same three lines in all three cells and only the body differs.
        """
        return render_oracle_cell(ORACLE_TEMPLATE, task_id, convention, row_count)

    def parse_oracle(self, ast: ReceiptAST) -> dict[str, str]:
        return parse_oracle_cell(ORACLE_TEMPLATE, ast)


def _shown(value: str) -> str:
    return value if value else BLANK_TOKEN


def _observed(outcome: RowOutcome) -> str:
    """What the receipt echoes back in the filed column."""
    if not outcome.was_filed:
        return UNFILED_TOKEN
    return _shown(outcome.filed)


GENERATOR = LedgerGenerator()


__all__ = [
    "ALL_CONVENTIONS",
    "AXES",
    "BLANK_TOKEN",
    "UNFILED_TOKEN",
    "DOMAINS",
    "ENVELOPE_SIZE",
    "GENERATOR",
    "ORACLE_TEMPLATE",
    "PENDING_TOKEN",
    "POOL_A",
    "POOL_B",
    "SHAPE",
    "SLOTS",
    "LedgerGenerator",
    "LedgerRow",
    "LedgerTable",
    "band_for",
    "daycount",
    "key_for",
    "leverage",
]
