"""The retail refund genre: one destination per authorized return, three hidden choices.

A schedule of already authorized returns, each with its order, its items, its refund
total and the complete list of payment instruments the house may refund it to. The
public rules settle eligibility, class membership, the empty-class fallback, ties and
normalization. Three decisions they never make are the hidden convention:

    route_policy   3 options   which destination class is preferred
    gift_pick      3 options   which gift card is taken inside the gift class
    origin_pick    3 options   which original contributor is taken inside its class

That is 27 conventions, drawn uniformly and independently. Sibling tasks A and B are
two schedules over different organizations, item pools, amounts, identifiers and code
bindings, both scored under the same drawn convention.

WHAT THE HIDDEN DECISION IS. It is a house rule for routing money to an eligible
economic instrument, not a measurement, not an entity match and not a string
transformation. Every candidate destination is printed, already authorized and
permitted for the whole refund; what is not printed is which class the house prefers
and how it picks inside a class. A public retail policy can establish which
instruments are permitted and cannot disclose an independently drawn preference.

THE CLASSES ARE DISJOINT, AND THAT IS A DERIVATIVE RULE. A gift card belongs to the
gift-card class whether or not it paid for the purchase, and the original class is
every listed credit card or PayPal instrument with a positive contribution. Native
tau2 permits an original payment method or an existing gift card and its return tool
restricts the non-gift destination to the first payment in the order's history; this
genre permits every positive non-gift contributor instead. Neither change may be
presented as unchanged upstream behaviour.

WHY THE RECEIPT CAN CARRY SOMETHING. The scored column holds a destination code the
rule COMPUTES, not the option the agent chose, and a code names an instrument only
inside its own case: D2 in one case and D2 in another are different instruments of
different types. Single-class cases with opposite amount orientations separate the
first-listed selector from the two extremum selectors; cases of both classes, with and
without a gift purchase, separate the three routes. `build_pair` searches for a pair of
schedules where that happens under EVERY draw rather than under the one this instance
drew.

WHAT IS EASY HERE AND WHAT IS NOT. Executing the rule is short: read a class off two
printed fields, compare at most three printed amounts, file a two-byte code. That is a
deliberate choice about what a copy has to do once it has the rule, and it is NOT a
repair for the low oracle grade an earlier engineering run recorded. That run's loss
was in uptake and in scope, not in execution: almost every filing was the exact answer
key of some convention the family draws from, an independent recomputation agreed with
the grader on every row, and no copy skipped a row. The repairs were to registered
wording and they were made. The remaining cost here is reading: side A prints 72
instrument entries and side B prints 108.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from shogym.envs.receipts import filing as shared_filing
from shogym.envs.receipts import streams
from shogym.envs.receipts.copy_profiles import ORDERED_TOKENS
from shogym.envs.receipts.filing import scope_sentence
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.oracle import parse as parse_oracle_cell
from shogym.envs.receipts.oracle import render as render_oracle_cell
from shogym.envs.receipts.protocol import (
    FULL_RECEIPT,
    ROW_ADDITIVE_EQUAL_WEIGHT,
    Axis,
    Column,
    ConstructionExhausted,
    Filing,
    PublicTask,
    ReceiptPolicy,
    RowOutcome,
    Shape,
    Task,
)
from shogym.envs.receipts.receipt_ast import (
    Envelope,
    ReceiptAST,
    SlotSpec,
    envelope_size_for,
)
from shogym.envs.receipts.render import Feedback, graded_receipt, placebo_receipt
from shogym.receipts import ROW_LABEL

# ----- the destination vocabulary --------------------------------------------
#: The published code vocabulary, in listing order. It is COMPLETE and PUBLIC: both
#: task texts print it, so the copy screen's maps between two published orders can be
#: built by a reader with no idea what the rule is. Every accepted table realizes all
#: six codes under every draw, which is what keeps that monoid the six rotations and
#: the closure finite.
CODES = ("D1", "D2", "D3", "D4", "D5", "D6")

GIFT_CARD = "gift_card"
CREDIT_CARD = "credit_card"
PAYPAL = "paypal"
NON_GIFT_TYPES = (CREDIT_CARD, PAYPAL)

#: What the receipt prints for a row filed with an empty value, and for a row filed
#: with nothing at all. Two different acts, two different tokens, and neither is a
#: legal destination code.
BLANK_TOKEN = "(empty)"
UNFILED_TOKEN = "(none)"

AXES: tuple[Axis, ...] = (
    Axis(
        "route_policy",
        ("route_gift", "route_origin", "route_usedgift"),
        "which destination class the house prefers",
    ),
    Axis(
        "gift_pick",
        ("gift_list", "gift_max", "gift_min"),
        "which gift card is taken inside the gift-card class",
    ),
    Axis(
        "origin_pick",
        ("origin_list", "origin_max", "origin_min"),
        "which original contributor is taken inside its class",
    ),
)

#: The option to the operation it names, kept as three small tables rather than as
#: branching inside the selector. The validation module reimplements them from the
#: option identifiers instead of importing them, so an edited table here is a
#: disagreement there rather than a change both sides make together.
PREFERS_GIFT = {"route_gift": True, "route_origin": False}
FIRST_LISTED = "_list"
GREATEST = "_max"
LEAST = "_min"

SHAPE = Shape(
    columns=(
        Column("case_id", "CA on A or CB on B and eight uppercase hexadecimal "
                          "characters, sampled without replacement in the table"),
        Column("order_id", "#W and seven digits, unique within the table"),
        Column("items", "one or two returned items, each with an item id, a name from "
                        "the surface's pool and a price in whole dollars"),
        Column("refund_total", "the sum of the positive instrument contributions, in "
                               "integer cents, equal to the sum of the item prices"),
        Column("gift_used", "yes when a gift card contributed to the purchase"),
        Column("instruments", "the ordered destination list: code, payment method id, "
                              "type, balance where applicable, purchase contribution"),
    ),
    rows=24,
    case="one authorized return awaiting one refund destination",
    note="every case has at least one eligible destination, and no case ties",
)
ROWS = SHAPE.rows

# ----- the registered envelope constants -------------------------------------
# Family maxima and fixed widths, none of them read off a particular draw, which is
# what makes the envelope size convention-independent.

IDENTIFIER_WIDTH = 12
OBSERVED_WIDTH = 16
VERDICT_WIDTH = 4
#: TWELVE IS ENOUGH HERE, AND IT WOULD NOT BE FOR A NATIVE PAYMENT ID. The canonical
#: answer is a two-byte destination code, so a correction is never truncated. A later
#: instrument that files `credit_card_1234567` needs at least 19 bytes in both the
#: observed and the correction field, and a silently truncated correct destination is
#: the one failure this width may never produce.
CORRECTION_WIDTH = 12
VERDICT_TOKENS = ("PASS", "FAIL")
SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec("verdict", VERDICT_WIDTH, vocabulary=VERDICT_TOKENS),
    SlotSpec(
        "correction", CORRECTION_WIDTH,
        vocabulary=(BLANK_TOKEN,), allows_answers=True, allows_empty=True,
    ),
)
COLUMN_TITLES = ("mark", "note")
#: Digits only. A digit string can never spell a verdict token, a display token or a
#: destination code under either spelling, which is the property the neutral check
#: reads; sharing the character 1 with D1 is not a disclosed grade.
FILLER_ALPHABET = "0123456789"
ORACLE_BODY_ALLOWANCE = 1100
ENVELOPE_SIZE = envelope_size_for(
    max_rows=ROWS,
    identifier_width=IDENTIFIER_WIDTH,
    observed_width=OBSERVED_WIDTH,
    slots=SLOTS,
    body_allowance=ORACLE_BODY_ALLOWANCE,
)

# ----- the four surfaces ------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """One presentation template. A surface decides the organization, the item names
    and how the schedule is printed, and nothing else: every option of every axis
    means the same operation on all four."""

    name: str
    organization: str
    items: tuple[str, ...]
    format_note: str
    json_body: bool


PIPE_NOTE = "pipe-separated columns; lists use JSON"
JSON_NOTE = "one JSON object per case; array order is printed order"

SURFACES: dict[tuple[str, int], Surface] = {
    ("A", 0): Surface(
        "apparel", "North Quay Outfitters", ("Tee", "Scarf", "Belt", "Cap"),
        PIPE_NOTE, False,
    ),
    ("A", 1): Surface(
        "home", "Willow House", ("Towel", "Mug", "Lamp", "Tray"), PIPE_NOTE, False
    ),
    ("B", 0): Surface(
        "electronics", "Relay Electronics", ("Cable", "Mouse", "Speaker", "Charger"),
        JSON_NOTE, True,
    ),
    ("B", 1): Surface(
        "outdoors", "Trail Store", ("Bottle", "Flask", "Glove", "Lantern"),
        JSON_NOTE, True,
    ),
}
SURFACE_BY_NAME = {surface.name: surface for surface in SURFACES.values()}
HEADER = (
    "case_id", "order_id", "items", "refund_total", "gift_used", "instruments",
)


def surface_index(ordinal: int, label: str) -> int:
    """Which of a side's two surfaces this ordinal prints.

    The two sides advance differently on purpose, so the pair of surfaces an ordinal
    shows is not one fact about the ordinal that a reader could carry between them.
    """
    n = int(ordinal)
    return n % 2 if label.upper() == "A" else (n + n // 2) % 2


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RetailItem:
    """One returned item. Quantities are all one and every listed item is returned."""

    item_id: str
    name: str
    price: int  # integer cents


@dataclass(frozen=True)
class RetailInstrument:
    """One printed destination: its row-local code and everything a reader compares.

    `balance` is None on a non-gift instrument, where balance is not applicable, and
    prints as JSON null. `contribution` is what this instrument actually paid for this
    order, in integer cents; zero means it did not pay for this purchase.
    """

    code: str
    payment_method_id: str
    type: str
    balance: int | None
    contribution: int


@dataclass(frozen=True)
class RetailCase:
    """One authorized return awaiting one destination."""

    case_id: str
    order_id: str
    items: tuple[RetailItem, ...]
    refund_total: int
    gift_used: bool
    instruments: tuple[RetailInstrument, ...]


@dataclass(frozen=True)
class RetailTable:
    """One side's frozen schedule: its surface, its ordered cases and its printed body."""

    domain: str
    rows: tuple[RetailCase, ...]
    body: str

    @property
    def template(self) -> Surface:
        return SURFACE_BY_NAME[self.domain]


# --------------------------------------------------------------------------
# the rule: the scoring function's own selector
# --------------------------------------------------------------------------


def gift_class(case: RetailCase) -> tuple[RetailInstrument, ...]:
    """Every listed gift card, in printed order, contributor or not."""
    return tuple(i for i in case.instruments if i.type == GIFT_CARD)


def original_class(case: RetailCase) -> tuple[RetailInstrument, ...]:
    """Every listed non-gift instrument with a positive contribution, in printed order."""
    return tuple(
        i for i in case.instruments
        if i.type in NON_GIFT_TYPES and i.contribution > 0
    )


def gift_purchase(case: RetailCase) -> bool:
    """Whether any gift card made a positive contribution to this purchase."""
    return any(i.contribution > 0 for i in gift_class(case))


def _selected(
    members: Sequence[RetailInstrument], selector: str, amount_of
) -> RetailInstrument:
    """One member of a nonempty class, by the selector the convention names.

    Ties go to the earliest printed member, which is public and fixed rather than a
    hidden boundary decision. The admitted distribution never produces one; the rule
    still has to state what happens, and the hand fixtures test it.
    """
    if selector.endswith(FIRST_LISTED):
        return members[0]
    if selector.endswith(GREATEST):
        return min(
            enumerate(members), key=lambda pair: (-amount_of(pair[1]), pair[0])
        )[1]
    if selector.endswith(LEAST):
        return min(
            enumerate(members), key=lambda pair: (amount_of(pair[1]), pair[0])
        )[1]
    raise ValueError(f"no selector is registered for {selector!r}")


def destination(case: RetailCase, convention: Mapping[str, str]) -> str:
    """The correct destination code for one case under one convention.

    The route names a preferred class; an empty preferred class falls back to the
    other one, publicly and always, and the convention's selector for THAT class is
    what then picks inside it. An original selector never compares a gift's
    contribution with a card's, because the two selectors run on disjoint classes.
    """
    gifts = gift_class(case)
    originals = original_class(case)
    route = convention["route_policy"]
    prefer_gift = PREFERS_GIFT.get(route)
    if prefer_gift is None:
        prefer_gift = gift_purchase(case)
    members = gifts if prefer_gift else originals
    if not members:
        prefer_gift = not prefer_gift
        members = gifts if prefer_gift else originals
    if not members:
        raise ValueError(f"case {case.case_id} lists no eligible destination at all")
    if prefer_gift:
        return _selected(
            members, convention["gift_pick"], lambda i: int(i.balance or 0)
        ).code
    return _selected(members, convention["origin_pick"], lambda i: i.contribution).code


def key_for(table: RetailTable, convention: Mapping[str, str]) -> tuple[str, ...]:
    """The correct destination for every printed case, in table row order."""
    return tuple(destination(case, convention) for case in table.rows)


ALL_CONVENTIONS: tuple[dict[str, str], ...] = tuple(
    {"route_policy": r, "gift_pick": g, "origin_pick": o}
    for r in AXES[0].options
    for g in AXES[1].options
    for o in AXES[2].options
)


# --------------------------------------------------------------------------
# the public case kinds, and the profile multiset
# --------------------------------------------------------------------------

GIFTS_ONLY = "gifts_only"
ORIGINALS_ONLY = "originals_only"
BOTH_GIFT_PURCHASE = "both_gift_purchase"
BOTH_NO_GIFT_PURCHASE = "both_no_gift_purchase"
KINDS = (GIFTS_ONLY, ORIGINALS_ONLY, BOTH_GIFT_PURCHASE, BOTH_NO_GIFT_PURCHASE)
#: How many cases of each public kind a side prints. Six each, and the whole of the
#: table: the fallback strata and both purchase types are what keep every selector
#: material under every route, so they are counts and not proportions.
PER_KIND = 6

#: A's two candidate orientations per class, three cases each. Without both, the first
#: listed member and one extremum agree throughout a class and the two selectors stop
#: being separable on A's receipt. It is also how many cases carry each of the eight
#: JOINT profiles below, which is the stronger statement: the two classes of one case
#: are oriented together, and a tally of each class on its own cannot see that.
PER_ORIENTATION = 3

#: The dollar pools. Gift balances may be zero, because a zero-balance gift card is
#: still a permitted destination and the greatest-balance and least-balance selectors
#: have to mean the same operation on it as on any other.
GIFT_BALANCES = tuple(range(0, 100, 10))
ORIGINAL_AMOUNTS = tuple(range(10, 100, 10))
GIFT_CONTRIBUTIONS = tuple(range(10, 100, 10))
CENTS = 100

#: A's classes carry two candidates and B's carry three, deliberately. B's first
#: candidate has the middle amount, so a reader applying an inferred extremum
#: preference has to file a different code from the printed first candidate, and there
#: is no cardinality-and-rank profile the two sides share.
A_CLASS_SIZE = 2
B_CLASS_SIZE = 3

#: Identifier namespaces, disjoint between the siblings. They identify a side and
#: never a convention.
ORDER_RANGE = {"A": (1000000, 5000000), "B": (5000000, 10000000)}
PAYMENT_RANGE = ORDER_RANGE
ITEM_RANGE = {"A": (1000000000, 5000000000), "B": (5000000000, 10000000000)}
CASE_HEX = 16 ** 8

#: How many whole pairs one ordinal may try before construction is a named failure.
MAX_ATTEMPTS = 400


#: One row of a side, as the recipe states it: the public kind, the orientation of
#: each class A carries, and the codes the gift class and the original class bind.
Row = tuple[str, bool | None, bool | None, tuple[str, ...], tuple[str, ...]]

#: A's eight joint profiles, in the order one block prints them, with each class's
#: codes in that class's own printed order. Three blocks of these eight are the whole
#: side, so A's correct destination repeats with period eight under every one of the
#: 27 conventions, and every rotation and reversal keeps that period.
#:
#: WHAT THE PERIOD IS FOR. A reader holding A's corrected answers may relabel the six
#: codes and move the rows, and the wider of the two copying families allows all 720
#: relabellings against the 48 registered row moves. One relabelling is one function
#: of A's answer, so against a period-eight A it returns the same code at three
#: positions eight apart, and B binds three different codes there. At most one of each
#: such triple can be right: eight of the 24 rows, whatever the relabelling and
#: whatever the row move. The bound is a property of the two layouts rather than of a
#: lucky draw, which is what makes the wider qualification something the construction
#: meets instead of something a search hunts for. Rejecting proposals until one
#: happened to clear it was 4000 attempts for nothing, because unstructured keys run
#: near 14 of 24.
A_BLOCK: tuple[Row, ...] = (
    (GIFTS_ONLY, True, None, ("D6", "D2"), ()),
    (GIFTS_ONLY, False, None, ("D6", "D1"), ()),
    (ORIGINALS_ONLY, None, True, (), ("D3", "D5")),
    (ORIGINALS_ONLY, None, False, (), ("D3", "D4")),
    (BOTH_GIFT_PURCHASE, True, True, ("D2", "D5"), ("D1", "D4")),
    (BOTH_GIFT_PURCHASE, False, False, ("D1", "D4"), ("D2", "D6")),
    (BOTH_NO_GIFT_PURCHASE, True, False, ("D5", "D6"), ("D4", "D2")),
    (BOTH_NO_GIFT_PURCHASE, False, True, ("D4", "D2"), ("D5", "D1")),
)

#: B's four public kinds in the order one block prints them, and the codes a class
#: binds to its median, least and greatest amounts in the first block. Six blocks of
#: four, each advancing every code through the published vocabulary by its own block
#: index, so one kind carries codes two and four apart at positions eight apart. That
#: is the other half of the bound: three different right answers under one function of
#: A's repeating answer.
B_BLOCK: tuple[str, ...] = KINDS
B_GIFT_CODES = ("D1", "D2", "D3")
B_ORIGIN_CODES = ("D4", "D5", "D6")


def _advanced(codes: Sequence[str], shift: int) -> tuple[str, ...]:
    """The same codes, each advanced through the published vocabulary by one shift."""
    return tuple(CODES[(CODES.index(code) + shift) % len(CODES)] for code in codes)


def profile_plan(label: str) -> list[Row]:
    """The exact rows of one side, in the order the recipe prints them.

    Each entry is a public case kind, the orientation of each class it carries on A,
    and the codes that class binds. True means the smaller amount is printed first. B
    has no orientation entry, because every B class prints its median first and
    randomizes the rest, and B binds its codes to the amount ranks rather than to the
    printed positions, which is what survives that randomization.

    THE ORDER IS THE RECIPE, NOT A PRESENTATION. The profile multiset is the same one
    the specification registered, and it is laid out in blocks rather than shuffled
    because the copying bound is a statement about rows eight apart. `build_side`
    still moves the whole side by one registered rotation or reversal afterwards, so
    no instance opens on the same profile as the last one.
    """
    if label.upper() == "B":
        return [
            (kind, None, None,
             _advanced(B_GIFT_CODES, block), _advanced(B_ORIGIN_CODES, block))
            for block in range(ROWS // len(B_BLOCK))
            for kind in B_BLOCK
        ]
    return [row for _ in range(ROWS // len(A_BLOCK)) for row in A_BLOCK]


def case_kind(case: RetailCase) -> str:
    """The public kind a reader derives from the printed case."""
    gifts = gift_class(case)
    originals = original_class(case)
    if gifts and not originals:
        return GIFTS_ONLY
    if originals and not gifts:
        return ORIGINALS_ONLY
    return BOTH_GIFT_PURCHASE if gift_purchase(case) else BOTH_NO_GIFT_PURCHASE


# --------------------------------------------------------------------------
# inventing a schedule
# --------------------------------------------------------------------------


def _ordered(values: list[int], small_first: bool | None, rng: random.Random) -> list[int]:
    """One class's amounts in printed order, by the side's own ordering law.

    On A the two amounts are printed smaller first or larger first, which is what the
    orientation strata are. On B the median is printed first and the minimum and
    maximum follow in a uniformly random order, which is what makes the first printed
    candidate neither extremum.
    """
    ordered = sorted(values)
    if small_first is None:
        rest = [ordered[0], ordered[-1]]
        rng.shuffle(rest)
        return [ordered[1]] + rest
    return ordered if small_first else list(reversed(ordered))


def _interleaved(
    gifts: list[RetailInstrument],
    originals: list[RetailInstrument],
    rng: random.Random,
) -> list[RetailInstrument]:
    """One uniformly chosen interleaving that preserves each class's own order.

    Without it a class would always print first and "first listed" would be a fact
    about the class rather than about the printed line, which is not the operation the
    rule names.
    """
    if not gifts:
        return list(originals)
    if not originals:
        return list(gifts)
    width = len(gifts) + len(originals)
    spots = sorted(rng.sample(range(width), len(gifts)))
    out: list[RetailInstrument | None] = [None] * width
    for spot, gift in zip(spots, gifts):
        out[spot] = gift
    rest = [n for n in range(width) if out[n] is None]
    for spot, original in zip(rest, originals):
        out[spot] = original
    return [instrument for instrument in out if instrument is not None]


def _money(cents: int) -> str:
    """One monetary amount as the schedule prints it: two places, never a float."""
    return "%d.%02d" % divmod(int(cents), CENTS)


class _Names:
    """The identifier namespaces of one side, drawn without replacement.

    A generator rather than a pile of loops because every one of them is the same act:
    take the next value of a stream that cannot repeat itself. Exhausting one is a
    named construction failure and not a duplicate identifier nobody notices.
    """

    def __init__(self, rng: random.Random, label: str) -> None:
        side = label.upper()
        low, high = ORDER_RANGE[side]
        item_low, item_high = ITEM_RANGE[side]
        self.cases = iter(rng.sample(range(CASE_HEX), ROWS))
        self.orders = iter(rng.sample(range(low, high), ROWS))
        self.payments = iter(rng.sample(range(low, high), ROWS * 6))
        self.items = iter(rng.sample(range(item_low, item_high), ROWS * 2))
        self.prefix = "CA" if side == "A" else "CB"

    def case_id(self) -> str:
        return self.prefix + "%08X" % next(self.cases)

    def order_id(self) -> str:
        return "#W%07d" % next(self.orders)

    def payment_id(self, kind: str) -> str:
        return "%s_%07d" % (kind, next(self.payments))

    def item_id(self) -> str:
        return "%010d" % next(self.items)


def _bound(
    members: list[RetailInstrument], codes: Sequence[str], label: str
) -> list[RetailInstrument]:
    """One class's instruments, with the recipe's codes bound to its own members.

    On A the code follows the member's printed position inside its class, which is
    what the two orientation strata move. On B it follows the member's amount rank,
    median then least then greatest, because B prints its median first and shuffles
    the other two: a code bound to a position would move with that shuffle, and a code
    bound to a rank is the same code in every instance of that block.
    """
    if not members:
        return []
    if label.upper() == "A":
        chosen = list(codes[: len(members)])
    else:
        amounts = [
            int(m.balance if m.balance is not None else m.contribution) for m in members
        ]
        order = sorted(range(len(members)), key=lambda n: amounts[n])
        chosen = [""] * len(members)
        for place, rank in enumerate((1, 0, 2)):
            chosen[order[place]] = codes[rank]
    return [
        RetailInstrument(
            code=code, payment_method_id=member.payment_method_id, type=member.type,
            balance=member.balance, contribution=member.contribution,
        )
        for code, member in zip(chosen, members)
    ]


def _instruments(
    kind: str,
    gift_small_first: bool | None,
    origin_small_first: bool | None,
    gift_codes: Sequence[str],
    origin_codes: Sequence[str],
    label: str,
    amounts: random.Random,
    interleave: random.Random,
    names: _Names,
) -> tuple[list[RetailInstrument], int]:
    """One case's printed instruments and its refund total, in integer cents."""
    size = A_CLASS_SIZE if label.upper() == "A" else B_CLASS_SIZE
    gifts: list[RetailInstrument] = []
    originals: list[RetailInstrument] = []

    if kind != ORIGINALS_ONLY:
        balances = _ordered(
            amounts.sample(GIFT_BALANCES, size), gift_small_first, interleave
        )
        if kind == GIFTS_ONLY:
            paid = [amounts.choice(GIFT_CONTRIBUTIONS) for _ in balances]
        elif kind == BOTH_GIFT_PURCHASE:
            chosen = amounts.randrange(size)
            value = amounts.choice(GIFT_CONTRIBUTIONS)
            paid = [value if n == chosen else 0 for n in range(size)]
        else:
            paid = [0] * size
        gifts = [
            RetailInstrument(
                code="", payment_method_id=names.payment_id(GIFT_CARD),
                type=GIFT_CARD, balance=balance * CENTS, contribution=value * CENTS,
            )
            for balance, value in zip(balances, paid)
        ]

    if kind != GIFTS_ONLY:
        contributions = _ordered(
            amounts.sample(ORIGINAL_AMOUNTS, size), origin_small_first, interleave
        )
        for value in contributions:
            payment_type = amounts.choice(NON_GIFT_TYPES)
            originals.append(
                RetailInstrument(
                    code="", payment_method_id=names.payment_id(payment_type),
                    type=payment_type, balance=None, contribution=value * CENTS,
                )
            )

    printed = _interleaved(
        _bound(gifts, gift_codes, label), _bound(originals, origin_codes, label),
        interleave,
    )
    return printed, sum(i.contribution for i in printed)


def _items(
    refund_total: int, surface: Surface, items: random.Random, names: _Names
) -> tuple[RetailItem, ...]:
    """One or two returned items whose prices sum to the refund total exactly."""
    count = items.choice((1, 2))
    dollars = refund_total // CENTS
    if count == 1 or dollars < 2:
        return (
            RetailItem(names.item_id(), items.choice(surface.items), refund_total),
        )
    first = items.randrange(1, dollars) * CENTS
    return (
        RetailItem(names.item_id(), items.choice(surface.items), first),
        RetailItem(names.item_id(), items.choice(surface.items), refund_total - first),
    )


def _compact(payload: object) -> str:
    """One nested value as the schedule prints it: compact, ASCII, key order kept."""
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _item_record(item: RetailItem) -> dict[str, str]:
    return {
        "item_id": item.item_id, "name": item.name, "price": _money(item.price)
    }


def _instrument_record(instrument: RetailInstrument) -> dict[str, object]:
    return {
        "code": instrument.code,
        "payment_method_id": instrument.payment_method_id,
        "type": instrument.type,
        "balance": None if instrument.balance is None else _money(instrument.balance),
        "contribution": _money(instrument.contribution),
    }


def render_body(cases: Sequence[RetailCase], surface: Surface) -> str:
    """The printed schedule: one line per case, joined by LF and no trailing newline."""
    lines: list[str] = []
    if not surface.json_body:
        lines.append(" | ".join(HEADER))
    for case in cases:
        values = {
            "case_id": case.case_id,
            "order_id": case.order_id,
            "items": [_item_record(item) for item in case.items],
            "refund_total": _money(case.refund_total),
            "gift_used": "yes" if case.gift_used else "no",
            "instruments": [_instrument_record(i) for i in case.instruments],
        }
        if surface.json_body:
            lines.append(_compact({name: values[name] for name in HEADER}))
            continue
        printed = []
        for name in HEADER:
            value = values[name]
            printed.append(value if isinstance(value, str) else _compact(value))
        lines.append(" | ".join(printed))
    return "\n".join(lines)


def _relabelled(
    cases: list[RetailCase], relabelling: Mapping[str, str]
) -> list[RetailCase]:
    """Every case of one side under one relabelling of the published codes.

    A relabelling of a whole side moves no amount, no rank, no class and no row, so
    every predicate the construction filter reads stays where it was and both copying
    families are closed under it. What it changes is which token a reader sees, so two
    ordinals do not print the same code on the same profile.
    """
    return [
        RetailCase(
            case_id=case.case_id, order_id=case.order_id, items=case.items,
            refund_total=case.refund_total, gift_used=case.gift_used,
            instruments=tuple(
                RetailInstrument(
                    code=relabelling[i.code], payment_method_id=i.payment_method_id,
                    type=i.type, balance=i.balance, contribution=i.contribution,
                )
                for i in case.instruments
            ),
        )
        for case in cases
    ]


def _moved(cases: list[RetailCase], structure: random.Random) -> list[RetailCase]:
    """The same rows under one registered row move: a rotation, of the order or of its reversal.

    The recipe's blocks bound copying through rows eight apart, and which block prints
    first is not part of that. Rotations and reversals are the moves the copy screen
    already prices, and each of them carries one such triple of positions onto
    another, so the bound survives the move and no two instances open on the same
    profile.
    """
    order = list(range(len(cases)))
    if structure.choice((False, True)):
        order.reverse()
    shift = structure.randrange(len(cases))
    return [cases[n] for n in order[shift:] + order[:shift]]


def build_side(master: bytes, ordinal: int, label: str, attempt: int) -> RetailTable:
    """One side's schedule for one construction attempt, from that side's own streams.

    The six streams are separate coordinates under the side's registered label, so the
    amounts do not move when the codes do and neither moves when the convention does.
    None of them is an ambient generator and none of them reads the draw. The recipe
    fixes which profile carries which codes; the codes stream picks the relabelling of
    the side and the structure stream picks its row move, which is where a side's
    presentation comes from now that its layout is the recipe.
    """
    side = label.upper()
    stream = streams.SURFACE_A if side == "A" else streams.SURFACE_B
    where = ("retail_refund", int(ordinal), int(attempt))
    structure = streams.rng(master, stream, *where, "structure")
    amounts = streams.rng(master, stream, *where, "amounts")
    identifiers = streams.rng(master, stream, *where, "identifiers")
    codes = streams.rng(master, stream, *where, "codes")
    items = streams.rng(master, stream, *where, "items")
    interleave = streams.rng(master, stream, *where, "interleave")

    names = _Names(identifiers, side)
    surface = SURFACES[(side, surface_index(ordinal, side))]
    cases: list[RetailCase] = []
    for kind, gift_small_first, origin_small_first, gift_codes, origin_codes in (
        profile_plan(side)
    ):
        printed, refund_total = _instruments(
            kind, gift_small_first, origin_small_first, gift_codes, origin_codes,
            side, amounts, interleave, names,
        )
        case = RetailCase(
            case_id=names.case_id(),
            order_id=names.order_id(),
            items=_items(refund_total, surface, items, names),
            refund_total=refund_total,
            gift_used=any(
                i.contribution > 0 for i in printed if i.type == GIFT_CARD
            ),
            instruments=tuple(printed),
        )
        cases.append(case)
    cases = _moved(
        _relabelled(cases, dict(zip(CODES, codes.sample(CODES, len(CODES))))), structure
    )
    return RetailTable(
        domain=surface.name, rows=tuple(cases), body=render_body(cases, surface)
    )


@lru_cache(maxsize=32)
def build_pair(master: bytes, ordinal: int) -> tuple[RetailTable, RetailTable]:
    """Both sibling schedules for one ordinal, from the public draw and nothing else.

    PURE, AND BOTH SIDES AT ONCE. The construction filter is a statement about the
    PAIR: it asks what a reader of A's corrected key can do on B, so neither side can
    be accepted on its own.

    THE FILTER NEVER CONSULTS THE DRAWN CONVENTION. Every predicate quantifies over
    the whole support, so acceptance is a function of the two schedules alone. A
    filter that read the live draw would make rejection informative: an agent that
    knew which pairs are accepted could exclude conventions from the fact that this
    pair was served, and `describe` being invariant would not save it.

    Running out of attempts is `ConstructionExhausted`, a named whole-bank failure
    that says which ordinal, not a skipped ordinal and not an unnamed exception.
    """
    from shogym.envs.receipts.generators import retail_validation as validation

    for attempt in range(MAX_ATTEMPTS):
        a_table = build_side(master, ordinal, "A", attempt)
        b_table = build_side(master, ordinal, "B", attempt)
        if validation.pair_refusal(a_table, b_table):
            continue
        return a_table, b_table
    raise ConstructionExhausted(
        f"no retail_refund pair at ordinal {ordinal} cleared the construction filter "
        f"in {MAX_ATTEMPTS} attempts"
    )


# --------------------------------------------------------------------------
# the task text
# --------------------------------------------------------------------------

TASK_TEMPLATE = """{organization}: refund destination schedule

Select one refund destination for every case below, using the house convention
in force for this schedule.

All orders below have been delivered. Customer identity, the return request,
the returned items and authorization to use any listed destination have already
been confirmed. Every listed item is being returned, once, in full. There are
no earlier refunds. Each case is independent. Selecting a destination in one
case changes no balance or contribution in another case.

The schedule supplies the complete list of payment instruments relevant to
each case. Each instrument has a destination code, a payment-method ID, a type,
a current balance when applicable, and its contribution to this purchase.
Contribution is the total actually paid with that instrument for this order.
Zero means that instrument did not pay for this purchase. Amounts are USD,
shown to two decimal places. A null balance means balance is not applicable.
The refund_total is supplied; you do not need to calculate it from the items.

There are two destination classes:
* Gift cards: every listed instrument of type gift_card, whether or not it paid
  for this purchase.
* Original instruments: every listed credit_card or paypal instrument with a
  positive contribution to this purchase.

A gift card that paid for the purchase belongs only to the gift-card class.
The gift_used field is yes exactly when at least one gift card made a positive
contribution to the purchase; otherwise it is no. This field is supplied for
convenience and agrees with the contributions.

Every listed instrument belongs to one of those two classes and is a permitted
destination for the entire refund. There is no cap based on its contribution,
and a gift card may receive a refund even when its current balance is zero.
No unlisted instrument, split refund, cash payment or new gift card is allowed.
At least one class is nonempty in every case.

The house convention determines which class is preferred and how an instrument
is chosen within the selected class. Those preferences are not stated here.
If the preferred class has no listed instruments, use the other class and
apply the house convention for choosing within that class.

Whenever a choice compares amounts, compare the printed numeric amounts.
If several instruments tie for the amount selected by the convention, choose
the one appearing earliest in this case's instruments list. Printed order is
the left-to-right order of that list, after restricting it to the relevant
class. It is not alphabetical order, destination-code order or payment-ID order.

Destination codes name instruments only within their own case. For example,
D2 in one case and D2 in another case can name different instruments of different
types. The published code vocabulary is D1, D2, D3, D4, D5, D6, in that listing
order. This listing assigns no preference. Choose a code that is present in
that case's instruments list, and file the code rather than its payment-ID.

{scope}

SCHEDULE ({format_note})

{body}

File one line per case: the case_id, a comma, and the destination code, with no
header and no other text. File all cases in the order printed above.

Identifiers and destination codes are compared without regard to ASCII letter
case, after stripping outer whitespace and replacing each internal run of
whitespace with one space. Other characters outside printable ASCII become ?;
they are not transliterated. D 2 does not mean D2. Quotation marks and Markdown
fences are not removed. A comma line is split at its first comma; the entire
remaining text is its value. Extra commas therefore do not separate choices.
Blank lines are ignored. The first line naming a known case wins, even if its
value is empty or invalid. Later lines for that case are duplicates and do not
replace it. Unknown case IDs are extras and do not create scored rows. A known
case with no line is omitted. Empty, invalid and omitted destinations earn no
credit. If there are no commas anywhere, an exact count of {rows} nonempty lines is
read as destination codes in printed case order; any other comma-free filing
names no case. A wholly empty filing or one naming no known case earns zero.

The grade is the fraction of the {rows} printed cases with the correct destination,
with equal weight per case, rounded to six decimal places. A filing seals this
schedule. No verdict or score is returned at submission.
"""


# --------------------------------------------------------------------------
# the oracle template, and reading it back
# --------------------------------------------------------------------------

#: THE HEADING NAMES THE SCOPE BY THE SCHEDULES THE READER FILES.
#: A heading that scoped the rule to a word for the body of work it was drawn for was
#: read, and what it did was measured: of 119 readers given a correct statement of the
#: rule, 64 restated it correctly, wrote it into their own notes, and then filed the
#: held-out schedule under a different rule, because they had taken the scope word for
#: the organization printed at the top of the schedule and the two siblings never
#: carry one organization between them. A scope word a reader can map onto the surface
#: is therefore a scope word that names the wrong thing. This heading names the scope
#: by what the reader does, which is file this schedule and then the next one, and
#: says in the same breath that the schedule after those is scored under a convention
#: of its own, which leaves a convention worked out earlier somewhere to live. It is
#: ledger's heading with the domain noun kept and the claim untouched.
ORACLE_HEAD: tuple[str, ...] = (
    "REFUND CONVENTION FOR THIS SCHEDULE AND THE NEXT ONE YOU FILE (it",
    "governs the schedule above and the next schedule you are given,",
    "whichever organization it names; the schedule after that is scored",
    "under a convention of its own)",
    "Use the public eligibility, fallback, printed-order and tie rules.",
    "",
)

ROUTE_WORDS = {
    "route_gift": (
        "give the gift-card class unconditional priority over original non-gift "
        "contributors"
    ),
    "route_origin": (
        "give original non-gift contributors unconditional priority over the "
        "gift-card class"
    ),
    "route_usedgift": (
        "prefer the gift-card class when gift_used is yes, and prefer original "
        "non-gift contributors when gift_used is no"
    ),
}
GIFT_WORDS = {
    "gift_list": "take the earliest listed gift card without comparing balances",
    "gift_max": "take a gift card whose current balance is greatest",
    "gift_min": "take a gift card whose current balance is least",
}
ORIGIN_WORDS = {
    "origin_list": (
        "take the earliest listed original non-gift contributor without comparing "
        "contributions"
    ),
    "origin_max": (
        "take an original non-gift contributor whose purchase contribution is greatest"
    ),
    "origin_min": (
        "take an original non-gift contributor whose purchase contribution is least"
    ),
}

#: What each option means, in words. This is a declaration, not a renderer: the
#: package renders the oracle from it and reads one back with the same table, so the
#: sentence an oracle child reads and the rule an admission check believes it states
#: cannot come apart. No flattened phrase is contained in another, within an axis or
#: across the whole table, which is what makes each of the 27 bodies readable to
#: exactly one convention.
ORACLE_TEMPLATE = OracleTemplate(
    head=ORACLE_HEAD,
    sentences={
        "route_policy": "For each case, {}.",
        "gift_pick": "When the selected class is gift cards, {}.",
        "origin_pick": "When the selected class is original instruments, {}.",
    },
    phrases={
        "route_policy": ROUTE_WORDS,
        "gift_pick": GIFT_WORDS,
        "origin_pick": ORIGIN_WORDS,
    },
)


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class RetailRefundGenerator:
    """The retail refund genre, as the generator protocol wants it."""

    name = "retail_refund"
    genre = "economic destination choice"
    SHAPE = SHAPE
    AXES = AXES
    SCORING: str = ROW_ADDITIVE_EQUAL_WEIGHT
    #: A destination code is a token of a complete ordered vocabulary both tasks
    #: print, so the copy screen prices this family through the maps between two
    #: published orders. See `copy_profiles`.
    COPY_PROFILE: str = ORDERED_TOKENS
    #: The extra checks this family brings beyond the eleven every family runs. They
    #: are pinned code and not an operator-supplied flag, and they run wherever the
    #: eleven run: at materialization and at bundle verification.
    ADDITIONAL_CHECKS: tuple[str, ...] = (
        "retail_surface", "retail_support", "retail_profile_transfer",
        "retail_bijection",
    )
    #: EVERY CASE, verdict and correction, which is the receipt this genre was built
    #: and gated under and the one every number in its audit was computed on. It is
    #: declared here rather than assumed, because a generator that declares none is
    #: refused at registration.
    RECEIPT_POLICY: ReceiptPolicy = FULL_RECEIPT
    BLANK_TOKEN = BLANK_TOKEN
    UNFILED_TOKEN = UNFILED_TOKEN

    # ----- the instance -----

    def surface_for(self, ordinal: int, label: str) -> str:
        return SURFACES[(label.upper(), surface_index(ordinal, label))].name

    def surface_templates(self) -> tuple[str, ...]:
        return tuple(surface.name for surface in SURFACES.values())

    def build_table(self, master: bytes, ordinal: int, label: str) -> RetailTable:
        """This side of the pair the construction filter accepted for this ordinal."""
        a_table, b_table = build_pair(bytes(master), int(ordinal))
        return a_table if label.upper() == "A" else b_table

    def table_record(self, table: RetailTable) -> dict[str, Any]:
        """Every field of a retail schedule, as one canonical value.

        Item prices, instrument order, code-to-ID bindings and the printed body are
        all in it, because all of them are read between describing the task and
        rendering the receipt. A field that is read and not recorded is a field that
        can move between two rebuilds without the digest noticing.
        """
        return {
            "domain": table.domain,
            "rows": [
                {
                    "case_id": case.case_id,
                    "order_id": case.order_id,
                    "items": [
                        {
                            "item_id": item.item_id,
                            "name": item.name,
                            "price": item.price,
                        }
                        for item in case.items
                    ],
                    "refund_total": case.refund_total,
                    "gift_used": case.gift_used,
                    "instruments": [
                        {
                            "code": i.code,
                            "payment_method_id": i.payment_method_id,
                            "type": i.type,
                            "balance": i.balance,
                            "contribution": i.contribution,
                        }
                        for i in case.instruments
                    ],
                }
                for case in table.rows
            ],
            "body": table.body,
        }

    def build_envelope(self, master: bytes, ordinal: int) -> Envelope:
        """The registered envelope, with this instance's committed filler and neutrals.

        The coordinates name this generator, so a bank of one genre and a bank of
        another under one master key do not share a filler stream.
        """
        neutral: dict[str, tuple[str, ...]] = {}
        for spec in SLOTS:
            neutral[spec.name] = tuple(
                streams.filler_stream(
                    master, FILLER_ALPHABET, spec.width,
                    "retail_refund", ordinal, "neutral", spec.name, row,
                )
                for row in range(ROWS)
            )
        return Envelope(
            size=ENVELOPE_SIZE,
            identifier_width=IDENTIFIER_WIDTH,
            observed_width=OBSERVED_WIDTH,
            slots=SLOTS,
            filler=streams.filler_stream(
                master, FILLER_ALPHABET, ENVELOPE_SIZE, "retail_refund", ordinal, "pad"
            ),
            column_titles=COLUMN_TITLES,
            neutral=neutral,
        )

    def key_for(
        self, table: RetailTable, convention: Mapping[str, str]
    ) -> tuple[str, ...]:
        return key_for(table, convention)

    def normalize_answer(self, value: str) -> str:
        # The same function the scorer compares with, so the two cannot drift.
        return shared_filing.fold(value)

    def row_identifiers(self, table: RetailTable) -> tuple[str, ...]:
        return tuple(case.case_id for case in table.rows)

    def row_classes(self, table: RetailTable) -> tuple[str, ...]:
        """The four public case kinds, derived from what the schedule prints."""
        return tuple(case_kind(case) for case in table.rows)

    def answer_ranks(self, table: RetailTable) -> tuple[str, ...]:
        """The COMPLETE published vocabulary, never the codes one draw realizes.

        Both task texts print D1 through D6 in this order, so a reader with no idea
        what the rule is can read both orders off the two texts and map first to
        first. Returning the codes B's drawn key happened to realize would price a
        transfer nobody can perform and would move with the draw.
        """
        return CODES

    def row_label(self, table: RetailTable) -> tuple[str, ...]:
        # Every row names the case it grades. Nothing on this receipt names an axis.
        return (ROW_LABEL,) * len(table.rows)

    # ----- reading the filing -----

    def parse_and_canonicalize(self, task: Task, raw: object) -> Filing:
        """One canonical value per printed case, by the shared registered rules.

        The reading is the one in `filing.py`: a line is a case id, a comma and a
        destination code; identifiers match without case after whitespace is
        collapsed; the first line for a case wins; a comma-free filing is read
        positionally only when it has exactly one line per printed case. A known case
        with a value that is not a legal code stays filed and wrong, because failing
        to read it would turn a bad answer into no answer.
        """
        return shared_filing.parse_rows(self.row_identifiers(task.table), raw)

    def score(self, task: Task, canonical: Filing) -> tuple[float, tuple[RowOutcome, ...]]:
        """The sealed scalar and what the filing did on every case.

        Equal weight per case, the denominator is the printed case count, rounded to
        six places. A case counts only if it was FILED: no correct destination is ever
        the empty string, and the distinction between a filed empty value and an
        omission is what the receipt prints, so the scorer keeps it too.
        """
        identifiers = self.row_identifiers(task.table)
        truth = task.key if task.key else ("",) * len(identifiers)
        return shared_filing.score_rows(
            identifiers, truth, canonical, self.normalize_answer
        )

    # ----- the task text -----

    def describe(self, task: PublicTask) -> str:
        """The public rules, which schedules share a convention, and the schedule.

        It takes the PUBLIC task, so there is no argument here the drawn rule could
        arrive through, and the scope sentence is chosen by the sibling label and by
        nothing else: the same bytes go to every arm of a fork.
        """
        table: RetailTable = task.table
        surface = table.template
        return TASK_TEMPLATE.format(
            organization=surface.organization,
            scope=scope_sentence(task.label),
            format_note=surface.format_note,
            body=table.body,
            rows=len(table.rows),
        )

    # ----- the three cells -----

    def render_receipt(
        self, task: Task, canonical: Filing, truth: Sequence[str], feedback: Feedback
    ) -> ReceiptAST:
        """One verdict per printed case, on what the filing did.

        The rows are built by the shared grader from the scorer's own outcomes, so
        what a row says is not a choice this module gets to make. There is no axis
        name, no selected preference, no class heading, no option summary, no refund
        explanation and no sibling answer: a failed row carries that same row's own
        correct code, and a passed row carries nothing. A code by itself is not a
        selector label, because its payment binding is different in every case.
        """
        graded = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text=task.text, key=tuple(truth), mask=task.mask,
        )
        _, outcomes = self.score(graded, canonical)
        return graded_receipt(
            task.task_id, outcomes, BLANK_TOKEN, UNFILED_TOKEN, feedback
        )

    def render_placebo(
        self, task: PublicTask, canonical: Filing, envelope: Envelope
    ) -> ReceiptAST:
        """The inert cell: congruent with the graded one outside the registered slots.

        It takes the PUBLIC task and the envelope, so there is no argument here
        through which the hidden rule could reach it. Every byte of both slots on
        every row is the committed neutral token, including the empty correction of a
        row that would have passed; the case id and the echoed filing are not
        neutralized.
        """
        blind = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text="", key=(), mask=task.mask,
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
        """The drawn convention, from the declared phrases, with no rows to align."""
        return render_oracle_cell(ORACLE_TEMPLATE, task_id, convention, row_count)

    def parse_oracle(self, ast: ReceiptAST) -> dict[str, str]:
        return parse_oracle_cell(ORACLE_TEMPLATE, ast)

    def check_additional(self, name: str, instance, master: bytes):
        """Run one of this family's declared extra checks, by name.

        Imported at call time so the code pin's import walk reaches the validation
        module through this generator rather than through a manual hash exemption.
        """
        from shogym.envs.receipts.generators import retail_validation as validation

        return validation.run_named(name, self, instance, master)


GENERATOR = RetailRefundGenerator()


__all__ = [
    "ALL_CONVENTIONS",
    "AXES",
    "A_BLOCK",
    "A_CLASS_SIZE",
    "BLANK_TOKEN",
    "B_CLASS_SIZE",
    "BOTH_GIFT_PURCHASE",
    "BOTH_NO_GIFT_PURCHASE",
    "B_BLOCK",
    "B_GIFT_CODES",
    "B_ORIGIN_CODES",
    "CODES",
    "CREDIT_CARD",
    "ENVELOPE_SIZE",
    "GIFTS_ONLY",
    "GIFT_BALANCES",
    "GIFT_CARD",
    "GIFT_CONTRIBUTIONS",
    "GENERATOR",
    "HEADER",
    "KINDS",
    "MAX_ATTEMPTS",
    "NON_GIFT_TYPES",
    "ORACLE_TEMPLATE",
    "ORIGINALS_ONLY",
    "ORIGINAL_AMOUNTS",
    "PAYPAL",
    "PER_KIND",
    "PER_ORIENTATION",
    "ROWS",
    "SHAPE",
    "SLOTS",
    "SURFACES",
    "UNFILED_TOKEN",
    "RetailCase",
    "RetailInstrument",
    "RetailItem",
    "RetailRefundGenerator",
    "RetailTable",
    "Surface",
    "build_pair",
    "build_side",
    "case_kind",
    "destination",
    "gift_class",
    "gift_purchase",
    "key_for",
    "original_class",
    "profile_plan",
    "render_body",
    "surface_index",
]
