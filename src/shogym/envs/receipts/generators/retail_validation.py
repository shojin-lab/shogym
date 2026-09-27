"""A second implementation of the retail rule, and the predicates a pair is filtered on.

WHY A SECOND IMPLEMENTATION. `judge_cells` compares what the renderer printed with
what the scorer computed, so it catches a renderer that disagrees with the scorer and
cannot catch a scorer that is wrong: a selector that compares a gift's contribution
instead of its balance, or lets a purchasing gift into both classes, or breaks a tie by
the last printed member, is carried identically into the receipt, the placebo, the key
and the gate, and every named check passes on it. The serializer can also fit an
overwide field by truncating it rather than refusing it, so a correction longer than
its slot becomes a shorter legal looking answer. Both are mistakes an author makes and
cannot see, and the only thing that finds them is another implementation that was not
written from the first.

So this module never imports the production selector and never reads the production
table's stored amounts to compute an answer. It reads the PRINTED BODY back, which is
the only thing the agent ever sees, and it derives the operation each option names from
the option's own identifier rather than from the production table: `route_usedgift`
routes by the purchase flag and `gift_min` takes a least balance, so an edited
production table is a disagreement here rather than a change both sides make together.
Its selectors are explicit stable sorts; the production selector is a keyed minimum.

THE PREDICATES ARE PURE AND CONVENTION FREE. `pair_refusal` takes two tables and
returns a reason or the empty string. It never draws, never sees the live convention,
and quantifies over the whole support, which is what makes acceptance a function of the
two schedules alone: a filter that read the live draw would make rejection informative,
and an agent that knew which pairs are accepted could exclude conventions from the fact
that this pair was served.

WHAT THE COPY NUMBERS HERE ARE. Two families, and they are not interchangeable. The
SHIPPED family is the registered one the copy screen prices: the six rotations of the
published code order composed with the 48 row moves, and its 0.50 bar is the registered
bar. The ALL-BIJECTION family is larger: every one of the 720 relabellings of the six
codes composed with the same 48 row moves, computed here by a six-by-six match-count
matrix and exact assignment optimization. Its maximum is reported separately and is
never relabelled as the registered closure.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from math import comb
from typing import Any, Mapping, Sequence

import numpy as np

from shogym.envs.receipts.checks import (
    CheckResult,
    check_copy,
    check_exercise,
    check_materiality,
)
from shogym.envs.receipts.generators.retail_refund import (
    A_CLASS_SIZE,
    ALL_CONVENTIONS,
    AXES,
    B_CLASS_SIZE,
    BOTH_GIFT_PURCHASE,
    BOTH_NO_GIFT_PURCHASE,
    CENTS,
    CODES,
    CREDIT_CARD,
    GIFTS_ONLY,
    GIFT_CARD,
    HEADER,
    ORIGINALS_ONLY,
    PAYPAL,
    PER_KIND,
    PER_ORIENTATION,
    ROWS,
    RetailCase,
    RetailTable,
)
from shogym.envs.receipts.observe import observe
from shogym.envs.receipts.protocol import FULL_RECEIPT, Instance, Task, policy_of
from shogym.envs.receipts.receipt_ast import (
    GAP,
    ORDINAL_WIDTH,
    frozen_envelope,
    row_lines,
    serialize,
)
from shogym.envs.receipts.receipt_law import (
    REGISTERED_MAX_IDEAL,
    REGISTERED_MIN_DISTINGUISHING,
    REGISTERED_MIN_IDEAL,
    REGISTERED_MIN_LAW_ROOM,
    law_for,
)
from shogym.envs.receipts.render import feedback_for

# ----- the registered bars this module reads ---------------------------------
#
# The construction bars, which quantify over the whole support rather than under the
# drawn convention, and the two copying bounds.

#: Rows one option change has to move on A and on B, minimized over every draw and
#: every alternative option. They are the specification's own movement minima.
MIN_MOVED_A = {"route_policy": 6, "gift_pick": 3, "origin_pick": 3}
MIN_MOVED_B = {"route_policy": 6, "gift_pick": 6, "origin_pick": 6}

#: The registered copy bar, read against the SHIPPED family: the six rotations of the
#: published code order composed with the 48 row moves.
MAX_SHIPPED_COPY = 0.50
#: The bar the review asks the wider family to be qualified at. It is a separate
#: registration over a larger family of maps and it inherits none of the shipped
#: family's calibration. A number measured against one family means nothing held
#: against the other, which is why the two are never reported as one.
MAX_BIJECTION_COPY = 0.50
#: The registered one-axis-wrong and leverage bars, asserted over the whole support
#: here rather than only under the drawn convention.
MAX_FLIP = 0.875
MIN_LEVERAGE = 0.10

#: What counts as equal when a registered bar is read against a floating-point mean.
TOLERANCE = 1e-9

#: The printed field bounds the schedule promises. Each is a maximum printed size in
#: ASCII bytes, and each is what a width in the receipt or a column in the body was
#: registered against.
MAX_CASE_ID = 10
MAX_ORDER_ID = 9
MAX_ITEM_ID = 10
MAX_ITEM_NAME = 7
MAX_PRICE = 36000
MAX_MONEY_FIELD = 5
MAX_PAYMENT_ID = 19
MAX_TYPE = 11
MAX_CODE = 2

#: The eight joint profiles A's recipe prints, stated here rather than imported, so a
#: constructor that paired its orientations some other way is a disagreement between
#: two statements of the recipe and not a table validated against itself. Each entry
#: is a public kind and the orientation of each class it carries, where True means the
#: smaller amount prints first and None means the case does not carry that class.
#:
#: WHY THE PAIRING IS THE THING. Counting gift orientations and original orientations
#: separately accepts a mixed case with a smaller-first gift class and a larger-first
#: original class, as long as another case pays it back, so the recipe's pairing
#: survives in the two tallies and is gone from the table. The pairing is what it
#: takes to separate the two selectors: a gift-purchase case reads the same
#: orientation on both classes and a mixed case without a gift purchase reads opposite
#: ones, so the first-listed member and one extremum agree on one of them and disagree
#: on the other.
A_JOINT_PROFILES: tuple[tuple[str, bool | None, bool | None], ...] = (
    (GIFTS_ONLY, True, None),
    (GIFTS_ONLY, False, None),
    (ORIGINALS_ONLY, None, True),
    (ORIGINALS_ONLY, None, False),
    (BOTH_GIFT_PURCHASE, True, True),
    (BOTH_GIFT_PURCHASE, False, False),
    (BOTH_NO_GIFT_PURCHASE, True, False),
    (BOTH_NO_GIFT_PURCHASE, False, True),
)

NON_GIFT = (CREDIT_CARD, PAYPAL)
AXIS_NAMES = tuple(axis.name for axis in AXES)
#: Every permutation of the six codes, as flat indices into a six-by-six match-count
#: matrix. Built once: 720 rows of 6, which is what makes the assignment optimum a
#: gather and a row sum rather than a search.
_ASSIGNMENTS = np.array(
    [[x * 6 + p[x] for x in range(6)] for p in itertools.permutations(range(6))],
    dtype=np.int64,
)
_CODE_INDEX = {code: n for n, code in enumerate(CODES)}


# --------------------------------------------------------------------------
# the second rule, derived from the option identifiers
# --------------------------------------------------------------------------


def prefers_gift(option: str, gift_used: bool) -> bool:
    """Which class the route names, read out of the option's own identifier."""
    which = option.split("_", 1)[1]
    if which == "gift":
        return True
    if which == "origin":
        return False
    if which == "usedgift":
        return gift_used
    raise ValueError(f"no route is named by {option!r}")


def selector_of(option: str) -> str:
    """Which operation a selector option names: the first listed, greatest or least."""
    which = option.split("_", 1)[1]
    if which in ("list", "max", "min"):
        return which
    raise ValueError(f"no selector is named by {option!r}")


def audit_destination(
    instruments: Sequence[Mapping[str, Any]], convention: Mapping[str, str]
) -> str:
    """The destination one printed case takes, by explicit stable sorting.

    `instruments` is a parsed case, in printed order. A stable sort on the amount, or
    on its negation for a greatest, leaves the earliest printed member first among
    equals, which is the public tie rule reached by a different route from the
    production selector's keyed minimum.
    """
    gifts = [i for i in instruments if i["type"] == GIFT_CARD]
    originals = [
        i for i in instruments if i["type"] in NON_GIFT and i["contribution"] > 0
    ]
    used = any(i["contribution"] > 0 for i in gifts)
    gift_side = prefers_gift(convention["route_policy"], used)
    members = gifts if gift_side else originals
    if not members:
        gift_side = not gift_side
        members = gifts if gift_side else originals
    if not members:
        raise ValueError("a case lists no eligible destination at all")
    field = "balance" if gift_side else "contribution"
    option = convention["gift_pick"] if gift_side else convention["origin_pick"]
    which = selector_of(option)
    if which == "list":
        return str(members[0]["code"])
    if which == "max":
        return str(sorted(members, key=lambda i: -int(i[field] or 0))[0]["code"])
    return str(sorted(members, key=lambda i: int(i[field] or 0))[0]["code"])


def audit_key(
    cases: Sequence[Sequence[Mapping[str, Any]]], convention: Mapping[str, str]
) -> tuple[str, ...]:
    """Every parsed case's destination under one convention, in printed order."""
    return tuple(audit_destination(case, convention) for case in cases)


# --------------------------------------------------------------------------
# reading the printed body back
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedCase:
    """One case as the printed body states it, with money already in integer cents."""

    case_id: str
    order_id: str
    items: tuple[dict[str, Any], ...]
    refund_total: int
    gift_used: str
    instruments: tuple[dict[str, Any], ...]


def _cents(printed: object) -> int:
    """One printed two-place amount as integer cents, refused rather than rounded."""
    text = str(printed)
    whole, point, fraction = text.partition(".")
    if not point or len(fraction) != 2 or not whole.isdigit() or not fraction.isdigit():
        raise ValueError(f"{text!r} is not a two-place amount")
    return int(whole) * CENTS + int(fraction)


def parse_printed_body(body: str, json_body: bool) -> tuple[ParsedCase, ...]:
    """The schedule a reader sees, read back into records without the stored table.

    This is the independent half of the validator: everything downstream of here is
    computed from what the agent would read, so a stored value the body does not print
    and a printed value the store does not hold are both disagreements.
    """
    lines = body.split("\n")
    if not json_body:
        if not lines or lines[0] != " | ".join(HEADER):
            raise ValueError("the pipe body does not open with the registered header")
        lines = lines[1:]
    out: list[ParsedCase] = []
    for line in lines:
        record: dict[str, Any]
        if json_body:
            record = dict(json.loads(line))
            if list(record) != list(HEADER):
                raise ValueError(f"a JSON case names {list(record)}")
        else:
            fields = line.split(" | ")
            if len(fields) != len(HEADER):
                raise ValueError(f"a pipe case prints {len(fields)} columns")
            record = dict(zip(HEADER, fields))
            record["items"] = json.loads(record["items"])
            record["instruments"] = json.loads(record["instruments"])
        out.append(
            ParsedCase(
                case_id=str(record["case_id"]),
                order_id=str(record["order_id"]),
                items=tuple(
                    {
                        "item_id": str(item["item_id"]),
                        "name": str(item["name"]),
                        "price": _cents(item["price"]),
                    }
                    for item in record["items"]
                ),
                refund_total=_cents(record["refund_total"]),
                gift_used=str(record["gift_used"]),
                instruments=tuple(
                    {
                        "code": str(i["code"]),
                        "payment_method_id": str(i["payment_method_id"]),
                        "type": str(i["type"]),
                        "balance": None if i["balance"] is None else _cents(i["balance"]),
                        "contribution": _cents(i["contribution"]),
                    }
                    for i in record["instruments"]
                ),
            )
        )
    return tuple(out)


def parsed_cases(table: RetailTable) -> tuple[ParsedCase, ...]:
    """One table's printed body, read back."""
    return parse_printed_body(table.body, table.template.json_body)


# --------------------------------------------------------------------------
# what a printed schedule has to be
# --------------------------------------------------------------------------


def parsed_kind(case: ParsedCase) -> str:
    """The public case kind, derived from the printed instruments alone."""
    gifts = [i for i in case.instruments if i["type"] == GIFT_CARD]
    originals = [
        i for i in case.instruments if i["type"] in NON_GIFT and i["contribution"] > 0
    ]
    if gifts and not originals:
        return GIFTS_ONLY
    if originals and not gifts:
        return ORIGINALS_ONLY
    used = any(i["contribution"] > 0 for i in gifts)
    return BOTH_GIFT_PURCHASE if used else BOTH_NO_GIFT_PURCHASE


def _classes(case: ParsedCase) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gifts = [i for i in case.instruments if i["type"] == GIFT_CARD]
    originals = [
        i for i in case.instruments if i["type"] in NON_GIFT and i["contribution"] > 0
    ]
    return gifts, originals


def case_refusal(case: ParsedCase, label: str) -> str:
    """Why this printed case is not one, or the empty string.

    Everything a reader is entitled to rely on inside a single case: the field bounds,
    the code binding, the class partition, the contribution totals, the purchase flag,
    eligibility and the side's class cardinality. It is separate from the table-level
    predicate so a hand fixture of one case can be held against it.
    """
    side = label.upper()
    size = A_CLASS_SIZE if side == "A" else B_CLASS_SIZE
    prefix = "CA" if side == "A" else "CB"
    if not case.case_id.startswith(prefix) or len(case.case_id) != MAX_CASE_ID:
        return f"side {side} prints the case id {case.case_id!r}"
    if len(case.order_id) > MAX_ORDER_ID or not case.order_id.startswith("#W"):
        return f"side {side} prints the order id {case.order_id!r}"
    if case.gift_used not in ("yes", "no"):
        return f"case {case.case_id} prints the flag {case.gift_used!r}"
    if not case.items or len(case.items) > 2:
        return f"case {case.case_id} prints {len(case.items)} items"
    for item in case.items:
        if len(item["item_id"]) != MAX_ITEM_ID:
            return f"case {case.case_id} prints the item id {item['item_id']!r}"
        if len(item["name"]) > MAX_ITEM_NAME or not item["name"]:
            return f"case {case.case_id} prints the item name {item['name']!r}"
        if not 0 < item["price"] <= MAX_PRICE:
            return f"case {case.case_id} prints the price {item['price']}"
    if sum(item["price"] for item in case.items) != case.refund_total:
        return f"case {case.case_id} prints items that do not sum to its refund"
    codes = [i["code"] for i in case.instruments]
    if len(set(codes)) != len(codes):
        return f"case {case.case_id} binds one code to two instruments"
    if any(code not in CODES for code in codes):
        return f"case {case.case_id} prints a code outside the published six"
    if len({code.strip().lower() for code in codes}) != len(codes):
        return f"case {case.case_id} prints two destinations that normalize alike"
    gifts, originals = _classes(case)
    for instrument in case.instruments:
        if instrument["type"] not in (GIFT_CARD, *NON_GIFT):
            return f"case {case.case_id} prints the type {instrument['type']!r}"
        if len(instrument["payment_method_id"]) > MAX_PAYMENT_ID:
            return f"case {case.case_id} prints an over-long payment id"
        if len(instrument["type"]) > MAX_TYPE or len(instrument["code"]) != MAX_CODE:
            return f"case {case.case_id} prints a field outside its registered width"
        if instrument["contribution"] < 0:
            return f"case {case.case_id} prints a negative contribution"
        if instrument["type"] == GIFT_CARD:
            if instrument["balance"] is None:
                return f"case {case.case_id} prints a gift card with no balance"
            if len("%d.%02d" % divmod(instrument["balance"], CENTS)) > MAX_MONEY_FIELD:
                return f"case {case.case_id} prints an over-wide balance"
        elif instrument["balance"] is not None:
            return f"case {case.case_id} prints a balance on a non-gift instrument"
        if not instrument["payment_method_id"].startswith(instrument["type"] + "_"):
            return f"case {case.case_id} binds a payment id to another type"
    if sum(i["contribution"] for i in case.instruments) != case.refund_total:
        return f"case {case.case_id} prints contributions that do not sum to its refund"
    if case.refund_total <= 0:
        return f"case {case.case_id} prints a refund of {case.refund_total}"
    used = any(i["contribution"] > 0 for i in gifts)
    if (case.gift_used == "yes") != used:
        return f"case {case.case_id} prints a purchase flag its contributions deny"
    if not gifts and not originals:
        return f"case {case.case_id} prints no eligible destination"
    if any(i["contribution"] <= 0 for i in case.instruments if i["type"] in NON_GIFT):
        return f"case {case.case_id} lists a non-contributing original instrument"
    for members in (gifts, originals):
        if members and len(members) != size:
            return (
                f"case {case.case_id} prints a class of {len(members)} against "
                f"side {side}'s {size}"
            )
    return ""


def body_refusal(cases: Sequence[ParsedCase], label: str) -> str:
    """Why this printed schedule is not one, or the empty string.

    Each case on its own, then what only the whole table can say: that no identifier
    repeats across it, that the four public kinds appear the registered number of
    times, and that the side's amount ordering law holds, which on A is the eight
    joint profiles rather than a tally of each class on its own.
    """
    side = label.upper()
    if len(cases) != ROWS:
        return f"side {side} prints {len(cases)} cases against the registered {ROWS}"
    seen: set[str] = set()
    counted: dict[str, int] = {}
    for case in cases:
        refusal = case_refusal(case, side)
        if refusal:
            return refusal
        named = (
            [case.case_id, case.order_id]
            + [item["item_id"] for item in case.items]
            + [i["payment_method_id"] for i in case.instruments]
        )
        if seen & set(named) or len(set(named)) != len(named):
            return f"side {side} repeats an identifier at case {case.case_id}"
        seen |= set(named)
        kind = parsed_kind(case)
        counted[kind] = counted.get(kind, 0) + 1
    for kind in (GIFTS_ONLY, ORIGINALS_ONLY, BOTH_GIFT_PURCHASE, BOTH_NO_GIFT_PURCHASE):
        if counted.get(kind, 0) != PER_KIND:
            return (
                f"side {side} prints {counted.get(kind, 0)} cases of kind {kind} "
                f"against the registered {PER_KIND}"
            )
    return _orientation_refusal(cases, side)


def _joint_profile(case: ParsedCase) -> tuple[str, bool | None, bool | None] | str:
    """One A case's public kind and the orientation of each class it carries.

    True means the class prints its smaller amount first, and None means the case does
    not carry that class. A tie inside a class has no orientation at all, so it comes
    back as a refusal rather than as a profile.
    """
    gifts, originals = _classes(case)
    marks: list[bool | None] = []
    for name, members, field in (
        ("gift", gifts, "balance"), ("original", originals, "contribution")
    ):
        if len(members) != A_CLASS_SIZE:
            marks.append(None)
            continue
        first, second = (int(m[field] or 0) for m in members)
        if first == second:
            return f"case {case.case_id} prints a tie in its {name} class"
        marks.append(first < second)
    return (parsed_kind(case), marks[0], marks[1])


def _profile_words(profile: tuple[str, bool | None, bool | None]) -> str:
    """One joint profile in words, for a refusal a reader can act on."""
    kind, gift, origin = profile
    said = [kind]
    for name, mark in (("gift", gift), ("original", origin)):
        if mark is not None:
            said.append(f"{name} {'smaller' if mark else 'larger'} first")
    return ", ".join(said)


def _orientation_refusal(cases: Sequence[ParsedCase], side: str) -> str:
    """A's eight joint profiles, and B's median-first ordering."""
    if side == "A":
        counted: dict[tuple[str, bool | None, bool | None], int] = {}
        for case in cases:
            profile = _joint_profile(case)
            if isinstance(profile, str):
                return profile
            counted[profile] = counted.get(profile, 0) + 1
        for profile in sorted(A_JOINT_PROFILES, key=str):
            if counted.get(profile, 0) != PER_ORIENTATION:
                return (
                    f"side A prints {counted.get(profile, 0)} cases of the joint "
                    f"profile ({_profile_words(profile)}), not {PER_ORIENTATION}"
                )
        outside = sorted(set(counted) - set(A_JOINT_PROFILES), key=str)
        if outside:
            return (
                f"side A prints the joint profile ({_profile_words(outside[0])}), "
                "which the recipe does not pair"
            )
        return ""
    for case in cases:
        gifts, originals = _classes(case)
        for name, members, field in (
            ("gift", gifts, "balance"), ("original", originals, "contribution")
        ):
            if len(members) != B_CLASS_SIZE:
                continue
            values = [int(m[field] or 0) for m in members]
            if len(set(values)) != B_CLASS_SIZE:
                return f"case {case.case_id} prints two equal amounts in its {name} class"
            if values[0] != sorted(values)[1]:
                return (
                    f"case {case.case_id} does not print its median {name} amount first"
                )
    return ""


# --------------------------------------------------------------------------
# what the two schedules have to be together
# --------------------------------------------------------------------------


def fingerprint(case: ParsedCase) -> tuple:
    """One case's code-free structural position, as a reader could match it.

    The ordered class of each printed instrument, the weak rank of each class's
    amounts inside its own class, the purchase flag and the two class sizes. No code,
    no identifier, no magnitude: two cases with the same fingerprint are two cases a
    reader could answer by reusing the other's corrected destination position.
    """
    gifts, originals = _classes(case)

    def ranks(members: Sequence[Mapping[str, Any]], field: str) -> tuple[int, ...]:
        values = [int(m[field] or 0) for m in members]
        return tuple(sorted(values).index(v) for v in values)

    return (
        tuple(i["type"] == GIFT_CARD for i in case.instruments),
        ranks(gifts, "balance"),
        ranks(originals, "contribution"),
        case.gift_used,
        len(gifts),
        len(originals),
    )


def _coded(key: Sequence[str]) -> np.ndarray:
    return np.array([_CODE_INDEX[value] for value in key], dtype=np.int64)


def _row_moves(values: np.ndarray) -> np.ndarray:
    """The registered row moves: every rotation of the sequence and of its reversal."""
    n = len(values)
    out = np.empty((2 * n, n), dtype=np.int64)
    for j, base in enumerate((values, values[::-1])):
        for shift in range(n):
            out[j * n + shift] = np.roll(base, -shift)
    return out


def shipped_copy_maximum(source: Sequence[str], target: Sequence[str]) -> float:
    """The best the SHIPPED family earns: six code rotations by 48 row moves.

    This is the registered closure the 0.50 bar is read against, computed directly on
    the two keys rather than through the parser, so a disagreement with the shared
    screen is a disagreement about the family and not about the reading.
    """
    moved = _row_moves(_coded(source))
    best = 0
    for shift in range(len(CODES)):
        hits = ((moved + shift) % len(CODES) == _coded(target)[None, :]).sum(axis=1)
        best = max(best, int(hits.max()))
    return best / float(len(target))


def bijection_copy_maximum(source: Sequence[str], target: Sequence[str]) -> float:
    """The best the ALL-BIJECTION family earns: 720 relabellings by 48 row moves.

    Exact, not sampled: for each row move the six-by-six match-count matrix is built
    and every one of the 720 assignments is evaluated, so the reported number is a
    maximum some map in the family actually reaches. It is a SEPARATE registration
    from the shipped closure above and is never relabelled as it.
    """
    coded_target = _coded(target)
    best = 0
    for moved in _row_moves(_coded(source)):
        counts = np.bincount(moved * len(CODES) + coded_target, minlength=36)
        best = max(best, int(counts[_ASSIGNMENTS].sum(axis=1).max()))
    return best / float(len(target))


def support_keys(cases: Sequence[Sequence[Mapping[str, Any]]]) -> dict[tuple, tuple[str, ...]]:
    """Every convention's key for one parsed side, by the second implementation."""
    return {
        tuple(sorted(convention.items())): audit_key(cases, convention)
        for convention in ALL_CONVENTIONS
    }


def movement(keys: Mapping[tuple, tuple[str, ...]]) -> dict[str, int]:
    """Per axis, the fewest rows one option change moves anywhere in the support."""
    out: dict[str, int] = {}
    for axis in AXES:
        least = len(next(iter(keys.values())))
        for convention in ALL_CONVENTIONS:
            base = keys[tuple(sorted(convention.items()))]
            for option in axis.options:
                if option == convention[axis.name]:
                    continue
                alt = dict(convention, **{axis.name: option})
                other = keys[tuple(sorted(alt.items()))]
                least = min(least, sum(1 for x, y in zip(base, other) if x != y))
        out[axis.name] = least
    return out


def support_flip_and_leverage(keys: Mapping[tuple, tuple[str, ...]]) -> tuple[float, float]:
    """The worst one-axis-wrong score and the weakest axis leverage over the support."""
    worst_flip = 0.0
    weakest = 1.0
    width = float(len(next(iter(keys.values()))))
    for convention in ALL_CONVENTIONS:
        base = keys[tuple(sorted(convention.items()))]
        for axis in AXES:
            agreed: list[float] = []
            for option in axis.options:
                if option == convention[axis.name]:
                    continue
                alt = keys[tuple(sorted(dict(convention, **{axis.name: option}).items()))]
                agreed.append(sum(1 for x, y in zip(base, alt) if x == y) / width)
            worst_flip = max(worst_flip, max(agreed))
            weakest = min(weakest, 1.0 - sum(agreed) / len(agreed))
    return worst_flip, weakest


def _printed_fields(case: ParsedCase) -> tuple[tuple[str, object], ...]:
    """One printed case as named values, in a fixed order, position included."""
    out: list[tuple[str, object]] = [
        ("case_id", case.case_id),
        ("order_id", case.order_id),
        ("refund_total", case.refund_total),
        ("gift_used", case.gift_used),
        ("item count", len(case.items)),
        ("instrument count", len(case.instruments)),
    ]
    for n, item in enumerate(case.items):
        for field in ("item_id", "name", "price"):
            out.append((f"item {n}'s {field}", item[field]))
    for n, instrument in enumerate(case.instruments):
        for field in ("code", "payment_method_id", "type", "balance", "contribution"):
            out.append((f"instrument {n}'s {field}", instrument[field]))
    return tuple(out)


def _stored_fields(case: RetailCase) -> tuple[tuple[str, object], ...]:
    """The same named values, taken from the record the schedule holds."""
    out: list[tuple[str, object]] = [
        ("case_id", case.case_id),
        ("order_id", case.order_id),
        ("refund_total", case.refund_total),
        ("gift_used", "yes" if case.gift_used else "no"),
        ("item count", len(case.items)),
        ("instrument count", len(case.instruments)),
    ]
    for n, item in enumerate(case.items):
        out.append((f"item {n}'s item_id", item.item_id))
        out.append((f"item {n}'s name", item.name))
        out.append((f"item {n}'s price", item.price))
    for n, instrument in enumerate(case.instruments):
        out.append((f"instrument {n}'s code", instrument.code))
        out.append((f"instrument {n}'s payment_method_id", instrument.payment_method_id))
        out.append((f"instrument {n}'s type", instrument.type))
        out.append((f"instrument {n}'s balance", instrument.balance))
        out.append((f"instrument {n}'s contribution", instrument.contribution))
    return tuple(out)


def record_refusal(table: RetailTable, label: str) -> str:
    """Why the printed schedule is not the stored one, field by field.

    THE KEYS AGREEING IS NOT THE RECORDS AGREEING. A renderer that prints a gift
    balance or a contribution the record does not hold, and does not move that
    amount's rank inside its class, leaves every destination and all 27 keys where
    they were: the independent selector reading the body and the production scorer
    reading the store still agree, so comparing keys sees nothing and the agent
    answers a schedule the bank did not commit. This compares the complete parsed
    record with the complete stored record, every item, instrument, amount, flag,
    identifier and position, rather than the answers they lead to.
    """
    side = label.upper()
    try:
        cases = parsed_cases(table)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return f"side {side}'s printed body does not read back: {exc}"
    if len(cases) != len(table.rows):
        return (
            f"side {side} prints {len(cases)} cases where the stored schedule holds "
            f"{len(table.rows)}"
        )
    for printed, held in zip(cases, table.rows):
        for (name, shown), (_, stored) in zip(
            _printed_fields(printed), _stored_fields(held)
        ):
            if shown != stored:
                return (
                    f"side {side} prints {name} as {shown!r} at case {held.case_id}, "
                    f"where the stored case holds {stored!r}"
                )
    return ""


def side_refusal(table: RetailTable, label: str) -> str:
    """Why one schedule is not admissible on its own, or the empty string."""
    try:
        cases = parsed_cases(table)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return f"side {label.upper()}'s printed body does not read back: {exc}"
    refusal = body_refusal(cases, label)
    if refusal:
        return refusal
    instruments = [case.instruments for case in cases]
    keys = support_keys(instruments)
    if len(set(keys.values())) != len(ALL_CONVENTIONS):
        return (
            f"side {label.upper()} gives two of the {len(ALL_CONVENTIONS)} conventions "
            "the same complete key"
        )
    for key in keys.values():
        if set(key) != set(CODES):
            return f"side {label.upper()} has a key that does not use all six codes"
    bars = MIN_MOVED_A if label.upper() == "A" else MIN_MOVED_B
    moved = movement(keys)
    for axis, least in sorted(moved.items()):
        if least < bars[axis]:
            return (
                f"side {label.upper()} moves {least} rows on {axis} somewhere in the "
                f"support, under the registered {bars[axis]}"
            )
    return ""


def pair_refusal(a_table: RetailTable, b_table: RetailTable) -> str:
    """Why this pair of schedules may not be served, or the empty string.

    Convention free and quantified over the whole support, so acceptance is a
    function of the two schedules alone.

    BOTH COPYING FAMILIES ARE BOUNDS HERE. The shipped closure and the wider
    all-bijection family are separate registrations at separate bars, and each is
    asserted under every convention. A pair that cleared one and not the other used to
    be admitted and reported; a bar that is reported beside a passing check is not a
    bar, so it is applied where the pair is accepted.
    """
    for table, label in ((a_table, "A"), (b_table, "B")):
        refusal = side_refusal(table, label)
        if refusal:
            return refusal
    a_cases = parsed_cases(a_table)
    b_cases = parsed_cases(b_table)
    if {c.case_id for c in a_cases} & {c.case_id for c in b_cases}:
        return "the two schedules share a case id"
    shared = {fingerprint(c) for c in a_cases} & {fingerprint(c) for c in b_cases}
    if shared:
        return "a case of each schedule has the same code-free structural fingerprint"
    a_keys = support_keys([c.instruments for c in a_cases])
    b_keys = support_keys([c.instruments for c in b_cases])
    flip, leverage = support_flip_and_leverage(b_keys)
    if flip > MAX_FLIP:
        return f"getting one axis wrong earns {flip:.4f} on B somewhere in the support"
    if leverage < MIN_LEVERAGE:
        return f"B pays {leverage:.4f} for an axis somewhere in the support"
    for convention in ALL_CONVENTIONS:
        at = tuple(sorted(convention.items()))
        earned = shipped_copy_maximum(a_keys[at], b_keys[at])
        if earned > MAX_SHIPPED_COPY:
            return (
                f"reusing A's answers earns {earned:.4f} on B under one convention, "
                f"over the registered {MAX_SHIPPED_COPY:.4f}"
            )
        wider = bijection_copy_maximum(a_keys[at], b_keys[at])
        if wider > MAX_BIJECTION_COPY:
            return (
                f"relabelling A's answers earns {wider:.4f} on B under one "
                f"convention, over the {MAX_BIJECTION_COPY:.4f} the wider family is "
                "qualified at"
            )
    return ""


# --------------------------------------------------------------------------
# the named checks this family adds
# --------------------------------------------------------------------------


def _retasked(instance: Instance, generator, convention: Mapping[str, str]) -> Instance:
    """The same pair scored under another convention, with nothing else moved."""
    sides = {}
    for label, task in (("a", instance.a), ("b", instance.b)):
        sides[label] = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text=task.text,
            key=tuple(generator.key_for(task.table, convention)),
            mask=task.mask,
        )
    return Instance(
        generator=instance.generator, genre=instance.genre, ordinal=instance.ordinal,
        convention=dict(convention), a=sides["a"], b=sides["b"],
        envelope=instance.envelope,
    )


def _read_back(generator, instance: Instance, side: str) -> str:
    """What the serialized graded cell says the case ids and corrections are.

    The serializer fits a field by truncating it rather than refusing it, so a value
    longer than its registered slot becomes a shorter legal looking one and every
    other check still passes. This reads the columns back out of the bytes the agent
    would receive and compares them with the values they were built from.
    """
    task = instance.side(side)
    envelope = frozen_envelope(instance.envelope)
    raw = "\n".join(
        "%s," % identifier for identifier in generator.row_identifiers(task.table)
    )
    canonical = generator.parse_and_canonicalize(task, raw)
    ast = generator.render_receipt(
        task, canonical, task.key, feedback_for(generator, task, envelope)
    )
    payload = serialize(ast, envelope)
    identifier_start = len(GAP) + ORDINAL_WIDTH + len(GAP)
    identifier_end = identifier_start + envelope.identifier_width
    low, high = envelope.slot_span("correction")
    for line, row in zip(row_lines(payload, ast, envelope), ast.rows):
        printed_id = line[identifier_start:identifier_end].decode("ascii").strip()
        printed_correction = line[low:high].decode("ascii").strip()
        wanted = {slot.name: slot.value for slot in row.slots}["correction"]
        if printed_id != row.identifier:
            return (
                f"side {side.upper()} prints the case id {printed_id!r} where the row "
                f"names {row.identifier!r}"
            )
        if printed_correction != wanted:
            return (
                f"side {side.upper()} prints the correction {printed_correction!r} "
                f"where the row's own answer is {wanted!r}"
            )
    return ""


def _reporting_every_case(generator, instance: Instance):
    """The same family and the same pair, with the receipt reporting every case.

    SOME OF WHAT THESE CHECKS ASK IS ABOUT THE ROW RECIPE AND NOT ABOUT A DRAW. That
    the 27 conventions serialize to 27 different receipts, that the receipt resolves
    three blocks on every axis, that no case is evident and that the ceiling and the
    lookup floor are 1 and 7/12 or 11/18 are facts about six cases of each of four
    public kinds. Under a policy that reports four cases of twenty four every one of
    them is also a fact about which four a committed stream happened to draw, and
    several are deliberately false on masks the policy can draw: a mask that drew four
    cases of one kind resolves fewer blocks and aliases conventions, and it is not
    redrawn. The recipe is what those checks are for, so they are asked of the receipt
    that reports every case, and what the served receipt leaves is asked of the
    registered mask law beside them.

    It returns the family and the pair unchanged where the family already reports every
    row, so nothing about a full-policy family passes through a second object.
    """
    if not policy_of(generator).samples:
        return generator, instance

    class EveryCase(type(generator)):
        RECEIPT_POLICY = FULL_RECEIPT

    whole = tuple(range(len(instance.a.table.rows)))
    return EveryCase(), replace(
        instance,
        a=replace(instance.a, mask=whole),
        b=replace(instance.b, mask=whole),
    )


def support_distinction(cases: Sequence[Sequence[Mapping[str, Any]]], q: int) -> float:
    """The smallest chance a receipt of q cases speaks to a single-axis alternative.

    Over EVERY convention and every one-option change to it, not only the one that was
    drawn: two conventions that disagree on d of the printed cases are told apart
    exactly when the mask drew one of those cases, which is `1 - C(24-d, q)/C(24, q)`.
    The keys come from this module's own selector reading the printed body, so the
    number is not the production law's arithmetic restated.
    """
    keys = {
        tuple(sorted(convention.items())): audit_key(cases, convention)
        for convention in ALL_CONVENTIONS
    }
    rows = ROWS
    worst = 1.0
    for convention in ALL_CONVENTIONS:
        here = keys[tuple(sorted(convention.items()))]
        for axis in AXES:
            for option in axis.options:
                if option == convention[axis.name]:
                    continue
                other = dict(convention)
                other[axis.name] = option
                there = keys[tuple(sorted(other.items()))]
                moved = sum(1 for x, y in zip(here, there) if x != y)
                worst = min(
                    worst, 1.0 - comb(rows - moved, q) / comb(rows, q)
                )
    return worst


def check_retail_surface(generator, instance: Instance, master: bytes) -> CheckResult:
    """The printed schedule, read back and answered by a second implementation.

    It fails if a body does not parse, if a printed field leaves its registered
    bounds, if the class partition, the contribution totals, the purchase flag, the
    joint profiles or the side's class cardinality are not what the schedule
    promises, if any parsed record differs from the stored record in any field or
    position, if any of the 27 keys disagrees with the production scorer on either
    side, if two destinations inside one case normalize alike, if two conventions
    share a complete key, or if a case id or a correct code changes under serialize
    and read back.

    THE READ BACK IS ASKED TWICE UNDER A POLICY THAT SAMPLES. The receipt that is
    served prints a correction on the four cases the mask drew and this instance's
    committed neutral tokens on the other twenty, so reading it back confirms what a
    reader is handed and leaves twenty corrections unprinted. The geometry this check
    exists for is about all twenty four, so the same read back is asked of a receipt
    that reports every case as well. Nothing is loosened: both still refuse.
    """
    lines: list[str] = []
    for side, label in (("a", "A"), ("b", "B")):
        table = instance.side(side).table
        refusal = side_refusal(table, label)
        if refusal:
            return CheckResult("retail_surface", False, refusal)
        wrong = record_refusal(table, label)
        if wrong:
            return CheckResult("retail_surface", False, wrong)
        cases = parsed_cases(table)
        instruments = [case.instruments for case in cases]
        for convention in ALL_CONVENTIONS:
            produced = tuple(generator.key_for(table, convention))
            expected = audit_key(instruments, convention)
            if produced != expected:
                wrong = next(
                    n for n, (x, y) in enumerate(zip(produced, expected)) if x != y
                )
                return CheckResult(
                    "retail_surface", False,
                    f"side {label} case {cases[wrong].case_id} is scored "
                    f"{produced[wrong]!r} and an independent selector reading the "
                    f"printed body makes it {expected[wrong]!r}",
                )
        if tuple(c.case_id for c in cases) != tuple(generator.row_identifiers(table)):
            return CheckResult(
                "retail_surface", False,
                f"side {label} prints case ids the row identifiers do not name",
            )
        lines.append(
            "%s %d cases, %d instrument entries, %d distinct keys"
            % (
                label, len(cases), sum(len(c.instruments) for c in cases),
                len({audit_key(instruments, c) for c in ALL_CONVENTIONS}),
            )
        )
    every_case, whole_pair = _reporting_every_case(generator, instance)
    for family, pair in ((generator, instance), (every_case, whole_pair)):
        for side in ("a", "b"):
            wrong = _read_back(family, pair, side)
            if wrong:
                return CheckResult("retail_surface", False, wrong)
    a_cases = parsed_cases(instance.a.table)
    b_cases = parsed_cases(instance.b.table)
    wide = max(
        bijection_copy_maximum(
            audit_key([c.instruments for c in a_cases], convention),
            audit_key([c.instruments for c in b_cases], convention),
        )
        for convention in ALL_CONVENTIONS
    )
    return CheckResult(
        "retail_surface", True,
        "an independent selector reading the printed body agrees with the scorer on "
        "all %d draws of both sides; %s; the separately registered all-bijection "
        "maximum is %.6f against its %.4f qualification"
        % (len(ALL_CONVENTIONS), "; ".join(lines), wide, MAX_BIJECTION_COPY),
    )


def check_retail_support(generator, instance: Instance, master: bytes) -> CheckResult:
    """The registered gates and checks, run under every convention rather than one.

    The drawn-convention checks say what this pair is worth under the rule it
    happened to draw. This says the same thing under all 27, so the pass predicate
    does not depend on which convention was sampled and admission cannot correlate
    the public surface with the live option.

    THE RECEIPT-SHAPED HALF IS ASKED OF THE RECEIPT THAT REPORTS EVERY CASE, and what
    the served receipt leaves is asked of the registered mask law beside it. Which
    four cases a stream drew is not what a recipe check is about, and the gate's
    blocks, its evident cases and its ceiling and floor are deliberately false on
    masks this policy can draw: a mask that drew four cases of one public kind
    resolves fewer blocks and aliases conventions, and it is not redrawn. So they are
    computed on the full receipt, which is a fact about six cases of each of four
    kinds, and the law beside them is the same recipe's statement about what is
    served.
    """
    from shogym.envs.receipts.admission import Thresholds
    from shogym.receipts import gate

    bars = Thresholds()
    worst_headroom = 1.0
    fewest_blocks = len(ALL_CONVENTIONS)
    every_case, whole_pair = _reporting_every_case(generator, instance)
    samples = policy_of(generator).samples
    for convention in ALL_CONVENTIONS:
        retasked = _retasked(instance, generator, convention)
        recipe = _retasked(whole_pair, every_case, convention)
        result = gate(
            observe(every_case, recipe, "a"),
            min_arity=bars.min_arity,
            min_blocks=bars.min_blocks,
            min_headroom=bars.min_headroom,
        )
        if not result.verdict:
            return CheckResult(
                "retail_support", False,
                "under one convention the registered gate set refuses the pair: "
                + (result.reasons[0] if result.reasons else "no reason given"),
            )
        worst_headroom = min(worst_headroom, result.ceiling - result.floor)
        for named in (
            check_exercise(every_case, recipe),
            check_materiality(generator, retasked, bars.min_material_rows),
            # THE COPY BAR IS A FULL-KEY TRANSFER STRESS TEST and says so in its own
            # words, so it is asked here of the receipt that reports every case. The
            # two numbers it bars are computed from the answer keys and are the same
            # either way; what asking it of the served receipt would add is the reduced
            # transfer diagnostic, recomputed at all 27 conventions, which is a
            # reported number and never a bar. The shared copy check reports it once,
            # at the convention that was drawn.
            check_copy(
                every_case, recipe, bars.max_copy_score, bars.max_flip_score,
                bars.min_leverage,
            ),
        ):
            if not named.passed:
                return CheckResult(
                    "retail_support", False,
                    f"under one convention {named.name} fails: {named.detail}",
                )
        fewest_blocks = min(
            fewest_blocks, min(result.blocks[name] for name in AXIS_NAMES)
        )
    detail: list[str] = []
    for label, table in (("A", instance.a.table), ("B", instance.b.table)):
        cases = parsed_cases(table)
        keys = support_keys([c.instruments for c in cases])
        moved = movement(keys)
        bars_for = MIN_MOVED_A if label == "A" else MIN_MOVED_B
        for axis, least in sorted(moved.items()):
            if least < bars_for[axis]:
                return CheckResult(
                    "retail_support", False,
                    f"side {label} moves {least} rows on {axis} somewhere in the "
                    f"support, under the registered {bars_for[axis]}",
                )
        for key in keys.values():
            if set(key) != set(CODES):
                return CheckResult(
                    "retail_support", False,
                    f"side {label} has a key that does not realize all six codes",
                )
        detail.append(
            "%s movement %s" % (label, ", ".join(f"{a} {n}" for a, n in sorted(moved.items())))
        )
    law_line = ""
    if samples:
        # AND WHAT THE SERVED RECEIPT LEAVES, on the same recipe. The row recipe fixes
        # these in the reported count alone: the ideal level and the lookup floor are
        # averaged over every reference convention, so they are one pair of numbers for
        # the pair of schedules rather than one per draw, and the weakest distinction
        # is minimized over the whole support by this module's own selector rather than
        # taken at the convention that happened to be drawn.
        policy = policy_of(generator)
        law = law_for(generator, instance, "a")
        weakest = support_distinction(
            [case.instruments for case in parsed_cases(instance.a.table)],
            policy.reported,
        )
        for name, got, low, high in (
            ("ideal graded level", law.ideal, REGISTERED_MIN_IDEAL, REGISTERED_MAX_IDEAL),
            ("room over the lookup floor", law.room, REGISTERED_MIN_LAW_ROOM, 1.0),
            (
                "weakest single-axis distinction over the support",
                weakest, REGISTERED_MIN_DISTINGUISHING, 1.0,
            ),
        ):
            if not low - TOLERANCE <= got <= high + TOLERANCE:
                return CheckResult(
                    "retail_support", False,
                    "the registered mask law's %s is %.6f and the law-level bars are "
                    "%.4f to %.4f" % (name, got, low, high),
                )
        law_line = (
            "; over the %d masks %d of %d cases can draw, ideal %.6f, lookup floor "
            "%.6f, room %.6f, weakest distinction over the support %.6f"
            % (
                law.masks, policy.reported, policy.rows, law.ideal, law.floor,
                law.room, weakest,
            )
        )
    return CheckResult(
        "retail_support", True,
        "every one of the %d conventions passes the registered gates and checks on a "
        "receipt reporting every case; worst headroom %.6f, fewest blocks %d; %s%s"
        % (
            len(ALL_CONVENTIONS), worst_headroom, fewest_blocks, "; ".join(detail),
            law_line,
        ),
    )


def check_retail_profile_transfer(
    generator, instance: Instance, master: bytes
) -> CheckResult:
    """Whether a corrected destination's structural position is reusable on the sibling.

    It fails if any case of A and any case of B share a code-free fingerprint. The
    two-against-three class cardinality makes that impossible by construction here,
    which is what this check is for: a later revision that made the two sides the same
    shape would reopen it silently.

    WHAT IT DOES NOT ESTABLISH. Partial-feature reuse is not priced by it, and a
    reader that matched only the class sequence or only the purchase flag is outside
    it. It does not show that every alternative algorithm has to induce the rule.
    """
    a_cases = parsed_cases(instance.a.table)
    b_cases = parsed_cases(instance.b.table)
    a_marks = {fingerprint(c) for c in a_cases}
    b_marks = {fingerprint(c) for c in b_cases}
    shared = a_marks & b_marks
    if shared:
        return CheckResult(
            "retail_profile_transfer", False,
            "%d structural fingerprints appear on both schedules, so a corrected "
            "destination's position is reusable" % len(shared),
        )
    return CheckResult(
        "retail_profile_transfer", True,
        "%d fingerprints on A and %d on B share none; class sizes %d and %d"
        % (len(a_marks), len(b_marks), A_CLASS_SIZE, B_CLASS_SIZE),
    )


def bijection_qualification(generator, instance: Instance) -> tuple[float, str]:
    """The all-bijection maximum over the whole support, and whether it qualifies.

    SEPARATELY NAMED, and separately registered. The shipped screen prices the six
    rotations of the published order; this prices all 720 relabellings of the six
    codes, over the same 48 row moves and over every convention the sampler can draw.
    The result is reported as its own number against its own bound and is never
    substituted for the registered closure.
    """
    a_table, b_table = instance.a.table, instance.b.table
    worst = 0.0
    for convention in ALL_CONVENTIONS:
        worst = max(
            worst,
            bijection_copy_maximum(
                generator.key_for(a_table, convention),
                generator.key_for(b_table, convention),
            ),
        )
    if worst > MAX_BIJECTION_COPY:
        return worst, (
            "the all-bijection family earns %.6f on B somewhere in the support, over "
            "the %.4f this qualification asks for; the registered closure is priced "
            "separately and is not this number" % (worst, MAX_BIJECTION_COPY)
        )
    return worst, ""


def check_retail_bijection(generator, instance: Instance, master: bytes) -> CheckResult:
    """The separately named all-bijection qualification, at its own bound."""
    worst, refusal = bijection_qualification(generator, instance)
    if refusal:
        return CheckResult("retail_bijection", False, refusal)
    return CheckResult(
        "retail_bijection", True,
        "the all-bijection family earns at most %.6f on B over the whole support, "
        "against the %.4f qualification" % (worst, MAX_BIJECTION_COPY),
    )


#: The extra checks this module implements, by the name the generator declares them
#: under. `check_additional` dispatches here, and a name this table does not carry is
#: refused rather than defaulted.
NAMED_CHECKS = {
    "retail_surface": check_retail_surface,
    "retail_support": check_retail_support,
    "retail_profile_transfer": check_retail_profile_transfer,
    "retail_bijection": check_retail_bijection,
}


def run_named(name: str, generator, instance: Instance, master: bytes) -> CheckResult:
    """Run one declared extra check by name, refusing a name this module does not have."""
    run = NAMED_CHECKS.get(name)
    if run is None:
        raise ValueError(
            f"{name!r} is not one of this family's checks; it implements "
            f"{', '.join(sorted(NAMED_CHECKS))}"
        )
    return run(generator, instance, master)


# --------------------------------------------------------------------------
# the detail report, for the review pack and the roster release
# --------------------------------------------------------------------------


def pair_report(generator, instance: Instance) -> dict[str, object]:
    """Every number this module computes about one pair, for hashed review material."""
    out: dict[str, object] = {}
    for label, table in (("a", instance.a.table), ("b", instance.b.table)):
        cases = parsed_cases(table)
        keys = support_keys([c.instruments for c in cases])
        flip, leverage = support_flip_and_leverage(keys)
        out[label] = {
            "domain": table.domain,
            "cases": len(cases),
            "instrument_entries": sum(len(c.instruments) for c in cases),
            "distinct_keys": len(set(keys.values())),
            "movement": movement(keys),
            "worst_one_axis_wrong": flip,
            "weakest_leverage": leverage,
            "kinds": sorted({parsed_kind(c) for c in cases}),
        }
    shipped: list[float] = []
    wide: list[float] = []
    for convention in ALL_CONVENTIONS:
        a_key = generator.key_for(instance.a.table, convention)
        b_key = generator.key_for(instance.b.table, convention)
        shipped.append(shipped_copy_maximum(a_key, b_key))
        wide.append(bijection_copy_maximum(a_key, b_key))
    out["copying"] = {
        "shipped_family_maximum": max(shipped),
        "shipped_family_bound": MAX_SHIPPED_COPY,
        "all_bijection_maximum": max(wide),
        "all_bijection_bound": MAX_BIJECTION_COPY,
        "all_bijection_qualifies": max(wide) <= MAX_BIJECTION_COPY,
    }
    return out


__all__ = [
    "AXIS_NAMES",
    "A_JOINT_PROFILES",
    "MAX_BIJECTION_COPY",
    "MAX_FLIP",
    "MAX_SHIPPED_COPY",
    "MIN_LEVERAGE",
    "MIN_MOVED_A",
    "MIN_MOVED_B",
    "NAMED_CHECKS",
    "ParsedCase",
    "audit_destination",
    "audit_key",
    "bijection_copy_maximum",
    "bijection_qualification",
    "body_refusal",
    "case_refusal",
    "check_retail_bijection",
    "check_retail_profile_transfer",
    "check_retail_support",
    "check_retail_surface",
    "fingerprint",
    "movement",
    "pair_refusal",
    "pair_report",
    "parse_printed_body",
    "parsed_cases",
    "parsed_kind",
    "prefers_gift",
    "record_refusal",
    "run_named",
    "selector_of",
    "shipped_copy_maximum",
    "side_refusal",
    "support_distinction",
    "support_flip_and_leverage",
    "support_keys",
]
