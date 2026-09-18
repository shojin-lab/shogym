"""The retail refund genre: its rule, its schedules, its cells and its copying scope.

Every test here is a substantive contract with a failure condition, not a second copy
of a lookup table compared against the first. Where an answer is asserted it comes from
the specification's own worked table or from an independent enumeration, never from the
production selector this module is checking.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from shogym.envs.receipts import checks, copy_profiles, protocol
from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts.generators import retail_refund as retail
from shogym.envs.receipts.generators import retail_validation as validation
from shogym.envs.receipts.observe import observe
from shogym.envs.receipts.oracle import OracleTemplate, render_body
from shogym.envs.receipts.protocol import (
    Instance,
    NoFiling,
    SealedSubmission,
    Task,
    option_mentions,
)
from shogym.envs.receipts.receipt_ast import (
    GRADED,
    ORACLE,
    PLACEBO,
    frozen_envelope,
    mask_slots,
    serialize,
    slot_ranges,
)
from shogym.envs.receipts.render import judge_cells
from shogym.receipts import ROW_LABEL, gate

GENERATOR = retail.GENERATOR
MASTER = bytes(range(32))


@pytest.fixture(scope="module")
def drawn() -> Instance:
    """One drawn instance of this genre, shared by the tests that only read it."""
    return protocol.draw(GENERATOR, MASTER, 0)


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------


def _case(name: str, printed: Sequence[tuple[str, str, int | None, int]]) -> retail.RetailCase:
    """One hand case: (code, type, balance dollars or None, contribution dollars)."""
    instruments = tuple(
        retail.RetailInstrument(
            code=code,
            payment_method_id="%s_%07d" % (kind, 1000000 + n),
            type=kind,
            balance=None if balance is None else balance * retail.CENTS,
            contribution=contribution * retail.CENTS,
        )
        for n, (code, kind, balance, contribution) in enumerate(printed)
    )
    total = sum(i.contribution for i in instruments)
    return retail.RetailCase(
        case_id=name,
        order_id="#W1000000",
        items=(retail.RetailItem("1000000000", "Tee", total),),
        refund_total=total,
        gift_used=any(i.contribution > 0 for i in instruments if i.type == retail.GIFT_CARD),
        instruments=instruments,
    )


G = retail.GIFT_CARD
C = retail.CREDIT_CARD
P = retail.PAYPAL

#: The specification's separating eight-case worked table, transcribed. Amounts are
#: dollars and the sequence in each row is the printed instrument order.
WORKED = (
    ("R1", (("D1", G, 20, 10), ("D2", G, 80, 20))),
    ("R2", (("D4", G, 80, 30), ("D3", G, 20, 10))),
    ("R3", (("D2", C, None, 20), ("D5", P, None, 80))),
    ("R4", (("D6", P, None, 80), ("D1", C, None, 20))),
    ("R5", (("D5", G, 20, 10), ("D1", C, None, 20), ("D3", G, 80, 0), ("D4", P, None, 80))),
    ("R6", (("D2", G, 80, 0), ("D5", P, None, 80), ("D6", G, 20, 10), ("D3", C, None, 20))),
    ("R7", (("D3", G, 20, 0), ("D6", P, None, 80), ("D1", G, 80, 0), ("D4", C, None, 20))),
    ("R8", (("D4", G, 80, 0), ("D2", C, None, 20), ("D5", G, 20, 0), ("D6", P, None, 80))),
)

#: The three panels of the worked table, in the specification's column order: the gift
#: selector varies slowest and the original selector fastest, each running first
#: printed, greatest amount, least amount.
PANELS = {
    "route_gift": (
        "D1 D1 D1 D2 D2 D2 D1 D1 D1", "D4 D4 D4 D4 D4 D4 D3 D3 D3",
        "D2 D5 D2 D2 D5 D2 D2 D5 D2", "D6 D6 D1 D6 D6 D1 D6 D6 D1",
        "D5 D5 D5 D3 D3 D3 D5 D5 D5", "D2 D2 D2 D2 D2 D2 D6 D6 D6",
        "D3 D3 D3 D1 D1 D1 D3 D3 D3", "D4 D4 D4 D4 D4 D4 D5 D5 D5",
    ),
    "route_origin": (
        "D1 D1 D1 D2 D2 D2 D1 D1 D1", "D4 D4 D4 D4 D4 D4 D3 D3 D3",
        "D2 D5 D2 D2 D5 D2 D2 D5 D2", "D6 D6 D1 D6 D6 D1 D6 D6 D1",
        "D1 D4 D1 D1 D4 D1 D1 D4 D1", "D5 D5 D3 D5 D5 D3 D5 D5 D3",
        "D6 D6 D4 D6 D6 D4 D6 D6 D4", "D2 D6 D2 D2 D6 D2 D2 D6 D2",
    ),
    "route_usedgift": (
        "D1 D1 D1 D2 D2 D2 D1 D1 D1", "D4 D4 D4 D4 D4 D4 D3 D3 D3",
        "D2 D5 D2 D2 D5 D2 D2 D5 D2", "D6 D6 D1 D6 D6 D1 D6 D6 D1",
        "D5 D5 D5 D3 D3 D3 D5 D5 D5", "D2 D2 D2 D2 D2 D2 D6 D6 D6",
        "D6 D6 D4 D6 D6 D4 D6 D6 D4", "D2 D6 D2 D2 D6 D2 D2 D6 D2",
    ),
}
SELECTOR_ORDER = tuple(
    (gift, origin) for gift in retail.AXES[1].options for origin in retail.AXES[2].options
)


def test_the_worked_table_is_what_the_scorer_and_an_independent_selector_produce() -> None:
    """All 216 worked answers, from both implementations, and the draws they separate.

    It fails if the production selector compares a gift's contribution instead of its
    balance, lets a purchasing gift into the original class, breaks a tie by the last
    printed member, applies the preferred class's selector after falling back to the
    other class, or reads the purchase flag the wrong way round. It also fails if the
    eight worked cases stop separating all 27 draws, which is what makes them a
    separating table rather than eight cases.
    """
    cases = [_case(name, printed) for name, printed in WORKED]
    parsed = [
        tuple(
            {
                "code": i.code, "type": i.type, "balance": i.balance,
                "contribution": i.contribution,
            }
            for i in case.instruments
        )
        for case in cases
    ]
    checked = 0
    for route, rows in PANELS.items():
        for case, seen, printed in zip(cases, parsed, rows):
            want = printed.split()
            for column, (gift, origin) in enumerate(SELECTOR_ORDER):
                convention = {
                    "route_policy": route, "gift_pick": gift, "origin_pick": origin
                }
                assert retail.destination(case, convention) == want[column], (
                    route, case.case_id, gift, origin
                )
                assert validation.audit_destination(seen, convention) == want[column]
                checked += 1
    assert checked == 216

    vectors = {
        tuple(retail.destination(case, convention) for case in cases)
        for convention in retail.ALL_CONVENTIONS
    }
    assert len(vectors) == len(retail.ALL_CONVENTIONS) == 27


def test_a_purchasing_gift_sits_in_one_class_and_a_non_contributor_is_not_a_destination() -> None:
    """The explicit partition, and what the public fallback does with an empty class.

    It fails if a gift card that paid for the purchase is offered to the original
    selector, if a listed non-gift instrument with no contribution becomes an original
    destination, if a zero-balance gift card is treated as ineligible, or if an empty
    preferred class does not fall through to the other class under that class's own
    selector.
    """
    mixed = _case("K1", (("D1", G, 40, 30), ("D2", C, None, 20), ("D3", P, None, 50)))
    assert [i.code for i in retail.gift_class(mixed)] == ["D1"]
    assert [i.code for i in retail.original_class(mixed)] == ["D2", "D3"]
    assert retail.gift_purchase(mixed) is True

    # A purchasing gift is never a member of the original class, so the original
    # selector cannot reach it however large its contribution was.
    for origin in retail.AXES[2].options:
        picked = retail.destination(
            mixed, {"route_policy": "route_origin", "gift_pick": "gift_list",
                    "origin_pick": origin}
        )
        assert picked in ("D2", "D3")

    # A listed non-gift instrument that paid nothing is in neither class, so a case of
    # one gift and one such instrument falls back to the gift under every route.
    idle = _case("K2", (("D5", C, None, 0), ("D4", G, 0, 10)))
    assert [i.code for i in retail.original_class(idle)] == []
    for convention in retail.ALL_CONVENTIONS:
        assert retail.destination(idle, convention) == "D4"
    # And a zero-balance gift card is still a destination, greatest and least alike.
    zero = _case("K3", (("D1", G, 0, 10), ("D2", G, 50, 20)))
    assert retail.destination(
        zero, {"route_policy": "route_gift", "gift_pick": "gift_min",
               "origin_pick": "origin_list"}
    ) == "D1"

    # The printed-body validator refuses a case that lists an idle original.
    printed = validation.parse_printed_body(
        retail.render_body([idle], retail.SURFACES[("A", 0)]), False
    )[0]
    named = validation.ParsedCase(
        case_id="CA0000000A", order_id=printed.order_id, items=printed.items,
        refund_total=printed.refund_total, gift_used=printed.gift_used,
        instruments=printed.instruments,
    )
    assert "non-contributing original instrument" in validation.case_refusal(named, "A")


def test_a_tie_goes_to_the_earliest_printed_member_on_both_selectors() -> None:
    """The public tie rule, on fixtures the admitted distribution never produces.

    It fails if a tied greatest or a tied least resolves to anything but the earliest
    printed member of the relevant class, or if the two implementations disagree about
    which member that is. The admitted schedules carry no ties on purpose, so the rule
    is held here by hand rather than left to a draw that would never exercise it.
    """
    fixtures = {
        "tied_max_gifts": (
            _case("T1", (("D3", G, 90, 10), ("D5", G, 90, 20), ("D2", G, 10, 30))),
            {"gift_max": "D3", "gift_min": "D2", "gift_list": "D3"},
        ),
        "tied_min_gifts": (
            _case("T2", (("D1", G, 70, 10), ("D6", G, 20, 20), ("D4", G, 20, 30))),
            {"gift_max": "D1", "gift_min": "D6", "gift_list": "D1"},
        ),
        "all_equal_gifts": (
            _case("T3", (("D5", G, 50, 10), ("D1", G, 50, 20), ("D3", G, 50, 30))),
            {"gift_max": "D5", "gift_min": "D5", "gift_list": "D5"},
        ),
    }
    for name, (case, wanted) in fixtures.items():
        for option, code in wanted.items():
            convention = {
                "route_policy": "route_gift", "gift_pick": option,
                "origin_pick": "origin_list",
            }
            assert retail.destination(case, convention) == code, (name, option)

    originals = {
        "tied_max_originals": (
            _case("T4", (("D2", C, None, 80), ("D6", P, None, 80), ("D1", C, None, 10))),
            {"origin_max": "D2", "origin_min": "D1", "origin_list": "D2"},
        ),
        "tied_min_originals": (
            _case("T5", (("D4", P, None, 60), ("D3", C, None, 30), ("D5", P, None, 30))),
            {"origin_max": "D4", "origin_min": "D3", "origin_list": "D4"},
        ),
        "all_equal_originals": (
            _case("T6", (("D6", C, None, 40), ("D2", P, None, 40), ("D4", C, None, 40))),
            {"origin_max": "D6", "origin_min": "D6", "origin_list": "D6"},
        ),
    }
    for name, (case, wanted) in originals.items():
        parsed = tuple(
            {"code": i.code, "type": i.type, "balance": i.balance,
             "contribution": i.contribution}
            for i in case.instruments
        )
        for option, code in wanted.items():
            convention = {
                "route_policy": "route_origin", "gift_pick": "gift_list",
                "origin_pick": option,
            }
            assert retail.destination(case, convention) == code, (name, option)
            assert validation.audit_destination(parsed, convention) == code


# --------------------------------------------------------------------------
# the schedules the construction produces
# --------------------------------------------------------------------------


def _parsed(table: retail.RetailTable):
    return [case.instruments for case in validation.parsed_cases(table)]


def _keys(table: retail.RetailTable) -> dict[tuple, tuple[str, ...]]:
    return validation.support_keys(_parsed(table))


def test_a_generated_pair_prints_the_exact_profile_multiset_and_moves_on_every_axis(
    drawn: Instance,
) -> None:
    """What the construction has to have produced, checked on the printed body.

    It fails if a side does not print six cases of each public kind, if A does not
    print three cases of each amount orientation in each class, if B does not print its
    median amount first, if the two sides do not carry two-candidate and
    three-candidate classes, if any of the 27 keys fails to realize all six codes, if
    two conventions share a complete key, or if a one-option change moves fewer rows
    than the registered movement minima anywhere in the support.
    """
    assert validation.pair_refusal(drawn.a.table, drawn.b.table) == ""
    for label, table, size, bars in (
        ("A", drawn.a.table, retail.A_CLASS_SIZE, validation.MIN_MOVED_A),
        ("B", drawn.b.table, retail.B_CLASS_SIZE, validation.MIN_MOVED_B),
    ):
        cases = validation.parsed_cases(table)
        assert len(cases) == retail.ROWS
        counted: dict[str, int] = {}
        for case in cases:
            counted[validation.parsed_kind(case)] = (
                counted.get(validation.parsed_kind(case), 0) + 1
            )
            for member in (
                [i for i in case.instruments if i["type"] == retail.GIFT_CARD],
                [i for i in case.instruments
                 if i["type"] in validation.NON_GIFT and i["contribution"] > 0],
            ):
                assert not member or len(member) == size
        assert counted == {kind: retail.PER_KIND for kind in retail.KINDS}
        keys = _keys(table)
        assert len(set(keys.values())) == 27
        assert all(set(key) == set(retail.CODES) for key in keys.values())
        assert validation.movement(keys) == bars
    assert validation.movement(_keys(drawn.a.table)) == {
        "route_policy": 6, "gift_pick": 3, "origin_pick": 3
    }
    assert validation.movement(_keys(drawn.b.table)) == {
        "route_policy": 6, "gift_pick": 6, "origin_pick": 6
    }


def test_both_purchase_types_and_both_fallback_strata_keep_the_draws_apart() -> None:
    """What happens to the support when a stratum is dropped.

    It fails if a schedule of mixed gift-purchase cases alone still separates 27
    conventions, or if the original selector still moves a row there: with no
    original-only stratum and a gift purchase on every case, unconditional and
    conditional gift preference agree everywhere and the original selector is inert
    under two of the three routes. The registered 27-key distinctness and movement
    minima are what refuse such a schedule.
    """
    printed = [
        (("D1", G, 10 * (n % 4) + 10, 10), ("D2", G, 90, 0),
         ("D3", C, None, 20), ("D4", P, None, 60))
        for n in range(retail.ROWS)
    ]
    thin = [
        tuple(
            {"code": code, "type": kind,
             "balance": None if balance is None else balance * retail.CENTS,
             "contribution": value * retail.CENTS}
            for code, kind, balance, value in row
        )
        for row in printed
    ]
    keys = validation.support_keys(thin)
    assert len(set(keys.values())) < 27
    moved = validation.movement(keys)
    assert moved["origin_pick"] == 0
    assert moved["route_policy"] == 0


def test_the_two_a_orientations_are_what_separate_first_listed_from_an_extremum() -> None:
    """Why A prints three cases of each amount orientation in each class.

    It fails if a schedule whose gift class always prints its smaller balance first
    still tells the first-listed selector apart from the least-balance selector: on
    such a schedule the two produce the same complete key, so A's receipt merges them
    and the exercise check cannot reach three blocks on that axis.
    """
    one_way = [
        tuple(
            {"code": code, "type": G, "balance": balance * retail.CENTS,
             "contribution": 10 * retail.CENTS}
            for code, balance in (("D1", 10 + n), ("D2", 60 + n))
        )
        for n in range(retail.ROWS)
    ]
    keys = validation.support_keys(one_way)
    listed = keys[
        tuple(sorted({"route_policy": "route_gift", "gift_pick": "gift_list",
                      "origin_pick": "origin_list"}.items()))
    ]
    least = keys[
        tuple(sorted({"route_policy": "route_gift", "gift_pick": "gift_min",
                      "origin_pick": "origin_list"}.items()))
    ]
    assert listed == least
    assert validation.movement(keys)["gift_pick"] == 0


def test_the_room_arithmetic_is_what_the_specification_registered(drawn: Instance) -> None:
    """The prior optima, the floors, the ceiling and the one-axis-wrong bound.

    It fails if A's uninformed optimum is not 5/9 or B's is not 5/18, if the full
    receipt's ceiling is not 1, if the lookup floor is not 7/12 under either
    unconditional route or 11/18 under the conditional one, if the headroom falls under
    the registered 0.05, if the best one-axis-wrong score on B rises above 0.75, or if
    the weakest axis leverage on B falls under 0.25. Each is computed here by
    enumeration or through the shared gate, never read from a stored number.
    """
    def prior(table: retail.RetailTable) -> Fraction:
        keys = _keys(table)
        total = Fraction(0)
        for row in range(retail.ROWS):
            counts: dict[str, int] = {}
            for key in keys.values():
                counts[key[row]] = counts.get(key[row], 0) + 1
            total += Fraction(max(counts.values()), 27)
        return total / retail.ROWS

    assert prior(drawn.a.table) == Fraction(5, 9)
    assert prior(drawn.b.table) == Fraction(5, 18)

    flip, leverage = validation.support_flip_and_leverage(_keys(drawn.b.table))
    assert flip == pytest.approx(0.75)
    assert leverage == pytest.approx(0.25)

    floors: dict[str, set[tuple[float, float]]] = {}
    for convention in retail.ALL_CONVENTIONS:
        result = gate(
            observe(GENERATOR, validation._retasked(drawn, GENERATOR, convention), "a"),
            min_arity=3, min_blocks=2, min_headroom=0.05,
        )
        assert result.verdict
        floors.setdefault(convention["route_policy"], set()).add(
            (round(result.ceiling, 6), round(result.floor, 6))
        )
        assert result.placebo == pytest.approx(5 / 18)
    assert floors["route_gift"] == {(1.0, round(7 / 12, 6))}
    assert floors["route_origin"] == {(1.0, round(7 / 12, 6))}
    assert floors["route_usedgift"] == {(1.0, round(11 / 18, 6))}


def test_the_construction_filter_never_reads_the_drawn_convention() -> None:
    """Acceptance is a function of the two schedules, so rejection teaches nothing.

    It fails if `pair_refusal` takes a convention, if the pair an ordinal produces
    moves when the convention stream moves, or if two draws of one ordinal under one
    key return different schedules. A filter that read the live draw would let an agent
    exclude conventions from the fact that this pair was served.
    """
    import inspect

    signature = inspect.signature(validation.pair_refusal)
    assert list(signature.parameters) == ["a_table", "b_table"]
    source = inspect.getsource(validation.pair_refusal)
    assert "convention[" not in source

    first = protocol.draw(GENERATOR, MASTER, 3)
    retail.build_pair.cache_clear()
    again = protocol.draw(GENERATOR, MASTER, 3)
    assert GENERATOR.table_record(first.a.table) == GENERATOR.table_record(again.a.table)
    assert GENERATOR.table_record(first.b.table) == GENERATOR.table_record(again.b.table)


def test_an_integer_cent_schedule_is_the_one_a_second_reader_scores(drawn: Instance) -> None:
    """The printed body is the stored table, and answering it needs neither.

    It fails if a printed amount, flag, total or code binding disagrees with the stored
    table, if the items do not sum to the refund total in integer cents, if the
    contributions do not either, or if any of the 27 keys from a second implementation
    reading only the printed body disagrees with the production scorer. It also fails
    if a mutated body is still scored the same, which is what says the independent
    reader is reading the body rather than the store.
    """
    for label, side in (("A", "a"), ("B", "b")):
        table = drawn.side(side).table
        cases = validation.parsed_cases(table)
        record = GENERATOR.table_record(table)
        assert [c.case_id for c in cases] == [r["case_id"] for r in record["rows"]]
        for case, row in zip(cases, record["rows"]):
            assert case.refund_total == row["refund_total"]
            assert sum(i["price"] for i in case.items) == case.refund_total
            assert sum(i["contribution"] for i in case.instruments) == case.refund_total
            assert (case.gift_used == "yes") == row["gift_used"]
            assert [i["code"] for i in case.instruments] == [
                i["code"] for i in row["instruments"]
            ]
            assert [i["payment_method_id"] for i in case.instruments] == [
                i["payment_method_id"] for i in row["instruments"]
            ]
        instruments = [c.instruments for c in cases]
        for convention in retail.ALL_CONVENTIONS:
            assert tuple(GENERATOR.key_for(table, convention)) == validation.audit_key(
                instruments, convention
            ), label

    # A body whose first case prints another balance is answered differently, so the
    # second reader is reading the printed schedule and not the stored one.
    table = drawn.a.table
    body = table.body
    case = validation.parsed_cases(table)[0]
    gifts = [i for i in case.instruments if i["type"] == retail.GIFT_CARD]
    if gifts:
        printed = "%d.%02d" % divmod(gifts[0]["balance"], retail.CENTS)
        moved = body.replace(printed, "%d.%02d" % divmod(
            (gifts[0]["balance"] + 500) % 10000, retail.CENTS), 1)
        assert moved != body
        changed = validation.parse_printed_body(moved, table.template.json_body)
        assert changed[0].instruments != case.instruments


# --------------------------------------------------------------------------
# reading a filing
# --------------------------------------------------------------------------


def _read(task: Task, raw: object):
    return GENERATOR.parse_and_canonicalize(task, raw)


def test_the_reading_is_the_registered_one_on_this_genre_too(drawn: Instance) -> None:
    """Every clause of the published reading rules, on this family's own identifiers.

    It fails if a later line for a case replaces the first, if identifier matching is
    not case-insensitive after whitespace is collapsed, if an extra comma splits a
    value, if an unknown case id creates a scored row, if an invalid code is dropped
    rather than filed and wrong, if a list input is not accepted, if a character
    outside printable ASCII is transliterated rather than folded to a question mark, if
    a comma-free filing of exactly the printed row count is not read positionally, or
    if any other comma-free filing is.
    """
    task = drawn.a
    identifiers = list(GENERATOR.row_identifiers(task.table))
    key = list(task.key)

    first_wins = _read(task, "\n".join(
        [f"{identifiers[0]},{key[0]}", f"{identifiers[0]},D6", f"{identifiers[1]},{key[1]}"]
    ))
    assert isinstance(first_wins, SealedSubmission)
    assert first_wins.values[0] == key[0]
    assert first_wins.duplicates == (identifiers[0],)

    folded = _read(task, "  %s  ,  %s  " % (identifiers[2].lower(), key[2].lower()))
    assert isinstance(folded, SealedSubmission)
    assert GENERATOR.normalize_answer(folded.values[2]) == GENERATOR.normalize_answer(key[2])

    extra_commas = _read(task, f"{identifiers[3]},{key[3]},D1,D2")
    assert isinstance(extra_commas, SealedSubmission)
    assert extra_commas.values[3] == f"{key[3]},D1,D2"
    assert GENERATOR.score(task, extra_commas)[0] == 0.0

    unknown = _read(task, "CA00000000,D1\n%s,%s" % (identifiers[4], key[4]))
    assert isinstance(unknown, SealedSubmission)
    assert unknown.extras == ("CA00000000",)
    assert unknown.filed_rows == 1

    invalid = _read(task, f"{identifiers[5]},D 2")
    assert isinstance(invalid, SealedSubmission)
    assert invalid.values[5] == "D 2"
    assert invalid.filed[5] is True

    listed = _read(task, [f"{i},{v}" for i, v in zip(identifiers, key)])
    assert isinstance(listed, SealedSubmission)
    assert GENERATOR.score(task, listed)[0] == 1.0

    unprintable = _read(task, f"{identifiers[6]},{checks.UNPRINTABLE_VALUE}")
    assert isinstance(unprintable, SealedSubmission)
    assert unprintable.values[6].isascii()
    assert "?" in unprintable.values[6]

    positional = _read(task, "\n".join(key))
    assert isinstance(positional, SealedSubmission)
    assert GENERATOR.score(task, positional)[0] == 1.0
    assert isinstance(_read(task, "\n".join(key[:-1])), NoFiling)
    assert _read(task, "").reason == "empty"
    assert _read(task, {"a": 1}).reason == "unreadable"
    assert _read(task, "prose about the schedule").reason == "no_known_identifier"


def test_an_omitted_or_empty_destination_earns_nothing_and_still_prints_a_row(
    drawn: Instance,
) -> None:
    """The filed mask, and what the receipt has to show for a row nobody answered.

    It fails if an omitted case is credited, if a filed empty value is credited, if the
    two are recorded as the same act, or if a NoFiling produces fewer than the printed
    number of outcomes: a receipt rendered against an absent filing has to have the
    same shape as any other.
    """
    task = drawn.a
    identifiers = list(GENERATOR.row_identifiers(task.table))
    blank = _read(task, "\n".join("%s," % i for i in identifiers))
    score, outcomes = GENERATOR.score(task, blank)
    assert score == 0.0
    assert len(outcomes) == retail.ROWS
    assert all(o.was_filed and o.filed == "" and not o.matched for o in outcomes)

    half = _read(task, "\n".join(
        "%s,%s" % (i, v) for i, v in list(zip(identifiers, task.key))[:12]
    ))
    score, outcomes = GENERATOR.score(task, half)
    assert score == 0.5
    assert [o.was_filed for o in outcomes] == [True] * 12 + [False] * 12

    none = _read(task, "")
    score, outcomes = GENERATOR.score(task, none)
    assert score == 0.0
    assert len(outcomes) == retail.ROWS
    assert not any(o.was_filed for o in outcomes)


# --------------------------------------------------------------------------
# the three cells
# --------------------------------------------------------------------------


def test_every_cell_is_congruent_under_every_filing_class_and_every_draw(
    drawn: Instance,
) -> None:
    """What the fork would commit, over the whole support and every registered filing.

    It fails if a graded row says anything but what the scorer said, if a passing row
    carries a correction, if a failing row's correction is not that same row's own
    code, if the placebo leaves one byte of either slot on any row unreplaced, if the
    graded and placebo cells differ outside the registered slots, or if any of the
    three cells is not the registered envelope size.
    """
    envelope = frozen_envelope(drawn.envelope)
    seen = 0
    for side in ("a", "b"):
        task = drawn.side(side)
        for shape in checks.FILING_CLASSES:
            raw = checks.filing_of(GENERATOR, drawn, side, shape)
            for convention in retail.ALL_CONVENTIONS:
                truth = tuple(GENERATOR.key_for(task.table, convention))
                retasked = Task(
                    label=task.label, task_id=task.task_id, surface=task.surface,
                    table=task.table, text=task.text, key=truth,
                )
                canonical = GENERATOR.parse_and_canonicalize(retasked, raw)
                judged = judge_cells(
                    GENERATOR, retasked, canonical, convention, envelope
                )
                assert judged.problems == (), (side, shape, judged.problems)
                for kind in (GRADED, PLACEBO, ORACLE):
                    assert len(judged.payloads[kind]) == retail.ENVELOPE_SIZE
                ranges = slot_ranges(judged.asts[GRADED], envelope)
                assert mask_slots(judged.payloads[GRADED], ranges) == mask_slots(
                    judged.payloads[PLACEBO], ranges
                )
                for row, outcome in zip(judged.asts[GRADED].rows, judged.outcomes):
                    slots = {slot.name: slot.value for slot in row.slots}
                    assert slots["verdict"] == ("PASS" if outcome.matched else "FAIL")
                    assert slots["correction"] == (
                        "" if outcome.matched else outcome.correct
                    )
                for position, row in enumerate(judged.asts[PLACEBO].rows):
                    for slot in row.slots:
                        assert slot.value == envelope.neutral[slot.name][position]
                seen += 1
    assert seen == 2 * len(checks.FILING_CLASSES) * 27


def test_a_two_byte_code_survives_serialize_and_read_back(drawn: Instance) -> None:
    """The registered geometry, read out of the bytes rather than out of the structure.

    It fails if a case id or a correction changes under serialize and read back, if the
    row line is not 59 bytes, if the verdict and correction do not sit at their
    registered spans, or if the envelope is not 2657 bytes. A serializer fits an
    overwide field by truncating it, so a correction that no longer named a destination
    would still look like a legal answer.
    """
    envelope = frozen_envelope(drawn.envelope)
    assert retail.ENVELOPE_SIZE == 2657
    assert envelope.row_line_width == 59
    assert envelope.slot_span("verdict") == (40, 44)
    assert envelope.slot_span("correction") == (46, 58)
    for side in ("a", "b"):
        assert validation._read_back(GENERATOR, drawn, side) == ""


def test_the_task_text_leaves_every_axis_undetermined_and_names_what_it_governs(
    drawn: Instance,
) -> None:
    """The public description: no option token, the shared scope sentence, no engagement.

    It fails if either task text names an option of any axis, if the scope sentence is
    not the shared one chosen by the sibling label, if the word an earlier run's readers
    mapped onto the organization printed at the top of the schedule reappears, or if the
    text moves under any convention.
    """
    from shogym.envs.receipts.filing import scope_sentence

    for task in (drawn.a, drawn.b):
        assert option_mentions(retail.AXES, task.text) == []
        assert scope_sentence(task.label) in task.text
        assert "engagement" not in task.text.lower()
        assert "SCHEDULE (" in task.text
        assert "D1, D2, D3, D4, D5, D6" in task.text
    assert checks.check_invariance(GENERATOR, drawn).passed
    assert GENERATOR.row_label(drawn.a.table) == (ROW_LABEL,) * retail.ROWS
    assert set(GENERATOR.row_classes(drawn.a.table)) <= set(retail.KINDS)


def test_the_oracle_is_the_registered_cell_and_states_exactly_the_drawn_rule(
    drawn: Instance,
) -> None:
    """One statement per axis, readable to one convention, inside the allowance.

    It fails if any of the 27 rendered bodies does not read back to the convention it
    was rendered from, if a flattened option phrase is contained in another anywhere in
    the table rather than only within its own axis, if a rendered body exceeds the
    registered oracle allowance, or if the heading scopes the rule to anything but the
    schedules the reader files.
    """
    template = GENERATOR.ORACLE
    assert isinstance(template, OracleTemplate)
    flat = [
        (axis, option, re.sub(r"\s+", " ", phrase).strip())
        for axis, options in template.phrases.items()
        for option, phrase in options.items()
    ]
    assert len(flat) == 9
    for _, _, one in flat:
        for axis, option, other in flat:
            if one is not other:
                assert one not in other, (axis, option)

    for convention in retail.ALL_CONVENTIONS:
        body = render_body(template, convention)
        assert len("\n".join(body)) <= retail.ORACLE_BODY_ALLOWANCE
        ast = GENERATOR.render_oracle(drawn.a.task_id, convention, retail.ROWS)
        assert GENERATOR.parse_oracle(ast) == convention
        assert ast.rows == ()
    assert checks.check_oracle(GENERATOR, drawn).passed
    head = "\n".join(retail.ORACLE_TEMPLATE.head)
    assert "THIS SCHEDULE AND THE NEXT ONE YOU FILE" in head
    assert "engagement" not in head.lower()
    assert retail.ORACLE_TEMPLATE.head[-1] == ""


# --------------------------------------------------------------------------
# fixation and copying
# --------------------------------------------------------------------------


def test_the_whole_schedule_is_committed_and_a_changed_binding_moves_the_digest(
    drawn: Instance,
) -> None:
    """What the bank commits to, field by field.

    It fails if the instance does not rebuild to one digest, or if changing a nested
    item price, a code binding, the printed order of one case's instruments or the
    printed body leaves the committed digest where it was: a field that is read and not
    recorded can move between two rebuilds without the digest noticing.
    """
    assert checks.check_fixation(GENERATOR, drawn, MASTER).passed
    before = bank_mod.instance_digest(drawn, GENERATOR)

    def moved(table: retail.RetailTable) -> str:
        changed = Instance(
            generator=drawn.generator, genre=drawn.genre, ordinal=drawn.ordinal,
            convention=drawn.convention,
            a=Task(
                label=drawn.a.label, task_id=drawn.a.task_id, surface=drawn.a.surface,
                table=table, text=drawn.a.text, key=drawn.a.key,
            ),
            b=drawn.b, envelope=drawn.envelope,
        )
        return bank_mod.instance_digest(changed, GENERATOR)

    table = drawn.a.table
    first = table.rows[0]
    price = first.items[0]
    with_price = retail.RetailTable(
        domain=table.domain,
        rows=(
            retail.RetailCase(
                case_id=first.case_id, order_id=first.order_id,
                items=(retail.RetailItem(price.item_id, price.name, price.price + 1),)
                + first.items[1:],
                refund_total=first.refund_total, gift_used=first.gift_used,
                instruments=first.instruments,
            ),
        ) + table.rows[1:],
        body=table.body,
    )
    assert moved(with_price) != before

    rotated = first.instruments[1:] + first.instruments[:1]
    with_order = retail.RetailTable(
        domain=table.domain,
        rows=(
            retail.RetailCase(
                case_id=first.case_id, order_id=first.order_id, items=first.items,
                refund_total=first.refund_total, gift_used=first.gift_used,
                instruments=rotated,
            ),
        ) + table.rows[1:],
        body=table.body,
    )
    assert moved(with_order) != before

    rebound = tuple(
        retail.RetailInstrument(
            code=retail.CODES[(retail.CODES.index(i.code) + 1) % 6],
            payment_method_id=i.payment_method_id, type=i.type, balance=i.balance,
            contribution=i.contribution,
        )
        for i in first.instruments
    )
    with_codes = retail.RetailTable(
        domain=table.domain,
        rows=(
            retail.RetailCase(
                case_id=first.case_id, order_id=first.order_id, items=first.items,
                refund_total=first.refund_total, gift_used=first.gift_used,
                instruments=rebound,
            ),
        ) + table.rows[1:],
        body=table.body,
    )
    assert moved(with_codes) != before
    assert moved(retail.RetailTable(
        domain=table.domain, rows=table.rows, body=table.body + "\n"
    )) != before


def test_the_published_ranks_are_the_whole_vocabulary_and_the_closure_is_six_rotations(
    drawn: Instance,
) -> None:
    """What the copy screen prices this family against, and what it must never price.

    It fails if the published ranks are not the complete six-code vocabulary under
    every convention and on both sides, if the declared profile is not the ordered
    token one, if the closed token monoid is not the six rotations of that order, if
    the registered closure produces more than 288 filings, or if the shipped family
    earns more than the registered 0.50 on B anywhere in the support.
    """
    assert GENERATOR.COPY_PROFILE == copy_profiles.ORDERED_TOKENS
    for table in (drawn.a.table, drawn.b.table):
        assert GENERATOR.answer_ranks(table) == retail.CODES
        assert copy_profiles.answer_ranks_for(GENERATOR, table) == retail.CODES

    maps = copy_profiles.token_maps(list(drawn.a.key), retail.CODES, retail.CODES)
    assert len(maps) == 6
    rotations = [
        {retail.CODES[i]: retail.CODES[(i + shift) % 6] for i in range(6)}
        for shift in range(6)
    ]
    assert sorted(map(str, map(sorted, (m.items() for m in maps)))) == sorted(
        map(str, map(sorted, (r.items() for r in rotations)))
    )
    filings = checks.copy_map_filings(GENERATOR, drawn)
    assert len(filings["relabel"]) == 6
    assert len(filings["permutation"]) == 2 * retail.ROWS
    assert len(filings["closure"]) <= 6 * 2 * retail.ROWS

    for convention in retail.ALL_CONVENTIONS:
        earned = validation.shipped_copy_maximum(
            GENERATOR.key_for(drawn.a.table, convention),
            GENERATOR.key_for(drawn.b.table, convention),
        )
        assert earned <= validation.MAX_SHIPPED_COPY
    assert checks.check_copy(GENERATOR, drawn, 0.50, 0.875, 0.10).passed


def test_the_all_bijection_family_is_larger_and_is_priced_under_its_own_name(
    drawn: Instance,
) -> None:
    """The separately named qualification, and that it is not the registered closure.

    It fails if the all-bijection maximum is ever below the shipped family's, which
    would mean the wider family is not wider, if the qualification's verdict is reported
    under the registered copy check's name, or if the qualification's own result stops
    naming the number it measured. It does NOT assert that the qualification passes:
    what it is worth on this design is a measurement, and the measurement is what the
    check reports.
    """
    for convention in retail.ALL_CONVENTIONS:
        a_key = GENERATOR.key_for(drawn.a.table, convention)
        b_key = GENERATOR.key_for(drawn.b.table, convention)
        assert validation.bijection_copy_maximum(
            a_key, b_key
        ) >= validation.shipped_copy_maximum(a_key, b_key)

    worst, refusal = validation.bijection_qualification(GENERATOR, drawn)
    result = validation.check_retail_bijection(GENERATOR, drawn, MASTER)
    assert result.name == "retail_bijection"
    assert result.passed == (not refusal)
    assert ("%.6f" % worst) in result.detail
    # The registered closure is a different number and is never replaced by this one.
    assert worst >= max(
        checks.copy_scores(GENERATOR, drawn)[name]
        for name in checks.NO_INDUCTION_MAPS
    )


def test_a_corrected_position_is_not_reusable_on_the_sibling(drawn: Instance) -> None:
    """The code-free structural fingerprints of the two schedules share nothing.

    It fails if any case of A and any case of B have the same ordered class sequence,
    within-class amount ranks, purchase flag and class sizes, which is the position a
    reader could reuse a corrected destination from without inferring anything. The
    two-against-three cardinality is what makes it impossible here, so it also fails if
    a side stops printing its registered class size.
    """
    result = validation.check_retail_profile_transfer(GENERATOR, drawn, MASTER)
    assert result.passed, result.detail
    a_marks = {validation.fingerprint(c) for c in validation.parsed_cases(drawn.a.table)}
    b_marks = {validation.fingerprint(c) for c in validation.parsed_cases(drawn.b.table)}
    assert a_marks and b_marks and not (a_marks & b_marks)
    # A schedule of B's shape shares fingerprints with itself, so the check is not
    # passing because fingerprints never collide.
    assert validation.check_retail_profile_transfer(
        GENERATOR,
        Instance(
            generator=drawn.generator, genre=drawn.genre, ordinal=drawn.ordinal,
            convention=drawn.convention, a=drawn.b, b=drawn.b, envelope=drawn.envelope,
        ),
        MASTER,
    ).passed is False


def test_the_printed_schedule_and_its_answers_are_checked_by_a_second_reader(
    drawn: Instance,
) -> None:
    """The `retail_surface` check, and the fixtures that have to fail it.

    It fails if the check passes a schedule whose body does not read back, whose kind
    counts are wrong, whose case prints two destinations that normalize alike, or whose
    stored table and printed body disagree about an amount.
    """
    assert validation.check_retail_surface(GENERATOR, drawn, MASTER).passed

    broken = retail.RetailTable(
        domain=drawn.a.table.domain, rows=drawn.a.table.rows,
        body=drawn.a.table.body.replace("D1", "d1", 1),
    )
    refusal = validation.side_refusal(broken, "A")
    assert refusal

    short = retail.RetailTable(
        domain=drawn.a.table.domain, rows=drawn.a.table.rows,
        body="\n".join(drawn.a.table.body.split("\n")[:-1]),
    )
    assert "prints 23 cases" in validation.side_refusal(short, "A")


def _payload(instance: Instance, side: str, convention: Mapping[str, str]) -> bytes:
    task = instance.side(side)
    retasked = Task(
        label=task.label, task_id=task.task_id, surface=task.surface,
        table=task.table, text=task.text,
        key=tuple(GENERATOR.key_for(task.table, convention)),
    )
    canonical = GENERATOR.parse_and_canonicalize(
        retasked, checks.filing_of(GENERATOR, instance, side, "canonical")
    )
    ast = GENERATOR.render_receipt(retasked, canonical, retasked.key)
    return serialize(ast, frozen_envelope(instance.envelope))


def test_the_receipt_prints_no_neutral_token_that_reads_as_a_grade(drawn: Instance) -> None:
    """The committed neutral tokens, against every verdict token and every legal answer.

    It fails if a committed token equals a verdict token, a display token or a
    destination code under either spelling: scorer equality is case-insensitive, so a
    token that matched a code when folded would carry an apparent grade into the arm
    whose whole purpose is to carry none.
    """
    assert checks.check_neutral(GENERATOR, drawn).passed
    forbidden = {"PASS", "FAIL", retail.BLANK_TOKEN, retail.UNFILED_TOKEN}
    forbidden |= set(retail.CODES)
    folded = {value.lower() for value in forbidden}
    for tokens in drawn.envelope.neutral.values():
        for token in tokens:
            assert token.strip() not in forbidden
            assert token.strip().lower() not in folded
    # And a graded payload under one convention is not a payload under another.
    first = _payload(drawn, "a", retail.ALL_CONVENTIONS[0])
    assert any(
        _payload(drawn, "a", convention) != first
        for convention in retail.ALL_CONVENTIONS[1:]
    )


def test_the_schedule_prints_as_the_surface_declares_and_nothing_else(
    drawn: Instance,
) -> None:
    """The four surfaces, their two body formats and the bounds they promise.

    It fails if a pipe body does not open with the six registered column names, if a
    JSON body does not carry those six keys in that order, if a body ends with a
    newline the template is supposed to supply, if any JSON is not compact and ASCII,
    or if a surface's organization or item pool leaks into the other side's schedule.
    """
    assert GENERATOR.surface_templates() == (
        "apparel", "home", "electronics", "outdoors"
    )
    assert GENERATOR.surface_for(0, "A") == "apparel"
    assert GENERATOR.surface_for(1, "A") == "home"
    assert {GENERATOR.surface_for(n, "B") for n in range(8)} == {
        "electronics", "outdoors"
    }
    a_table, b_table = drawn.a.table, drawn.b.table
    assert not a_table.body.endswith("\n") and not b_table.body.endswith("\n")
    assert a_table.body.split("\n")[0] == " | ".join(retail.HEADER)
    assert len(a_table.body.split("\n")) == retail.ROWS + 1
    for line in b_table.body.split("\n"):
        record = json.loads(line)
        assert list(record) == list(retail.HEADER)
        assert line == json.dumps(record, ensure_ascii=True, separators=(",", ":"))
    assert a_table.body.isascii() and b_table.body.isascii()
    assert a_table.template.organization in drawn.a.text
    assert a_table.template.organization not in drawn.b.text
    for name in b_table.template.items:
        assert name not in a_table.body


# --------------------------------------------------------------------------
# the checks a family declares for itself
# --------------------------------------------------------------------------


def test_a_family_declares_its_own_checks_and_they_run_where_the_eleven_run(
    drawn: Instance,
) -> None:
    """The optional extension, and that it changes nothing for a family without one.

    It fails if the three checks this family declares are not run beside the eleven, if
    one of them failing still leaves the instance admissible, or if a family that
    declares none has its check list changed by the extension existing.
    """
    names = [
        result.name
        for result in checks.run_checks(
            GENERATOR, drawn, MASTER, max_copy_score=0.50, max_flip_score=0.875,
            min_leverage=0.10,
        )
    ]
    assert names == list(checks.STANDARD_CHECKS) + list(GENERATOR.ADDITIONAL_CHECKS)
    assert GENERATOR.ADDITIONAL_CHECKS == (
        "retail_surface", "retail_support", "retail_profile_transfer"
    )

    from shogym.envs.receipts import admission
    from shogym.envs.receipts.generators import ledger
    from shogym.envs.receipts.generators.vectors import VECTORS

    report = admission.report(GENERATOR, drawn, MASTER, admission.Thresholds())
    assert report.admitted
    assert [c.name for c in report.checks] == names

    held = validation.NAMED_CHECKS["retail_profile_transfer"]
    try:
        validation.NAMED_CHECKS["retail_profile_transfer"] = (
            lambda *_: checks.CheckResult(
                "retail_profile_transfer", False, "a fixture refusal"
            )
        )
        refused = admission.report(GENERATOR, drawn, MASTER, admission.Thresholds())
        assert "retail_profile_transfer" in refused.failed_checks
        assert not refused.admitted
    finally:
        validation.NAMED_CHECKS["retail_profile_transfer"] = held

    for other in (ledger.GENERATOR, VECTORS["merge"]):
        assert getattr(other, "ADDITIONAL_CHECKS", ()) == ()
        assert checks.additional_checks(other, drawn, MASTER, []) == []


class _Declaring:
    """A stand-in that declares whatever the test needs it to declare."""

    name = "declaring"
    AXES = retail.AXES

    def __init__(self, declared, run) -> None:
        self.ADDITIONAL_CHECKS = declared
        if run is not None:
            self.check_additional = run


def _ran(generator, instance: Instance) -> dict[str, checks.CheckResult]:
    planned = checks.additional_checks(generator, instance, MASTER, ["copy", "oracle"])
    return {name: checks._guarded(name, run) for name, run in planned}


def test_a_declared_check_cannot_borrow_another_name_or_shadow_one(
    drawn: Instance,
) -> None:
    """What the declaration is not trusted with.

    It fails if a declared name that shadows one of the eleven is run, if a repeated
    name is run twice, if a result returned under another name is recorded under that
    name, if a family that declares names without a runner is treated as declaring
    none, or if an exception inside a declared check escapes instead of becoming that
    check's failure.
    """
    shadow = _ran(
        _Declaring(("copy",), lambda name, *_: checks.CheckResult(name, True, "ok")),
        drawn,
    )
    assert not shadow["copy"].passed
    assert "shadows" in shadow["copy"].detail

    twice = _ran(
        _Declaring(("mine", "mine"), lambda name, *_: checks.CheckResult(name, True, "ok")),
        drawn,
    )
    assert twice["mine"].passed is False
    assert "declared twice" in twice["mine"].detail

    borrowed = _ran(
        _Declaring(("mine",), lambda *_: checks.CheckResult("copy", True, "ok")), drawn
    )
    assert not borrowed["mine"].passed
    assert "another name" in borrowed["mine"].detail

    def raising(*_):
        raise RuntimeError("the family blew up")

    blew = _ran(_Declaring(("mine",), raising), drawn)
    assert not blew["mine"].passed
    assert "RuntimeError" in blew["mine"].detail

    missing = _ran(_Declaring(("mine",), None), drawn)
    assert not missing["additional"].passed
    assert "no check_additional" in missing["additional"].detail


def _b_case(n: int, kind: str) -> retail.RetailCase:
    """One B case of a chosen kind, built by hand rather than by the construction."""
    gifts = [("D%d" % (n % 3 + 1), G, 10 * (n % 5) + 20 + 10 * k, 0) for k in range(3)]
    if kind == retail.GIFTS_ONLY:
        printed = [
            ("D%d" % (k + 1), G, 20 + 10 * k + (n % 3) * 10, 10 * (k + 1))
            for k in range(3)
        ]
    else:
        printed = [
            ("D1", G, 30, 20), ("D2", G, 10, 0), ("D3", G, 90, 0),
            ("D4", C, None, 30), ("D5", P, None, 10), ("D6", C, None, 90),
        ]
    del gifts
    instruments = tuple(
        retail.RetailInstrument(
            code=code, payment_method_id="%s_%07d" % (kind_, 5000000 + n * 10 + k),
            type=kind_, balance=None if balance is None else balance * retail.CENTS,
            contribution=value * retail.CENTS,
        )
        for k, (code, kind_, balance, value) in enumerate(printed)
    )
    total = sum(i.contribution for i in instruments)
    return retail.RetailCase(
        case_id="CB%08X" % (0x10000 + n),
        order_id="#W%07d" % (5000000 + n),
        items=(retail.RetailItem("%010d" % (5000000000 + n), "Cable", total),),
        refund_total=total,
        gift_used=any(
            i.contribution > 0 for i in instruments if i.type == retail.GIFT_CARD
        ),
        instruments=instruments,
    )


def test_support_wide_admission_does_not_depend_on_the_convention_sampled(
    drawn: Instance,
) -> None:
    """A pair that passes under one draw and fails under another is refused for all.

    The crafted sibling prints only gift-only cases and mixed cases with a gift
    purchase, so under either gift-preferring route no row ever reaches the original
    class and that selector moves nothing. Under the original-preferring route every
    axis is material and the drawn-convention checks pass. It fails if the
    drawn-convention materiality check does not show that split, or if the support-wide
    check's verdict moves with the convention the instance happened to draw.
    """
    rows = tuple(
        _b_case(n, retail.GIFTS_ONLY if n < 12 else retail.BOTH_GIFT_PURCHASE)
        for n in range(retail.ROWS)
    )
    surface = retail.SURFACES[("B", 0)]
    crafted = retail.RetailTable(
        domain=surface.name, rows=rows, body=retail.render_body(rows, surface)
    )

    def instance_of(route: str) -> Instance:
        convention = {
            "route_policy": route, "gift_pick": "gift_max", "origin_pick": "origin_min"
        }
        return Instance(
            generator=drawn.generator, genre=drawn.genre, ordinal=drawn.ordinal,
            convention=convention, a=drawn.a,
            b=Task(
                label="B", task_id=drawn.b.task_id, surface=surface.name,
                table=crafted, text=drawn.b.text,
                key=tuple(GENERATOR.key_for(crafted, convention)),
            ),
            envelope=drawn.envelope,
        )

    picked = instance_of("route_origin")
    assert checks.check_materiality(GENERATOR, picked, 1).passed
    assert not checks.check_materiality(GENERATOR, instance_of("route_gift"), 1).passed

    verdicts = {
        route: validation.check_retail_support(GENERATOR, instance_of(route), MASTER)
        for route in ("route_origin", "route_gift", "route_usedgift")
    }
    assert {result.passed for result in verdicts.values()} == {False}
    assert len({result.detail for result in verdicts.values()}) == 1


def test_adding_this_genre_changed_nothing_ledger_or_soundchange_does() -> None:
    """The two families already on the roster, frozen at the base of this branch.

    It fails if the drawn convention, the committed instance digest, the whole
    admission verdict digest, the list of checks run, the copy maxima, the axis
    leverage, either task text, either answer key, or any of the three cells under any
    of the seven registered filing classes moves on either family. The fixture was
    generated from the unmodified checkout this branch starts from.
    """
    from shogym.envs.receipts import admission, streams
    from shogym.envs.receipts.generators import ledger, soundchange

    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "_fixtures"
         / "receipts_before_retail.json").read_text(encoding="utf-8")
    )
    for name, generator in (
        ("ledger", ledger.GENERATOR), ("soundchange", soundchange.GENERATOR)
    ):
        for ordinal, expected in sorted(fixture[name].items()):
            instance = protocol.draw(generator, MASTER, int(ordinal))
            assert dict(instance.convention) == expected["convention"]
            assert bank_mod.instance_digest(instance, generator) == expected[
                "instance_digest"
            ]
            assert admission.report(
                generator, instance, MASTER, admission.Thresholds()
            ).digest() == expected["report_digest"]
            assert [
                c.name for c in checks.run_checks(
                    generator, instance, MASTER, max_copy_score=0.5,
                    max_flip_score=0.875, min_leverage=0.10,
                )
            ] == expected["check_names"]
            assert {
                k: round(v, 12)
                for k, v in checks.copy_scores(generator, instance).items()
            } == expected["copy_scores"]
            assert {
                k: round(v, 12)
                for k, v in checks.axis_leverage(generator, instance).items()
            } == expected["leverage"]
            for side in ("a", "b"):
                task = instance.side(side)
                envelope = frozen_envelope(instance.envelope)
                assert streams.digest(task.text.encode()) == expected[side][
                    "text_digest"
                ]
                assert list(task.key) == expected[side]["key"]
                assert list(generator.row_identifiers(task.table)) == expected[side][
                    "identifiers"
                ]
                for shape in checks.FILING_CLASSES:
                    raw = checks.filing_of(generator, instance, side, shape)
                    canonical = generator.parse_and_canonicalize(task, raw)
                    judged = judge_cells(
                        generator, task, canonical, instance.convention, envelope
                    )
                    assert judged.problems == ()
                    assert {
                        kind: streams.digest(payload)
                        for kind, payload in sorted(judged.payloads.items())
                    } == expected[side]["cells"][shape]
                    assert judged.score == expected[side]["scores"][shape]


# --------------------------------------------------------------------------
# registration, under the shared gate label and the shared renderer
# --------------------------------------------------------------------------


def test_the_genre_is_registered_and_both_of_its_modules_are_in_the_code_pin() -> None:
    """The one door every command reaches a generator through, and what it pins.

    It fails if the genre is not on the roster, if the door hands back a generator
    whose copy profile was never read, if the gate label or the renderer
    configuration moved to register it, or if either of this family's two modules is
    outside the code pin: a decider a bundle does not hash is a decider that can move
    under a certified bundle.
    """
    from shogym.envs.receipts import admission
    from shogym.envs.receipts.registry import GENRES, load_generator, module_path

    assert GENRES["retail_refund"] == "shogym.envs.receipts.generators.retail_refund"
    loaded = load_generator("retail_refund")
    assert loaded is GENERATOR
    assert copy_profiles.profile_of(loaded) == copy_profiles.ORDERED_TOKENS
    assert module_path("retail_refund").name == "retail_refund.py"

    # Registered under the labels already on the branch, not under new ones.
    assert admission.GATE_VERSION == "receipts-gates-v3"
    assert bank_mod.RENDERER_CONFIGURATION == "receipts-render-v2"

    pinned = bank_mod.pinned_modules(loaded)
    assert "shogym.envs.receipts.generators.retail_refund" in pinned
    assert "shogym.envs.receipts.generators.retail_validation" in pinned
    assert "shogym.envs.receipts.registry" not in pinned
    pin = bank_mod.code_pin(loaded)
    assert set(pin["modules"]) == set(pinned)
    assert pin["modules"]["shogym.envs.receipts.generators.retail_validation"]


def test_the_shipped_commands_materialize_gate_check_and_draw_this_genre(
    tmp_path, monkeypatch, capsys
) -> None:
    """The four commands a reader uses, run through the real command line.

    It fails if materialization cannot fill a small bank of this genre under the
    registered bars, if the frozen bank does not rebuild to the same population, if
    `gate` or `check` rejects an instance the bank holds, or if `draw` does not print
    both task texts and three cells of the registered envelope size.
    """
    from shogym.cli import main
    from shogym.envs.receipts.registry import BANK_DIR_VAR

    monkeypatch.setenv(BANK_DIR_VAR, str(tmp_path / "banks"))

    def run(argv: list[str]) -> int:
        try:
            main(argv)
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    assert run(["receipts", "materialize", "retail_refund", "--size", "2"]) == 0
    made = capsys.readouterr().out
    assert "materialized 2 instances" in made
    assert "receipts-gates-v3" in made
    assert run(["receipts", "gate", "retail_refund", "--instances", "2"]) == 0
    assert "0 of 2 instances rejected" in capsys.readouterr().out
    assert run(["receipts", "check", "retail_refund", "--instances", "2"]) == 0
    assert "0 of 2 instances failed" in capsys.readouterr().out
    assert run(["receipts", "draw", "retail_refund", "--side", "a", "--filing", "mixed"]) == 0
    drawn_out = capsys.readouterr().out
    assert "economic destination choice" in drawn_out
    assert "refund destination schedule" in drawn_out
    assert "all three match the envelope: True" in drawn_out


# --------------------------------------------------------------------------
# the review pack, the bundle and the served route
# --------------------------------------------------------------------------


def _screen_artifact(pairs: int = 40) -> dict:
    """A recorded room screen for this family, structurally valid and not measured here.

    The numbers stand in for a pilot nobody ran in a test. What is being exercised is
    the path that recomputes everything mechanical about a bundle, not the screen.
    """
    return {
        "family": GENERATOR.name,
        "model": "a scripted policy",
        "task_seeds": [str(i) for i in range(pairs)],
        "pairs": [
            {"instance": f"task-{i:02d}", "filing": f"filing-{i:02d}",
             "placebo": 0.3, "graded": 0.7, "oracle": 0.95}
            for i in range(pairs)
        ],
        "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36,
        "floor": 0.0, "floor_rule": "drop",
        "candidates_screened": 1, "selection_note": "",
    }


#: The key this module's bank is built under. FIXED, because a pack carries every
#: worksheet case and a bank of two need not exhibit all of them under a fresh key
#: every run, so what is exercised here is the exporter rather than the draw.
PACK_MASTER = hashlib.sha256(b"retail-review-pack").digest()


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory):
    """One small bank of this genre, its exported pack, and a bundle that verifies."""
    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.envs.receipts import retail_review

    room = tmp_path_factory.mktemp("retail")
    bank, held = bank_mod.materialized(GENERATOR, PACK_MASTER, 2)
    outcomes = room / "screen.json"
    outcomes.write_text(json.dumps(_screen_artifact()), encoding="utf-8")
    pack_root = room / "pack"
    pack = retail_review.export(bank, held, pack_root)
    # The exporter leaves the attestation unset, so the bundle refuses the pack until a
    # person has put their name to it. That refusal is the point of leaving it unset.
    with pytest.raises(ValueError):
        bundle_mod.build(room / "bundles", GENERATOR, bank, outcomes, pack)
    retail_review.attested(pack_root, "a named reader")
    built = bundle_mod.build(room / "bundles", GENERATOR, bank, outcomes, pack)
    assert bundle_mod.verify(built, GENERATOR).problems == ()
    return bank, held, built


def test_the_review_pack_covers_the_family_and_names_no_reviewer(
    frozen, tmp_path: Path
) -> None:
    """What the exported pack has to contain, and the one thing it must not.

    It fails if the pack misses a surface template, an option of any axis, a registered
    filing class, the row count the bank holds or a counterfactual render, if the 27
    oracle cells are not all there, if a counterfactual is missing for any option of
    any axis on any surface, if the worksheets do not carry every phenomenon, the
    equal-amount fixtures, the separating matrix and the validation and copying
    numbers, or if the exporter names a reviewer. It also fails if the worksheets are
    labelled as renders: they are explanatory material, they are not served tasks, and
    a bundle carrying them as evidence of what was served would be saying something
    nobody checked.
    """
    from shogym.envs.receipts import review, retail_review
    from shogym.envs.receipts.streams import digest

    bank, held, _ = frozen
    room = tmp_path / "again"
    pack = retail_review.export(bank, held, room)
    manifest = json.loads(pack.read_text(encoding="utf-8"))
    assert set(manifest) == set(review.REQUIRED_FIELDS)
    assert manifest["reviewer"] is None
    assert manifest["family"] == "retail_refund"
    assert manifest["bank"] == bank_mod.bank_identity(bank)
    assert manifest["seeds"] == list(held.ordinals)

    coverage = review.required_coverage(
        GENERATOR, checks.FILING_CLASSES, [retail.ROWS]
    )
    seen = [(e["category"], e["key"]) for e in manifest["renders"]]
    assert coverage.missing(seen) == []
    oracles = {
        e["path"] for e in manifest["renders"] if e["path"].startswith("renders/oracle-")
    }
    assert len(oracles) == 27
    for surface in GENERATOR.surface_templates():
        for axis in retail.AXES:
            for option in axis.options:
                assert ("counterfactual", f"{surface} {axis.name}={option}") in seen
    for entry in manifest["renders"]:
        assert not entry["path"].startswith(retail_review.WORKSHEETS)
        assert (room / entry["path"]).is_file()

    index = json.loads((room / retail_review.WORKSHEET_INDEX).read_text("utf-8"))
    assert index["cases"] == list(retail_review.WORKSHEET_CASES)
    for name, hashed in index["files"].items():
        assert digest((room / name).read_bytes()) == hashed
    for name in ("cases", "ties", "matrix", "validation"):
        assert f"{retail_review.WORKSHEETS}/{name}.json" in index["files"]

    ties = json.loads(
        (room / retail_review.WORKSHEETS / "ties.json").read_text("utf-8")
    )
    assert len(ties["cases"]) == len(retail_review.TIE_FIXTURES)
    matrix = json.loads(
        (room / retail_review.WORKSHEETS / "matrix.json").read_text("utf-8")
    )
    assert matrix["distinct_answer_vectors"] == 27
    assert len(matrix["cases"]) == 8
    numbers = json.loads(
        (room / retail_review.WORKSHEETS / "validation.json").read_text("utf-8")
    )
    assert numbers["pair"]["copying"]["shipped_family_bound"] == 0.50
    assert numbers["pair"]["copying"]["all_bijection_bound"] == 0.50
    assert all(c["masked_bytes_equal"] for c in numbers["cell_comparisons"])
    sheet = json.loads(
        (room / retail_review.WORKSHEETS
         / "apparel-route_policy-route_gift.json").read_text("utf-8")
    )
    assert len(sheet["rows"]) == retail.ROWS
    assert {row["class_selected"] for row in sheet["rows"]} == {
        "gift cards", "original instruments"
    }
    # Under unconditional gift preference every original-only case reaches its class
    # through the public fallback, and under the conditional route none does.
    assert sum(1 for row in sheet["rows"] if row["fallback_applied"]) == retail.PER_KIND
    conditional = json.loads(
        (room / retail_review.WORKSHEETS
         / "apparel-route_policy-route_usedgift.json").read_text("utf-8")
    )
    assert not any(row["fallback_applied"] for row in conditional["rows"])

    # The exported pack is the same pack twice, so two readers read one document.
    twice = tmp_path / "twice"
    retail_review.export(bank, held, twice)
    assert (twice / retail_review.PACK).read_bytes() == pack.read_bytes()


def test_a_frozen_bank_rebuilds_and_a_failed_extra_check_is_not_dealable(frozen) -> None:
    """Replay, rebuild, and what happens when one of the declared checks says no.

    It fails if a small valid frozen bank cannot be rebuilt byte-identically, if an
    instance failing one of the three checks this family declares is still admitted or
    still fills a bank, or if replaying a committed fork changes its cells or answers
    one filing with another filing's feedback.
    """
    from shogym.envs.receipts import admission

    bank, held, built = frozen
    again = bank_mod.population(bank, GENERATOR)
    assert again.ordinals == held.ordinals
    for first, second in zip(held.instances, again.instances):
        assert bank_mod.instance_digest(first, GENERATOR) == bank_mod.instance_digest(
            second, GENERATOR
        )

    instance = held.instances[0]
    assert admission.report(
        GENERATOR, instance, bank.master, admission.Thresholds()
    ).admitted
    kept = validation.NAMED_CHECKS["retail_surface"]
    try:
        validation.NAMED_CHECKS["retail_surface"] = lambda *_: checks.CheckResult(
            "retail_surface", False, "a fixture refusal"
        )
        report = admission.report(
            GENERATOR, instance, bank.master, admission.Thresholds()
        )
        assert "retail_surface" in report.failed_checks
        assert not report.admitted
        with pytest.raises(ValueError):
            bank_mod.population(bank, GENERATOR)
    finally:
        validation.NAMED_CHECKS["retail_surface"] = kept

    room = Path(built.root).parent / "forks"
    source = built.digest
    task = instance.a
    identifiers = list(GENERATOR.row_identifiers(task.table))
    perfect = "\n".join("%s,%s" % kv for kv in zip(identifiers, task.key))
    wrong = "\n".join("%s,%s" % (i, "zz") for i in identifiers)
    keyed = bank_mod.filing_digest(perfect)
    assert bank_mod.load_fork(room, task.task_id, keyed, source) is None
    first = bank_mod.fork_for(GENERATOR, instance, "a", perfect, room, source)
    assert bank_mod.load_fork(room, task.task_id, keyed, source) is not None
    replayed = bank_mod.fork_for(GENERATOR, instance, "a", perfect, room, source)
    assert replayed.replayed
    assert (replayed.graded, replayed.placebo, replayed.oracle) == (
        first.graded, first.placebo, first.oracle
    )
    assert replayed.component_score == first.component_score == 1.0
    other = bank_mod.fork_for(GENERATOR, instance, "a", wrong, room, source)
    assert other.graded != first.graded
    assert other.component_score == 0.0
    assert other.filing_digest != first.filing_digest


async def test_a_filing_an_independent_reader_believes_seals_at_one(
    frozen, tmp_path: Path
) -> None:
    """What the served environment does with a correct filing, an empty one and a grade.

    It fails if a filing computed by the second implementation reading the printed body
    does not seal at 1, if an empty filing is credited, if prose is scored rather than
    reason coded, or if either channel of the terminal result carries a grade: what one
    graded receipt is worth is the quantity this environment exists to measure, so a
    terminal that handed the grade back would put a receipt in every arm including the
    one meant to carry none.
    """
    from shogym.envs.receipts.env_v1 import ReceiptsV1Env
    from shogym.serve import ServedEpisode

    _, _, built = frozen
    private = tmp_path / "served"
    private.mkdir()
    opened = private / built.root.name
    shutil.copytree(built.root, opened)
    config = {"genre": "retail_refund", "bundle": str(opened), "side": "a"}
    env = ReceiptsV1Env(**config)
    ordinal = env._ordinals[0]
    instance = env.instance(ordinal)
    task = instance.a
    independent = validation.audit_key(
        [c.instruments for c in validation.parsed_cases(task.table)], instance.convention
    )
    assert independent == tuple(task.key)
    filing = "\n".join(
        "%s,%s" % kv
        for kv in zip(GENERATOR.row_identifiers(task.table), independent)
    )

    episode = await ServedEpisode.start(
        "receipts_v1", task=0, env_config=config, trace_path=tmp_path / "run.jsonl"
    )
    try:
        spec = episode.describe()
        assert "SCHEDULE (" in spec.instructions
        assert option_mentions(retail.AXES, json.dumps(spec.model_dump())) == []
        result = await episode.call("submit_filing", {"filing": filing})
        assert result.terminated
        content = json.loads(result.content)
        assert set(content) == {"filed", "rows", "finalize_error"}
        assert "1.0" not in json.dumps(content)
        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["component_score"] == 1.0
        assert feedback["rows_omitted"] == 0.0
    finally:
        await episode.close()

    blank = "\n".join(
        "%s," % identifier for identifier in GENERATOR.row_identifiers(task.table)
    )
    empty = await ServedEpisode.start(
        "receipts_v1", task=0, env_config=config, trace_path=tmp_path / "empty.jsonl"
    )
    try:
        await empty.call("submit_filing", {"filing": blank})
        feedback = {item["name"]: item["value"] for item in empty.terminal_feedback}
        assert feedback["component_score"] == 0.0
        assert feedback["rows_filed"] == float(retail.ROWS)
        assert "no_filing" not in feedback
    finally:
        await empty.close()

    unread = await ServedEpisode.start(
        "receipts_v1", task=0, env_config=config, trace_path=tmp_path / "prose.jsonl"
    )
    try:
        await unread.call(
            "submit_filing", {"filing": "I could not work out the convention"}
        )
        feedback = {item["name"]: item["value"] for item in unread.terminal_feedback}
        assert feedback["component_score"] == 0.0
        assert feedback["no_filing"] == "no_known_identifier"
    finally:
        await unread.close()
