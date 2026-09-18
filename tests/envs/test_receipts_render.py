"""The render path: the AST, the serializer, the envelope, and the three cells.

The properties checked here are the ones the whole design rests on. The three cells
of a fork total one envelope. The graded and the placebo differ ONLY inside the
registered slots, so a placebo carries nothing the slots do not account for, and no
byte of either moves with the drawn convention outside them. The oracle states the
rule and can be read back. And the same canonical filing feeds the scorer and both
renderers, so a receipt can never grade something the score did not.
"""

from __future__ import annotations

import pytest

from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import streams
from shogym.envs.receipts.generators import ledger
from shogym.envs.receipts.generators.ledger import GENERATOR
from shogym.envs.receipts.protocol import (
    NoFiling,
    SealedSubmission,
    conventions,
    draw,
    option_mentions,
)
from shogym.envs.receipts.receipt_ast import (
    GRADED,
    ORACLE,
    PLACEBO,
    ReceiptAST,
    frozen_envelope,
    mask_slots,
    serialize,
    slot_ranges,
)

MASTER = bytes(range(32))
ORDINALS = (0, 1, 2)


def _instance(ordinal: int = 0):
    return draw(GENERATOR, MASTER, ordinal)


def _canonical(instance, side: str = "a"):
    task = instance.side(side)
    raw = "\n".join(f"{i},{v}" for i, v in zip(GENERATOR.row_identifiers(task.table), task.key))
    return GENERATOR.parse_and_canonicalize(task, raw)


# ----- the draw is deterministic, and the streams are separate -----


def test_the_same_key_and_ordinal_draw_the_same_family() -> None:
    first, second = _instance(3), _instance(3)
    assert first.convention == second.convention
    assert first.a.task_id == second.a.task_id
    assert first.a.text == second.a.text
    assert first.a.key == second.a.key


def test_a_different_key_draws_a_different_family() -> None:
    other = draw(GENERATOR, bytes(range(1, 33)), 3)
    assert other.a.task_id != _instance(3).a.task_id
    assert other.a.text != _instance(3).a.text


def test_the_task_identifier_is_opaque_without_the_key() -> None:
    """It is stable under the key, unguessable without it, and encodes no coordinate.

    Checking that a hex string does not contain the letter "a" would prove nothing.
    What matters is that the identifier is a keyed function: the same coordinates
    under a second key give an unrelated string, so possessing every identifier a
    lineage was ever served tells you how many there were and nothing else.
    """
    instance = _instance(7)
    assert len(instance.a.task_id) == 16
    assert set(instance.a.task_id) <= set("0123456789abcdef")
    assert instance.a.task_id != instance.b.task_id
    assert instance.a.task_id != _instance(8).a.task_id
    # the same coordinates under another key are unrelated
    other = streams.task_identifier(bytes(range(64, 96)), "ledger", 7, "A")
    assert other != instance.a.task_id
    # and consecutive ordinals share no prefix, so ordering is not readable off them
    ids = [_instance(o).a.task_id for o in range(6)]
    assert len({i[:4] for i in ids}) == len(ids)


def test_two_streams_under_different_labels_are_independent() -> None:
    first = streams.derive(MASTER, streams.SURFACE_A, 1)
    second = streams.derive(MASTER, streams.SURFACE_B, 1)
    assert first != second
    # and the length prefixing means no two coordinate tuples collide
    assert streams.derive(MASTER, streams.FILLER, "a|b", "c") != streams.derive(
        MASTER, streams.FILLER, "a", "b|c"
    )


def test_the_sampler_reaches_the_whole_option_product() -> None:
    space = {tuple(sorted(c.items())) for c in conventions(GENERATOR.AXES)}
    drawn = {
        tuple(sorted(_draw_convention(o).items())) for o in range(4000)
    }
    assert drawn == space


def _draw_convention(ordinal: int):
    from shogym.envs.receipts.protocol import draw_convention

    return draw_convention(GENERATOR.AXES, MASTER, ordinal)


def test_an_undeclared_stream_label_is_refused() -> None:
    with pytest.raises(ValueError, match="undeclared stream label"):
        streams.derive(MASTER, "made-up", 1)


# ----- both siblings under one convention -----


def test_both_siblings_are_scored_under_the_one_drawn_convention() -> None:
    for ordinal in ORDINALS:
        instance = _instance(ordinal)
        assert instance.a.key == ledger.key_for(instance.a.table, instance.convention)
        assert instance.b.key == ledger.key_for(instance.b.table, instance.convention)
        assert instance.a.surface != instance.b.surface
        assert instance.a.text != instance.b.text


def test_every_axis_moves_the_sibling_task() -> None:
    for ordinal in ORDINALS:
        instance = _instance(ordinal)
        base = ledger.key_for(instance.b.table, instance.convention)
        for axis in GENERATOR.AXES:
            for option in axis.options:
                if option == instance.convention[axis.name]:
                    continue
                alt = dict(instance.convention)
                alt[axis.name] = option
                other = ledger.key_for(instance.b.table, alt)
                assert sum(1 for x, y in zip(base, other) if x != y) >= 1


def test_no_task_text_names_an_option() -> None:
    for ordinal in range(6):
        instance = _instance(ordinal)
        for task in (instance.a, instance.b):
            assert option_mentions(GENERATOR.AXES, task.text) == []


def test_each_sibling_is_told_which_schedules_share_its_conventions() -> None:
    """The sentence a sibling carries, word for word, and where it sits.

    A convention is drawn for a pair of schedules, and a reader told nothing about
    that has no reason to carry to the second what it worked out on the first. So
    each sibling says it, by what the reader does with the schedules rather than by
    anything printed on them, and it says nothing about what the conventions are.
    """
    for ordinal in ORDINALS:
        instance = _instance(ordinal)
        for side, label, other in (("a", "A", "B"), ("b", "B", "A")):
            task = instance.side(side)
            mine = ledger.scope_sentence(label)
            assert _flat([mine]) in _flat([task.text])
            assert _flat([ledger.scope_sentence(other)]) not in _flat([task.text])
            # after the holiday list the policy extract ends on, before the schedule
            assert task.text.index("covered by this schedule are:") < task.text.index(mine)
            assert task.text.index(mine) < task.text.index("\nSCHEDULE (")
            # and it names no option of the hidden rule
            assert option_mentions(GENERATOR.AXES, mine) == []
    assert ledger.scope_sentence("A") == (
        "This schedule and the next one you file are scored under the same house\n"
        "conventions; the pair after that is scored under conventions of its own."
    )
    assert ledger.scope_sentence("B") == (
        "This schedule is scored under the same house conventions as the one you filed\n"
        "before it."
    )


def test_the_description_gained_that_sentence_and_no_other_public_text() -> None:
    """Take the sentence back out and the text is the one the task printed before.

    Which is the whole claim: the two siblings' descriptions differ in that sentence
    and in the surface data they already differed in, and in nothing else.
    """
    instance = _instance()
    bare = ledger.TASK_TEMPLATE.replace("{scope}\n{receipt}\n", "")
    assert "{scope}" not in bare and "{receipt}" not in bare
    for side, label in (("a", "A"), ("b", "B")):
        task = instance.side(side)
        dom = task.table.dom
        before = bare.format(
            org=dom["org"],
            title=dom["title"],
            ref=dom["refdate"].isoformat(),
            entity=dom["entity"],
            manual=dom["manual"],
            table=ledger.band_table(dom),
            unit=dom["unit"],
            hol="\n".join(
                "    %s   %s" % (d.isoformat(), n)
                for d, n in zip(task.table.holidays, dom["holnames"])
            ),
            fmtnote=ledger.FMT_NOTE[dom["fmt"]],
            body=task.table.body,
        )
        sentence = ledger.scope_sentence(label)
        # The scope sentence and, under a policy that samples, the registered sentence
        # about what the receipt will report: both are what the description gained, and
        # the text with both taken out is the text this task printed before either.
        added = (
            sentence + "\n" + ledger.receipt_sentence(GENERATOR.RECEIPT_POLICY) + "\n"
        )
        assert task.text.count(sentence) == 1
        assert task.text.replace(added, "", 1) == before


def test_a_sibling_label_the_family_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="siblings A and B"):
        ledger.scope_sentence("C")


# ----- the envelope -----


def test_all_three_cells_total_the_envelope() -> None:
    instance = _instance()
    fork = bank_mod.render_fork(GENERATOR, instance, "a", _raw(instance, "a"))
    for kind in (GRADED, PLACEBO, ORACLE):
        assert len(fork.agent_bytes(kind)) == instance.envelope.size


def test_graded_and_placebo_differ_only_inside_the_registered_slots() -> None:
    for ordinal in ORDINALS:
        instance = _instance(ordinal)
        for side in ("a", "b"):
            canonical = _canonical(instance, side)
            task = instance.side(side)
            graded = GENERATOR.render_receipt(
                task, canonical, task.key, _feedback(instance, task)
            )
            placebo = GENERATOR.render_placebo(task, canonical, instance.envelope)
            ranges = slot_ranges(graded, instance.envelope)
            first = serialize(graded, instance.envelope)
            second = serialize(placebo, instance.envelope)
            assert first != second
            assert mask_slots(first, ranges) == mask_slots(second, ranges)


def test_no_byte_outside_the_slots_moves_with_the_drawn_convention() -> None:
    """The whole convention support, on one surface and one filing.

    A byte outside the slots that moved with the convention would be a channel the
    envelope check does not price and the gates do not read.
    """
    instance = _instance()
    task = instance.side("a")
    canonical = _canonical(instance, "a")
    reference = None
    ranges = None
    seen_keys = set()
    for convention in conventions(GENERATOR.AXES):
        key = ledger.key_for(task.table, convention)
        seen_keys.add(key)
        graded = GENERATOR.render_receipt(
            task, canonical, key, _feedback(instance, task)
        )
        payload = serialize(graded, instance.envelope)
        if ranges is None:
            ranges = slot_ranges(graded, instance.envelope)
            reference = mask_slots(payload, ranges)
        assert len(payload) == instance.envelope.size
        assert mask_slots(payload, ranges) == reference
    assert len(seen_keys) > 1


def test_the_placebo_cannot_see_the_convention_at_all() -> None:
    """It takes no convention and no key, so its bytes cannot move with either."""
    instance = _instance()
    task = instance.side("a")
    canonical = _canonical(instance, "a")
    once = serialize(
        GENERATOR.render_placebo(task, canonical, instance.envelope), instance.envelope
    )
    for convention in conventions(GENERATOR.AXES)[:12]:
        ledger.key_for(task.table, convention)
        again = serialize(
            GENERATOR.render_placebo(task, canonical, instance.envelope), instance.envelope
        )
        assert again == once


def test_the_placebo_prints_no_verdict_token() -> None:
    instance = _instance()
    canonical = _canonical(instance, "a")
    text = serialize(
        GENERATOR.render_placebo(instance.a, canonical, instance.envelope), instance.envelope
    ).decode()
    assert "PASS" not in text
    assert "FAIL" not in text


def test_a_cell_that_overruns_its_envelope_is_refused() -> None:
    instance = _instance()
    small = type(instance.envelope)(
        size=64,
        identifier_width=instance.envelope.identifier_width,
        observed_width=instance.envelope.observed_width,
        slots=instance.envelope.slots,
        filler=instance.envelope.filler,
        column_titles=instance.envelope.column_titles,
        neutral=instance.envelope.neutral,
    )
    canonical = _canonical(instance, "a")
    with pytest.raises(ValueError, match="registered envelope"):
        serialize(
            GENERATOR.render_receipt(
                instance.a, canonical, instance.a.key, _feedback(instance, instance.a)
            ),
            small,
        )


def test_a_non_ascii_field_is_refused_rather_than_shifting_every_offset() -> None:
    instance = _instance()
    ast = ReceiptAST(kind=ORACLE, task_id="0" * 16, row_count=0, body=("café",))
    with pytest.raises(ValueError, match="ascii"):
        serialize(ast, instance.envelope)


@pytest.mark.parametrize(
    "value",
    ["Routïne", "a “quoted” value", "\x1b[7mLOOK HERE\x1b[0m", "A\x00B", "🙂"],
    ids=["accented", "curly-quotes", "escape-sequence", "nul-byte", "emoji"],
)
def test_a_filing_the_agent_can_type_always_renders_three_cells(value: str) -> None:
    """The rule above is for a field the RENDERER authored. The observed column is the
    agent's, and the tool takes any string, so a value that could not be serialized
    would let an agent decide whether its own fork exists: the seal would raise, the
    episode would fail closed, and no cell of any kind would be committed for that
    link. The parser folds to printable ASCII instead, so the scorer and both
    renderers see one value and the three cells are always made.
    """
    from shogym.envs.receipts.render import judge_cells

    instance = _instance()
    task = instance.a
    identifiers = GENERATOR.row_identifiers(task.table)
    lines = [f"{i},{v}" for i, v in zip(identifiers, task.key)]
    lines[0] = f"{identifiers[0]},{value}"
    canonical = GENERATOR.parse_and_canonicalize(task, "\n".join(lines))
    judged = judge_cells(
        GENERATOR, task, canonical, instance.convention, instance.envelope
    )
    assert judged.problems == ()
    assert {len(p) for p in judged.payloads.values()} == {instance.envelope.size}
    # The parity that makes the placebo inert holds on the folded value too, and
    # nothing unprintable reaches the arm that is meant to be empty.
    committed = frozen_envelope(instance.envelope)
    ranges = slot_ranges(judged.asts[GRADED], committed)
    assert mask_slots(judged.payloads[GRADED], ranges) == mask_slots(
        judged.payloads[PLACEBO], ranges
    )
    for payload in judged.payloads.values():
        assert b"\x1b" not in payload and b"\x00" not in payload


# ----- the receipt grades what the filing did -----


def test_the_receipt_names_records_and_never_an_axis() -> None:
    instance = _instance()
    canonical = _canonical(instance, "a")
    text = serialize(
        GENERATOR.render_receipt(
            instance.a, canonical, instance.a.key, _feedback(instance, instance.a)
        ),
        instance.envelope,
    ).decode()
    for axis in GENERATOR.AXES:
        assert axis.name not in text
        for option in axis.options:
            assert option not in text
    for row in instance.a.table.rows:
        assert row.row_id in text


def test_the_scorer_and_the_receipt_read_the_same_canonical_filing() -> None:
    instance = _instance()
    task = instance.a
    identifiers = GENERATOR.row_identifiers(task.table)
    raw = "\n".join(
        f"{identifier},{value if i % 2 else 'Wrong'}"
        for i, (identifier, value) in enumerate(zip(identifiers, task.key))
    )
    fork = bank_mod.render_fork(GENERATOR, instance, "a", raw)
    graded = fork.graded.decode()
    # The receipt reports the rows its committed mask drew, and the verdict on each of
    # them is the one the scorer recorded for that row. The score is still over every
    # row: what the policy changes is what is reported, not what is graded.
    reported = [o for o in fork.outcomes if (o.ordinal - 1) in task.mask]
    passes = sum(1 for o in reported if o.matched)
    assert graded.count("PASS") == passes
    assert graded.count("FAIL") == len(reported) - passes
    scored = sum(1 for o in fork.outcomes if o.matched)
    assert fork.component_score == pytest.approx(scored / len(fork.outcomes))


# ----- the oracle states the rule and can be read back -----


def test_every_convention_survives_a_round_trip_through_the_oracle() -> None:
    for convention in conventions(GENERATOR.AXES):
        ast = GENERATOR.render_oracle("0" * 16, convention, 24)
        assert GENERATOR.parse_oracle(ast) == convention


def test_the_oracle_states_the_rule_and_no_answers() -> None:
    instance = _instance()
    ast = GENERATOR.render_oracle(instance.a.task_id, instance.convention, instance.a.n_rows)
    text = "\n".join(ast.body)
    assert "HOUSE CONVENTIONS" in text
    for row in instance.a.table.rows:
        assert row.row_id not in text


def _flat(lines) -> str:
    """The cell's prose with its wrapping taken out, which is how a phrase is read."""
    return " ".join(" ".join(lines).split())


def _rule_phrases(convention) -> tuple[str, ...]:
    return tuple(
        ledger.ORACLE_TEMPLATE.phrases[axis][option] for axis, option in convention.items()
    )


def test_the_oracle_heads_the_rule_with_the_schedules_the_reader_files() -> None:
    """The scope is named by the schedules the reader files, not by a word it can
    map onto the organisation printed on one of them.

    A rule is drawn for a pair of sibling schedules, and the two siblings are always
    different organisations. A heading that scoped the rule by anything printed on
    the surface therefore told the reader the rule did not reach the schedule it was
    about to file, and a reader that believed that filed under its own older rule,
    out of the arm the room is measured against.
    """
    instance = _instance()
    ast = GENERATOR.render_oracle(instance.a.task_id, instance.convention, instance.a.n_rows)
    text = _flat(ast.body)
    assert text.startswith("HOUSE CONVENTIONS FOR THIS SCHEDULE AND THE NEXT ONE YOU FILE")
    assert (
        "they govern the schedule above and the next schedule you are given, "
        "whichever organisation it names" in text
    )
    assert "the schedule after that is scored under conventions of its own" in text
    assert "engagement" not in text
    assert "standing" not in text
    assert "every schedule" not in text
    # and under that heading, every decision the rule makes
    for phrase in _rule_phrases(instance.convention):
        assert _flat([phrase]) in text
    assert GENERATOR.parse_oracle(ast) == instance.convention


def test_no_cell_of_a_fork_states_a_rule_another_schedule_was_drawn() -> None:
    """The oracle states this schedule's rule; the other two state no rule at all.

    The rule another schedule took is the thing a reader would have to reconcile
    with this one, so no cell served here may name it, and the graded and placebo
    cells name no rule whatever.
    """
    instance = _instance()
    other = {
        axis.name: next(o for o in axis.options if o != instance.convention[axis.name])
        for axis in GENERATOR.AXES
    }
    raw = "\n".join(
        f"{identifier},{value}"
        for identifier, value in zip(
            GENERATOR.row_identifiers(instance.a.table), instance.a.key
        )
    )
    fork = bank_mod.render_fork(GENERATOR, instance, "a", raw)
    oracle = _flat([fork.oracle.decode()])
    for phrase in _rule_phrases(other):
        assert _flat([phrase]) not in oracle
    for phrase in _rule_phrases(instance.convention):
        assert _flat([phrase]) in oracle
    for cell in (fork.graded, fork.placebo):
        text = _flat([cell.decode()])
        for convention in (instance.convention, other):
            for phrase in _rule_phrases(convention):
                assert _flat([phrase]) not in text


# ----- reading a filing -----


def test_a_full_filing_reads_back_row_for_row() -> None:
    instance = _instance()
    canonical = _canonical(instance, "a")
    assert isinstance(canonical, SealedSubmission)
    assert canonical.values == instance.a.key
    assert canonical.omissions == ()
    assert canonical.extras == ()
    assert canonical.duplicates == ()


def test_the_parser_records_duplicates_extras_and_omissions() -> None:
    instance = _instance()
    task = instance.a
    identifiers = GENERATOR.row_identifiers(task.table)
    raw = "\n".join(
        [f"{identifiers[0]},Routine", f"{identifiers[0].lower()},Urgent", "ZZ-1,Urgent"]
    )
    canonical = GENERATOR.parse_and_canonicalize(task, raw)
    assert isinstance(canonical, SealedSubmission)
    assert canonical.values[0] == "Routine"  # the first line for an identifier wins
    assert canonical.duplicates == (identifiers[0],)
    assert canonical.extras == ("ZZ-1",)
    assert len(canonical.omissions) == len(identifiers) - 1


def test_no_filing_is_reason_coded_and_never_raised() -> None:
    instance = _instance()
    task = instance.a
    assert GENERATOR.parse_and_canonicalize(task, "").reason == "empty"
    assert GENERATOR.parse_and_canonicalize(task, None).reason == "unreadable"
    assert GENERATOR.parse_and_canonicalize(task, {"a": 1}).reason == "unreadable"
    prose = GENERATOR.parse_and_canonicalize(task, "I could not work out the rule")
    assert isinstance(prose, NoFiling)
    assert prose.reason == "no_known_identifier"


def test_a_comma_free_filing_is_positional_only_at_the_right_length() -> None:
    instance = _instance()
    task = instance.a
    exact = GENERATOR.parse_and_canonicalize(task, "\n".join(task.key))
    assert isinstance(exact, SealedSubmission)
    assert exact.values == task.key
    short = GENERATOR.parse_and_canonicalize(task, "\n".join(task.key[:3]))
    assert isinstance(short, NoFiling)


def test_a_no_filing_still_renders_a_congruent_pair() -> None:
    instance = _instance()
    fork = bank_mod.render_fork(GENERATOR, instance, "a", "")
    assert fork.component_score == 0.0
    assert bank_mod.no_filing_reason(fork.canonical) == "empty"
    assert len(fork.graded) == len(fork.placebo) == instance.envelope.size


def test_filing_nothing_scores_nothing_even_when_the_answer_is_empty() -> None:
    """The `missing` axis has an empty option, and a row nobody filed must not take it.

    Without the filed mask, an agent that submitted nothing at all would match every
    undated row whenever that option was drawn, and collect a sixth of the table for
    free. Filing an empty value and filing nothing are different acts.
    """
    instance = _instance()
    blank = dict(instance.convention)
    blank["missing"] = "blank"
    task = _retasked(instance.a, ledger.key_for(instance.a.table, blank))
    assert "" in task.key  # the undated rows take the empty band under this draw

    nothing = GENERATOR.parse_and_canonicalize(task, "")
    assert isinstance(nothing, NoFiling)
    assert GENERATOR.score(task, nothing)[0] == 0.0

    # and a filing that omits those rows while getting the rest right scores only
    # the rows it actually filed
    identifiers = GENERATOR.row_identifiers(task.table)
    filed_rows = [
        (i, v) for i, v in zip(identifiers, task.key) if v
    ]
    partial = GENERATOR.parse_and_canonicalize(
        task, "\n".join(f"{i},{v}" for i, v in filed_rows)
    )
    score, outcomes = GENERATOR.score(task, partial)
    assert score == round(len(filed_rows) / len(identifiers), 6)
    assert all(o.matched is False for o in outcomes if not o.was_filed)

    # filing the empty value explicitly does match
    explicit = GENERATOR.parse_and_canonicalize(
        task, "\n".join(f"{i},{v}" for i, v in zip(identifiers, task.key))
    )
    assert GENERATOR.score(task, explicit)[0] == 1.0


def test_the_receipt_tells_an_unfiled_row_from_an_empty_one() -> None:
    instance = _instance()
    task = instance.a
    identifiers = GENERATOR.row_identifiers(task.table)
    raw = f"{identifiers[0]},"  # one row, filed empty; the rest unfiled
    canonical = GENERATOR.parse_and_canonicalize(task, raw)
    text = serialize(
        GENERATOR.render_receipt(task, canonical, task.key, _feedback(instance, task)),
        instance.envelope,
    ).decode()
    assert ledger.BLANK_TOKEN in text
    assert ledger.UNFILED_TOKEN in text


def _retasked(task, key):
    from shogym.envs.receipts.protocol import Task

    return Task(
        label=task.label, task_id=task.task_id, surface=task.surface, table=task.table,
        text=task.text, key=tuple(key), mask=task.mask,
    )


def _feedback(instance, task):
    """The commitment the shared judge would hand this task's renderer."""
    from shogym.envs.receipts.receipt_ast import frozen_envelope
    from shogym.envs.receipts.render import feedback_for

    return feedback_for(GENERATOR, task, frozen_envelope(instance.envelope))


def _raw(instance, side: str) -> str:
    task = instance.side(side)
    return "\n".join(
        f"{i},{v}" for i, v in zip(GENERATOR.row_identifiers(task.table), task.key)
    )
