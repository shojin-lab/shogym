"""The feedback-on-the-wire contract (RFC 008 §4): _meta round-trip, the terminate flag
as a separate concern from the score, and the mid-episode visibility rule."""

from __future__ import annotations

import pytest

from shogym.feedback import (
    FEEDBACK_META_KEY,
    TERMINATE_META_KEY,
    build_meta,
    parse_meta,
    select_inband,
)
from shogym.types import EpisodeFeedback, InferenceFeedback


def test_build_meta_empty_is_empty() -> None:
    assert build_meta() == {}
    assert build_meta([], terminate=False) == {}


def test_roundtrip_inference_and_episode_items() -> None:
    items = [
        InferenceFeedback(name="valid_guess", value=True, step=2),
        EpisodeFeedback(name="solved", value=1.0),
        EpisodeFeedback(name="note", value="ran out of guesses"),  # text feedback
    ]
    meta = build_meta(items, terminate=True)
    assert set(meta) == {FEEDBACK_META_KEY, TERMINATE_META_KEY}
    parsed, terminate = parse_meta(meta)
    assert terminate is True
    assert parsed == items


def test_terminate_flag_is_independent_of_feedback() -> None:
    stop_only, term = parse_meta(build_meta(terminate=True))
    assert stop_only == [] and term is True
    scored, term = parse_meta(build_meta([InferenceFeedback(name="r", value=0.5, step=1)]))
    assert term is False and len(scored) == 1


def test_parse_empty_meta() -> None:
    assert parse_meta({}) == ([], False)


def test_dump_rejects_post_construction_mutation() -> None:
    # The feedback models are mutable and don't validate on assignment; the serializer
    # must catch a mutated-to-invalid item so build_meta never emits a sidecar that
    # parse_meta would reject (the same bypass TraceRecord closes before persistence).
    item = InferenceFeedback(name="r", value=1.0, step=1)
    item.step = True  # bool masquerading as an int step
    with pytest.raises(ValueError, match="must be an int"):
        build_meta([item])

    ep = EpisodeFeedback(name="r", value=1.0)
    ep.value = {"x": 1}  # non-scalar value, off-wire
    with pytest.raises(ValueError):
        build_meta([ep])


def test_wire_rejects_non_json_scalar_value() -> None:
    from decimal import Decimal
    from fractions import Fraction

    # Pydantic would coerce these numeric types to float (Decimal("NaN") even slips past
    # a float-only finite check), producing a sidecar that breaks json.dumps or reads back
    # as nan. The wire must reject the raw non-scalar type on both boundaries.
    for bad in (Decimal("1.5"), Decimal("NaN"), Fraction(1, 3)):
        item = EpisodeFeedback(name="r", value=1.0)
        item.value = bad  # post-construction mutation bypasses model coercion
        with pytest.raises(ValueError, match="JSON scalar"):
            build_meta([item])
    for bad in (Decimal("1.5"), Decimal("NaN")):
        with pytest.raises(ValueError, match="JSON scalar"):
            parse_meta(_one({"name": "r", "value": bad, "level": "episode"}))


def test_build_meta_rejects_non_finite_value() -> None:
    # The live MCP sidecar must reject non-finite exactly as the trace does; both go
    # through dump_item, so build_meta refuses NaN/Infinity rather than emitting them.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be finite"):
            build_meta([InferenceFeedback(name="r", value=bad, step=1)])


def test_build_meta_rejects_non_boolean_terminate() -> None:
    # The serializer must not truthy-coerce either: build_meta(terminate="false")
    # would otherwise emit a real stop signal.
    for bad in ("false", "true", 1, 0):
        with pytest.raises(ValueError, match="terminate must be a boolean"):
            build_meta(terminate=bad)
    assert build_meta(terminate=True) == {TERMINATE_META_KEY: True}
    assert build_meta(terminate=False) == {}


def test_terminate_flag_rejects_non_boolean() -> None:
    # A string "false" is truthy under bool(); the contract says boolean, so a
    # malformed value must be rejected, not silently terminate the episode.
    for bad in ("false", "true", 1, 0):
        with pytest.raises(ValueError, match=TERMINATE_META_KEY):
            parse_meta({TERMINATE_META_KEY: bad})
    # Real booleans still parse.
    assert parse_meta({TERMINATE_META_KEY: False}) == ([], False)
    assert parse_meta({TERMINATE_META_KEY: True}) == ([], True)


def test_unknown_feedback_level_is_rejected() -> None:
    # A misspelled "inference" must not silently become episode feedback (which
    # would hide the per-step signal until termination).
    for bad_level in ("inferencee", "episodee", None):
        meta = {FEEDBACK_META_KEY: [{"name": "r", "value": 0.5, "level": bad_level}]}
        with pytest.raises(ValueError, match="unknown feedback level"):
            parse_meta(meta)


def test_feedback_container_must_be_a_list() -> None:
    # A present-but-malformed container must be rejected, not read as "no feedback"
    # (falsy) or crashed-into (truthy non-list).
    for bad in (None, False, 0, {"name": "r", "value": 0.5, "level": "episode"}, "x"):
        with pytest.raises(ValueError, match=FEEDBACK_META_KEY):
            parse_meta({FEEDBACK_META_KEY: bad})
    # A list of non-mapping items is caught per-item.
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_meta({FEEDBACK_META_KEY: ["not-a-dict"]})
    # An absent key still means "no feedback"; an explicit empty list is fine.
    assert parse_meta({}) == ([], False)
    assert parse_meta({FEEDBACK_META_KEY: []}) == ([], False)


def _one(item: dict) -> dict:
    return {FEEDBACK_META_KEY: [item]}


def test_parse_rejects_non_finite_value() -> None:
    # Symmetric with dump_item: json.loads accepts NaN/Infinity and the model would
    # too, so the wire parser must reject them rather than round-trip invalid JSON.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be finite"):
            parse_meta(_one({"name": "reward", "value": bad, "level": "episode"}))
        with pytest.raises(ValueError, match="must be finite"):
            parse_meta(_one({"name": "reward", "value": bad, "level": "inference", "step": 1}))


def test_dict_feedback_value_is_rejected() -> None:
    # Values are scalar (number | bool | text). A dict "demonstration" is no longer
    # part of the model, so it cannot ride the wire (nor smuggle non-JSON objects).
    with pytest.raises(ValueError):
        parse_meta(_one({"name": "r", "value": {"ref": "x"}, "level": "inference", "step": 1}))


def test_inference_step_must_be_a_plain_int() -> None:
    # Pydantic would coerce "2" -> 2 and True -> 1; the wire says step is an int.
    for bad_step in ("2", True, 2.0):
        with pytest.raises(ValueError, match="'step' must be an int"):
            parse_meta(_one({"name": "x", "value": 1.0, "level": "inference", "step": bad_step}))
    # A real int still parses.
    items, _ = parse_meta(_one({"name": "x", "value": 1.0, "level": "inference", "step": 2}))
    assert items[0].step == 2


def test_item_key_set_is_exact_per_level() -> None:
    # step on an episode item, an unknown typo key, and a missing required key are
    # all rejected rather than silently dropped/normalized.
    with pytest.raises(ValueError, match="unexpected"):
        parse_meta(_one({"name": "x", "value": 1.0, "level": "episode", "step": 1}))
    with pytest.raises(ValueError, match="unexpected"):
        parse_meta(_one({"name": "x", "value": 1.0, "level": "episode", "notes": "hi"}))
    with pytest.raises(ValueError, match="missing"):
        parse_meta(_one({"value": 1.0, "level": "inference", "step": 1}))  # no name


def test_visibility_rule_default_and_optin() -> None:
    inf = InferenceFeedback(name="dense", value=0.2, step=1)
    ep = EpisodeFeedback(name="reward", value=1.0)
    items = [inf, ep]
    # Default (eval-safe): dense/inference is recorded-only; episode hidden until terminal.
    assert select_inband(items, terminal=False) == []
    assert select_inband(items, terminal=True) == [ep]
    # Explicit per-tool opt-in surfaces inference; episode still only at terminal.
    assert select_inband(items, terminal=False, surface_inference=True) == [inf]
    assert select_inband(items, terminal=True, surface_inference=True) == [inf, ep]


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(7, id="an ordinary integer"),
        pytest.param(9007199254740993, id="an integer past float precision"),
        pytest.param(10**400, id="an integer past float range"),
    ],
)
def test_an_integer_value_survives_the_wire_as_the_integer_it_was(value: int) -> None:
    """The wire admits any JSON scalar, and the models it was rebuilt through admitted fewer.

    ``value`` is declared ``float | bool | str`` on the models, so building an item through the
    *constructor* handed the annotation the last word over a contract that says ``int`` is a
    scalar like any other. A value published as ``9007199254740993`` came back as
    ``9007199254740992.0``, and one past the float range came back as nothing at all: the item was
    refused, on both boundaries, because the serializer validates through the same rebuild.

    So the rebuild carries what the checks passed. Value **and type**, because equality alone
    cannot see this: ``9007199254740992.0 == 9007199254740992`` is true, and the two numbers a
    reward could be here differ in the last place."""
    item = EpisodeFeedback(name="reward", value=1.0)
    item.value = value  # assignment validation is off, so the integer reaches the wire as one
    step_item = InferenceFeedback(name="reward", value=1.0, step=1)
    step_item.value = value

    meta = build_meta([step_item, item])
    (dumped_step, dumped) = meta[FEEDBACK_META_KEY]
    assert dumped["value"] == value and type(dumped["value"]) is int
    assert dumped_step["value"] == value and type(dumped_step["value"]) is int

    parsed, _ = parse_meta(meta)
    assert [p.value for p in parsed] == [value, value]
    assert [type(p.value) for p in parsed] == [int, int]
    # And the round trip is closed: what a consumer rebuilds serializes to what it was given.
    assert build_meta(parsed)[FEEDBACK_META_KEY] == meta[FEEDBACK_META_KEY]


def test_a_feedback_name_that_is_not_text_is_still_refused() -> None:
    """The check the models used to make, now made here.

    Rebuilding with ``model_construct`` skips the annotation, which is the point: the annotation
    was narrower than the wire for ``value``. It was exactly right for ``name``, which is the key
    a record is composed and headlined by, so that check is written down rather than inherited."""
    item = EpisodeFeedback(name="reward", value=1.0)
    item.name = 7  # type: ignore[assignment]  # off-wire, like every other mutation here
    with pytest.raises(ValueError, match="name must be text"):
        build_meta([item])
    with pytest.raises(ValueError, match="name must be text"):
        parse_meta({FEEDBACK_META_KEY: [{"name": 7, "value": 1.0, "level": "episode"}]})
    with pytest.raises(ValueError, match="name must be text"):
        parse_meta(
            {FEEDBACK_META_KEY: [{"name": None, "value": 1.0, "level": "inference", "step": 1}]}
        )
