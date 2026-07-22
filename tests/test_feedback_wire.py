"""The feedback-on-the-wire contract (RFC 008 §4): _meta round-trip, the terminate flag
as a separate concern from the score, and the mid-episode visibility rule."""

from __future__ import annotations

from hgym.feedback import (
    FEEDBACK_META_KEY,
    TERMINATE_META_KEY,
    build_meta,
    parse_meta,
    select_inband,
)
from hgym.types import EpisodeFeedback, InferenceFeedback


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
    assert parsed == items  # exact reconstruction, including the step on the inference item


def test_terminate_flag_is_independent_of_feedback() -> None:
    # A stop with no score...
    stop_only, term = parse_meta(build_meta(terminate=True))
    assert stop_only == [] and term is True
    # ...and a score with no stop.
    scored, term = parse_meta(build_meta([InferenceFeedback(name="r", value=0.5, step=1)]))
    assert term is False and len(scored) == 1


def test_parse_empty_meta() -> None:
    assert parse_meta({}) == ([], False)


def test_visibility_hides_episode_until_terminal() -> None:
    items = [
        InferenceFeedback(name="dense", value=0.2, step=1),
        EpisodeFeedback(name="reward", value=1.0),
    ]
    # Mid-episode: the terminal reward must not be surfaced.
    mid = select_inband(items, terminal=False)
    assert mid == [items[0]]
    # On the terminal result: everything surfaces.
    assert select_inband(items, terminal=True) == items
