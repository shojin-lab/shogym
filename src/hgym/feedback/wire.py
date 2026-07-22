"""Serialize feedback onto (and off) the MCP tool-result ``_meta`` sidecar (RFC 008 §4).

Two reserved keys, one namespace:

- ``hgym/feedback`` — a list of feedback items, each ``{name, value, level[, step]}``.
  ``level`` is ``"inference"`` (per-step, carries ``step``) or ``"episode"`` (terminal),
  mirroring :class:`InferenceFeedback` / :class:`EpisodeFeedback`.
- ``hgym/terminate`` — a boolean stop-hint. It is *control*, not a score, so it rides a
  separate key: a tool can end an episode (horizon, error) without smuggling a reward,
  and a reward can attach to any step without implying the episode ends.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from hgym.types import EpisodeFeedback, InferenceFeedback

FEEDBACK_META_KEY = "hgym/feedback"
TERMINATE_META_KEY = "hgym/terminate"

FeedbackItem = Union[InferenceFeedback, EpisodeFeedback]


def dump_item(item: FeedbackItem) -> Dict[str, Any]:
    """Serialize one feedback item to its wire dict."""
    data: Dict[str, Any] = {"name": item.name, "value": item.value}
    if isinstance(item, InferenceFeedback):
        data["level"] = "inference"
        data["step"] = item.step
    else:
        data["level"] = "episode"
    return data


def _load_item(raw: Mapping[str, Any]) -> FeedbackItem:
    if raw.get("level") == "inference":
        return InferenceFeedback(name=raw["name"], value=raw["value"], step=raw["step"])
    return EpisodeFeedback(name=raw["name"], value=raw["value"])


def build_meta(
    items: Sequence[FeedbackItem] = (), *, terminate: bool = False
) -> Dict[str, Any]:
    """Build the ``_meta`` sidecar for a tool result. Empty items + ``terminate=False``
    yields ``{}`` (nothing to attach)."""
    meta: Dict[str, Any] = {}
    if items:
        meta[FEEDBACK_META_KEY] = [dump_item(i) for i in items]
    if terminate:
        meta[TERMINATE_META_KEY] = True
    return meta


def parse_meta(meta: Mapping[str, Any]) -> Tuple[List[FeedbackItem], bool]:
    """Inverse of :func:`build_meta`: pull feedback items and the terminate flag out of
    a tool result's ``_meta`` (returns ``([], False)`` when neither key is present)."""
    raw_items = meta.get(FEEDBACK_META_KEY) or []
    items = [_load_item(raw) for raw in raw_items]
    return items, bool(meta.get(TERMINATE_META_KEY, False))


def select_inband(items: Sequence[FeedbackItem], *, terminal: bool) -> List[FeedbackItem]:
    """Apply the visibility rule (RFC 008 §4.4): **episode-level feedback is invisible to
    the harness until the terminal result.** Surfacing terminal reward mid-episode would
    leak it into the policy and contaminate the comparison. Inference-level items pass
    through — whether they are surfaced at all is the server's per-tool opt-in (it simply
    does not pass them to this function when it wants them recorded-only)."""
    if terminal:
        return list(items)
    return [i for i in items if not isinstance(i, EpisodeFeedback)]
