"""Serialize feedback onto (and off) the MCP tool-result ``_meta`` sidecar (RFC 008 §4).

Two reserved keys, one namespace:

- ``shogym/feedback`` — a list of feedback items, each ``{name, value, level[, step]}``.
  ``level`` is ``"inference"`` (per-step, carries ``step``) or ``"episode"`` (terminal),
  mirroring :class:`InferenceFeedback` / :class:`EpisodeFeedback`.
- ``shogym/terminate`` — a boolean stop-hint. It is *control*, not a score, so it rides a
  separate key: a tool can end an episode (horizon, error) without smuggling a reward,
  and a reward can attach to any step without implying the episode ends.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, List, Sequence, Tuple, Union

from shogym.types import EpisodeFeedback, InferenceFeedback

# The MCP `_meta` sidecar keys. Renamed with the package: both the writer (`build_meta`) and the
# reader (`parse_meta`) live here, the repo was private for the whole life of the published 0.0.1
# release, and that wheel predates this module entirely, so no third-party producer of these keys
# exists. They are transient wire fields rather than a storage format -- `TraceRecord` persists the
# parsed result (`feedback`, `terminated`), never the namespaced key -- so existing trace files
# need no translation.
FEEDBACK_META_KEY = "shogym/feedback"
TERMINATE_META_KEY = "shogym/terminate"

# The two episode-feedback names a *paired* pair of feedback policies is written against: one
# policy reveals the item published under `REPORT_FEEDBACK_NAME`, its twin the item published
# under `NOTICE_FEEDBACK_NAME`, and neither reveals anything else. They live here, beside the wire
# keys, because they are the one part of the feedback contract an env and a policy must agree on
# by name: an env publishes under them, a policy selects by them, and neither may import the
# other.
#
# `report` carries whatever the env decided the ending is worth saying; `notice` carries the
# env's own inert, length-matched stand-in for it. Both names are six characters, so two arms
# whose values are the same length *as the answer encodes them* answer a terminating call with
# byte-identical payloads apart from the value itself: the channel a run served under is a
# property of its record rather than something readable off the shape of what the agent was
# handed. Matching the values is the env's job, not this contract's.
REPORT_FEEDBACK_NAME = "report"
NOTICE_FEEDBACK_NAME = "notice"

FeedbackItem = Union[InferenceFeedback, EpisodeFeedback]


def _require_scalar_value(value: Any) -> None:
    """Enforce the JSON-scalar wire contract for a feedback value on *both* boundaries.

    The value must be an actual JSON scalar — ``bool``, ``int``, ``float``, or ``str`` —
    and any float must be finite. This is stricter than the Pydantic models on purpose:
    they would *coerce* a ``Decimal``/``Fraction`` into a float (``Decimal("NaN")`` even
    slips past a plain ``float`` finiteness test), which then either breaks ``json.dumps``
    on the sidecar or is read back as a floating ``nan``. Checking the raw type at
    serialize (``dump_item`` routes through ``_load_item``) and parse (``_load_item``)
    keeps the contract symmetric: a non-scalar or non-finite value can be neither written
    nor read.
    """
    if not isinstance(value, (bool, int, float, str)):
        raise ValueError(
            f"feedback value must be a JSON scalar (number, bool, or text), got {value!r}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"feedback value must be finite, got {value!r}")


def dump_item(item: FeedbackItem) -> Dict[str, Any]:
    """Serialize one feedback item to its wire dict.

    Both the live MCP ``_meta`` sidecar (``build_meta``) and the JSONL trace
    (``step_record``) go through here. The feedback models are mutable and do not
    validate on assignment, so a post-construction mutation (``item.step = True``, a
    non-scalar ``value``, a non-string ``name``) would otherwise emit a malformed
    sidecar that ``parse_meta`` then rejects — an asymmetric round-trip on the live
    wire. Validate the serialized form against the same schema ``_load_item``
    enforces, so a write path can only ever emit a round-trippable item.
    """
    data: Dict[str, Any] = {"name": item.name, "value": item.value}
    if isinstance(item, InferenceFeedback):
        data["level"] = "inference"
        data["step"] = item.step
    else:
        data["level"] = "episode"
    _load_item(data)
    return data


# The exact wire key set per level. dump_item emits exactly these; parse requires
# them so an external producer's typo or wrong-level field is caught, not silently
# coerced/dropped by the (lenient-by-default) Pydantic models.
_INFERENCE_KEYS = frozenset({"name", "value", "level", "step"})
_EPISODE_KEYS = frozenset({"name", "value", "level"})


def _require_exact_keys(raw: Mapping[str, Any], allowed: frozenset, level: str) -> None:
    keys = set(raw)
    if keys == allowed:
        return
    problems = []
    if missing := allowed - keys:
        problems.append(f"missing {sorted(missing)}")
    if extra := keys - allowed:
        problems.append(f"unexpected {sorted(extra)}")
    raise ValueError(f"{level} feedback item: {'; '.join(problems)}")


def _load_item(raw: Mapping[str, Any]) -> FeedbackItem:
    if not isinstance(raw, Mapping):
        raise ValueError(f"feedback item must be a mapping, got {raw!r}")
    level = raw.get("level")
    if level == "inference":
        _require_exact_keys(raw, _INFERENCE_KEYS, "inference")
        _require_scalar_value(raw["value"])  # JSON scalar only; json.loads/Pydantic are laxer
        step = raw["step"]
        # bool is an int subclass; the contract says step is a plain integer, so
        # reject True/False (which Pydantic would otherwise coerce to 1/0) and
        # string steps (coerced to int) — both are malformed producer messages.
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"inference feedback 'step' must be an int, got {step!r}")
        return InferenceFeedback(name=raw["name"], value=raw["value"], step=step)
    if level == "episode":
        _require_exact_keys(raw, _EPISODE_KEYS, "episode")
        _require_scalar_value(raw["value"])
        return EpisodeFeedback(name=raw["name"], value=raw["value"])
    # Reject rather than defaulting: silently treating an unknown level (a typo
    # like "inferencee") as episode feedback would flip its mid-episode visibility.
    raise ValueError(f"unknown feedback level {level!r}; expected 'inference' or 'episode'")


def build_meta(
    items: Sequence[FeedbackItem] = (), *, terminate: bool = False
) -> Dict[str, Any]:
    """Build the ``_meta`` sidecar for a tool result. Empty items + ``terminate=False``
    yields ``{}`` (nothing to attach)."""
    # Validate rather than test truthiness: `if terminate:` would turn a
    # string-valued false flag (`"false"`) into a real stop signal — the same
    # premature-termination the strict parser rejects, but from the serializer.
    if not isinstance(terminate, bool):
        raise ValueError(f"terminate must be a boolean, got {terminate!r}")
    meta: Dict[str, Any] = {}
    if items:
        meta[FEEDBACK_META_KEY] = [dump_item(i) for i in items]
    if terminate:
        meta[TERMINATE_META_KEY] = True
    return meta


def parse_meta(meta: Mapping[str, Any]) -> Tuple[List[FeedbackItem], bool]:
    """Inverse of :func:`build_meta`: pull feedback items and the terminate flag out of a
    tool result's ``_meta`` (returns ``([], False)`` when neither key is present)."""
    # `.get(key, [])` (not `... or []`) so an *absent* key defaults to empty but a
    # *present* malformed one (null / false / 0 / a bare item object) is caught,
    # not silently swallowed. The contract defines this key as a list.
    raw_items = meta.get(FEEDBACK_META_KEY, [])
    if not isinstance(raw_items, list):
        raise ValueError(f"{FEEDBACK_META_KEY!r} must be a list, got {raw_items!r}")
    items = [_load_item(raw) for raw in raw_items]
    terminate = meta.get(TERMINATE_META_KEY, False)
    # The contract defines this key as a boolean. Validate the type rather than
    # coercing: `bool("false")` is True, so a string-valued flag would silently
    # terminate an episode. Only a real boolean controls termination.
    if not isinstance(terminate, bool):
        raise ValueError(
            f"{TERMINATE_META_KEY!r} must be a boolean, got {terminate!r}"
        )
    return items, terminate


def select_inband(
    items: Sequence[FeedbackItem], *, terminal: bool, surface_inference: bool = False
) -> List[FeedbackItem]:
    """Apply the visibility rule (RFC 008 §4.4), which protects eval integrity by not
    leaking reward signals into the policy unless an experiment opts in:

    - **Episode-level feedback is hidden until the terminal result.** Surfacing terminal
      reward mid-episode would leak it into the policy and contaminate the comparison.
    - **Inference-level (dense/step) feedback is recorded-but-not-surfaced by default.**
      It is always written to the trace; it reaches the harness in-band only when a tool
      explicitly opts in with ``surface_inference=True``. Silent dense-reward shaping
      otherwise invalidates cross-harness comparisons.
    """
    out: List[FeedbackItem] = []
    for item in items:
        if isinstance(item, EpisodeFeedback):
            if terminal:
                out.append(item)
        elif surface_inference:  # InferenceFeedback — opt-in only
            out.append(item)
    return out
