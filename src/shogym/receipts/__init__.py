"""Gates and screens for graded receipts: can this task source carry a measurement.

The library is separate from the `receipts_v1` environment on purpose. The question
the gates ask, whether a graded receipt can tell an agent anything about a hidden
convention it could not have looked up, is a question about ANY task source, not
only about the generators this repo ships. A rented world, a scenario pair someone
else authored, or a receipt from a benchmark shogym does not own can all be handed
to `gate`, as long as their receipts can be rendered under every convention.

`resolution` holds gates R (the receipt resolves an axis past the pin), S (the
receipt does not print its own interpretation) and H (the ceiling stands above the
lookup floor, both optimized over the sibling task's legal action space). `screen`
holds the room screen, which is empirical: what one graded receipt was actually
worth against a placebo and an oracle.
"""

from shogym.receipts.resolution import (
    AXIS_LABEL,
    ROW_LABEL,
    AxisSpace,
    GateResult,
    Observation,
    axis_receipts,
    bits,
    evident_rows,
    gate,
    resolution_blocks,
    row_dependence,
    rowwise_scores,
    score_partition,
)
from shogym.receipts.screen import (
    REGISTERED_MIN_PAIRS,
    REGISTERED_MIN_RATIO,
    REGISTERED_MIN_ROOM,
    Outcomes,
    PairRecord,
    ScreenRecord,
    ScreenResult,
    ScreenRun,
    contrasts,
    floored_ratio,
    read_payload,
    screen,
    sd_influence,
)

__all__ = [
    "AXIS_LABEL",
    "ROW_LABEL",
    "AxisSpace",
    "GateResult",
    "REGISTERED_MIN_PAIRS",
    "REGISTERED_MIN_RATIO",
    "REGISTERED_MIN_ROOM",
    "Observation",
    "Outcomes",
    "PairRecord",
    "ScreenRecord",
    "ScreenResult",
    "ScreenRun",
    "axis_receipts",
    "bits",
    "contrasts",
    "evident_rows",
    "floored_ratio",
    "gate",
    "read_payload",
    "resolution_blocks",
    "row_dependence",
    "rowwise_scores",
    "score_partition",
    "screen",
    "sd_influence",
]
