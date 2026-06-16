"""Per-surface and combined harness hashing (RFC 007, the observability surface).

Every trace is tagged with the hash of the harness that produced it, so a result can
be attributed to an exact surface configuration. The hash is *structured*: one
sub-hash per optimizable surface, plus a combined ``harness_hash`` that is a pure
function of those sub-hashes. This is what makes attribution per-surface — flipping
one surface moves exactly one sub-hash (and the combined hash), leaving the rest
byte-identical, so a sweep can group results by "the surface I changed".

Hashes are content hashes over a canonical JSON encoding (sorted keys, no
whitespace), so they are deterministic across processes and independent of dict
insertion order. They are truncated to 16 hex chars — collision-safe for the scale of
a sweep, short enough to read in a dataframe.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from hgym.harness._format import Harness

_HASH_LEN = 16

# The surfaces this hash covers. Context / execution / verification are env-or-runner
# fixed (not harness attributes), so they are not part of the harness hash.
SURFACES = ("inference", "instruction", "tool", "control")


def _sha(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_HASH_LEN]


def surface_hashes(harness: Harness) -> Dict[str, str]:
    """One content hash per optimizable surface of ``harness``.

    Keys are :data:`SURFACES`. Two harnesses agree on a surface's hash iff that
    surface's content is identical (param order does not matter).
    """
    return {
        "inference": _sha({"model": harness.model, "params": harness.inference_params}),
        "instruction": _sha(harness.system_template or ""),
        "tool": _sha([spec.model_dump() for spec in harness.extra_specs]),
        "control": _sha({"horizon": harness.horizon}),
    }


def harness_hash(harness: Harness) -> str:
    """The combined harness hash: a pure function of the per-surface sub-hashes.

    Equal iff every surface hash is equal, so it is the single id for "this exact
    harness configuration".
    """
    return _sha(surface_hashes(harness))
