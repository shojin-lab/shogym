"""The harness: the editable, optimizable projection of a rollout-and-agent config.

See ``docs`` / the RFC stack for the design. Public surface:

- :class:`Harness` — a loaded harness (inference, instruction, tool-extras, limits).
- :func:`export_harness` — write a baseline harness directory from an env.
- :func:`load_harness` — read a harness directory back into a :class:`Harness`.
"""

from hgym.harness._format import (
    ExtraIsolationError,
    Harness,
    export_harness,
    load_harness,
)
from hgym.harness._hash import SURFACES, harness_hash, surface_hashes

__all__ = [
    "SURFACES",
    "ExtraIsolationError",
    "Harness",
    "export_harness",
    "harness_hash",
    "load_harness",
    "surface_hashes",
]
