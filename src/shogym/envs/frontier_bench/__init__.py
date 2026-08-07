"""shogym port of Frontier-Bench (Harbor / Terminal-Bench successor) — Docker-backed.

Public surface is the registered env (``shogym.make("frontier_bench")``); this package also
exposes the pure verdict scorer and the pinned manifest for offline tests. See
``README.md`` for the full describe / serve / verify contract and the fidelity pins.
"""

from shogym.envs.frontier_bench import manifest

__all__ = ["manifest"]
