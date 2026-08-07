"""Deterministic, disjoint train / held-out partition of the AutomationBench public tasks.

AutomationBench does not distribute its real private set, so we build a **private-set proxy**:
split the 600 public task indices into a train pool and a held-out pool that are disjoint by
construction and reproducible from a seed. The held-out pool is NEVER placed in the train
stream (the broker's queue is built from the train pool only), so the honest generalization
curve is measured on tasks the treatment arm never trained on.

This module is pure (a function of ``n``, ``seed``, ``heldout_frac``) so it is unit-testable
without loading the dataset, and so the broker and the held-out evaluator compute the *same*
partition independently.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from . import config


@dataclass(frozen=True)
class Split:
    """A reproducible partition of ``range(n)`` into disjoint ``train`` / ``heldout`` index
    lists (both sorted ascending). ``seed`` and ``heldout_frac`` fully determine it."""

    n: int
    seed: int
    heldout_frac: float
    train: Tuple[int, ...]
    heldout: Tuple[int, ...]

    def __post_init__(self) -> None:
        # Integrity, checked at construction: a true partition, no overlap, nothing dropped.
        t, h = set(self.train), set(self.heldout)
        assert t.isdisjoint(h), "train and held-out overlap — held-out would leak into training"
        assert t | h == set(range(self.n)), "split does not cover every task index exactly once"
        assert len(self.train) == len(t) and len(self.heldout) == len(h), "split has repeats"


def make_split(
    n: int,
    seed: int = config.SPLIT_SEED,
    heldout_frac: float = config.HELDOUT_FRAC,
) -> Split:
    """Partition ``range(n)`` into disjoint train / held-out pools.

    A fixed-seed shuffle fixes a canonical order; the first ``round(n * heldout_frac)`` of it
    are held out, the rest are train. Deterministic, disjoint, total."""
    if not 0.0 <= heldout_frac < 1.0:
        raise ValueError(f"heldout_frac must be in [0, 1), got {heldout_frac}")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    k = round(n * heldout_frac)
    heldout = tuple(sorted(order[:k]))
    train = tuple(sorted(order[k:]))
    return Split(n=n, seed=seed, heldout_frac=heldout_frac, train=train, heldout=heldout)


def train_stream(
    split: Split,
    size: int,
    seed: int = config.SPLIT_SEED,
) -> List[int]:
    """A non-repeating stream of ``size`` train indices (a fresh seeded shuffle of the train
    pool, capped). Never contains a held-out index, because it draws from ``split.train`` only.
    ``size`` larger than the pool is clamped to the whole pool (still no repeats)."""
    order = list(split.train)
    random.Random(seed + 1).shuffle(order)  # +1: distinct from the partition shuffle
    return order[: min(size, len(order))]
