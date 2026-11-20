"""Tournament selection shared by the sparse genetic operators."""

from __future__ import annotations

import numpy as np


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


def tournament_select(population, k: int = 5, rng=None):
    """Pick ``k`` random individuals and return the one with the best fitness."""
    rng = _rng(rng)
    picks = rng.integers(0, len(population), size=k)
    best = None
    for p in picks:
        c = population[int(p)]
        if c.fitness is not None and (best is None or c.fitness > best.fitness):
            best = c
    return best
