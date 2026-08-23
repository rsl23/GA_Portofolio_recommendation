"""Population creation helpers."""

from __future__ import annotations

import numpy as np

from .chromosome import Chromosome


def create_initial_population(config, rng=None):
    """Create ``config.population_size`` randomly initialised chromosomes."""
    rng = rng or np.random.default_rng(config.seed)
    population = []
    for _ in range(config.population_size):
        chrom = Chromosome(config)
        chrom.random_initialization(rng)
        population.append(chrom)
    return population
