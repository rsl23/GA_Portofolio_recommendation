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


def create_rebalance_population(config, rng=None):
    """
    Hybrid Initialization for Rebalancing:
    - 15% of the population is seeded based on the old portfolio weights.
    - 85% is randomly initialized (to ensure diversity).
    """
    rng = rng or np.random.default_rng(config.seed)
    population = []
    
    n_seeded = int(config.population_size * 0.15)
    n_seeded = max(1, n_seeded)  # Ensure at least 1 seeded chromosome
    
    old_weights = config.old_weights
    
    # Generate seeded chromosomes
    for i in range(n_seeded):
        chrom = Chromosome(config)
        if i == 0:
            # First chromosome is an exact copy of the old weights
            chrom.genes = list(old_weights)
        else:
            # Others are lightly mutated copies (creep mutation)
            mutated_genes = np.array(old_weights)
            # Add small random noise
            noise = rng.normal(0.0, 0.02, size=len(old_weights))
            # Apply noise with 30% probability per gene
            mask = rng.random(size=len(old_weights)) < 0.3
            mutated_genes[mask] += noise[mask]
            
            # Clip and normalize
            mutated_genes = np.clip(mutated_genes, 0.0, 1.0)
            s = mutated_genes.sum()
            if s > 1e-12:
                mutated_genes = mutated_genes / s
            else:
                mutated_genes = np.zeros_like(mutated_genes)
                
            chrom.genes = mutated_genes.tolist()
            
        chrom.normalize()
        if config.allow_cash and config.fitness_mode != "baseline":
            if i == 0:
                chrom.exposure = float(np.clip(
                    sum(old_weights), config.min_exposure, config.max_exposure
                ))
            else:
                chrom.exposure = (config.min_exposure
                                  if config.min_exposure == config.max_exposure
                                  else float(rng.uniform(config.min_exposure, config.max_exposure)))
        population.append(chrom)
        
    # Generate the rest randomly
    for _ in range(config.population_size - n_seeded):
        chrom = Chromosome(config)
        chrom.random_initialization(rng)
        population.append(chrom)
        
    return population
