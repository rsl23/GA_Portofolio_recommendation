"""Genetic Algorithm engine: the main evolution loop."""

from __future__ import annotations

import numpy as np

from .chromosome import Chromosome
from .operators import (
    adaptive_hybrid_mutation,
    tournament_select,
    two_point_crossover,
)
from .population import create_initial_population


def _fit(c):
    return c.fitness if c.fitness is not None else -1e300


class GeneticEngine:
    """Runs the population through selection, crossover, mutation and
    elitism until the generation budget or a stagnation criterion is met."""

    def __init__(self, config):
        self.config = config
        self.history = []
        self.solution = None

    def run(self, verbose: bool = True, log_every: int = 25) -> Chromosome:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        population = create_initial_population(cfg, rng)
        for chrom in population:
            chrom.evaluate_fitness()

        prior_best = None
        stagnant = 0

        for generation in range(cfg.generations):
            population.sort(key=_fit, reverse=True)
            best = population[0]
            self.history.append(best.fitness)

            # ---- stopping criterion: no significant improvement for N gens --
            if prior_best is None:
                prior_best = best.fitness
            elif best.fitness - prior_best >= cfg.improvement_tol:
                stagnant = 0
            else:
                stagnant += 1
            prior_best = max(prior_best, best.fitness)

            if stagnant >= cfg.early_stop_patience:
                if verbose:
                    print(f"[GA] Early stop at generation {generation} "
                          f"(best={best.fitness:.4f}, stagnant={stagnant})")
                break

            if verbose and generation % log_every == 0:
                avg = float(np.mean([c.fitness for c in population]))
                print(f"[GA] gen {generation:>4d} | best={best.fitness:12.4f} "
                      f"avg={avg:10.4f} | n_active={best.n_active}")

            # ---- elitism + reproduction -------------------------------
            elites = population[:cfg.elitism_count]
            needed = cfg.population_size - len(elites)
            children = []

            while len(children) < needed:
                parent_a = tournament_select(population, cfg.tournament_size, rng)
                parent_b = parent_a
                guard = 0
                while parent_b is parent_a and guard < 20:
                    parent_b = tournament_select(population, cfg.tournament_size, rng)
                    guard += 1

                if rng.random() < cfg.crossover_rate:
                    child_a, child_b = two_point_crossover(parent_a, parent_b, rng)
                else:
                    child_a, child_b = parent_a.clone(), parent_b.clone()
                    child_a.fitness = None
                    child_b.fitness = None

                for child in (child_a, child_b):
                    adaptive_hybrid_mutation(child, generation, cfg.generations, rng)
                    child.evaluate_fitness()
                    if len(children) < needed:
                        children.append(child)
                    else:
                        break

            population = elites + children

        population.sort(key=_fit, reverse=True)
        self.solution = population[0]
        if verbose:
            print(f"[GA] Final best fitness = {self.solution.fitness:.4f}")
        return self.solution
