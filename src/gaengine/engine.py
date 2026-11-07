"""Genetic Algorithm engine: the main evolution loop."""

from __future__ import annotations

import numpy as np

from .chromosome import Chromosome
from .operators import (
    adaptive_hybrid_mutation,
    tournament_select,
    two_point_crossover,
)
from .population import create_initial_population, create_rebalance_population
from .profile_policy import prepare_policy, ProfileConstraintError
from .constrained_operators import (
    check_affordability, repair_sparse, sparse_set_crossover,
    sparse_portfolio_mutation, PortfolioFeasibilityError,
)


def _fit(c):
    return c.fitness if c.fitness is not None else -1e300


class GeneticEngine:
    """Runs the population through selection, crossover, mutation and
    elitism until the generation budget or a stagnation criterion is met."""

    def __init__(self, config):
        self.config = config
        self.history = []
        self.solution = None
        self.diagnostics = []

    def run(self, verbose: bool = True, log_every: int = 25) -> Chromosome:
        cfg = self.config
        prepare_policy(cfg)
        rng = np.random.default_rng(cfg.seed)
        sparse = cfg.operator_policy == "sparse_v1"
        if sparse:
            try:
                check_affordability(cfg)
            except PortfolioFeasibilityError as exc:
                if cfg.profile_policy != "legacy":
                    raise ProfileConstraintError(str(exc)) from exc
                raise
        crossover = sparse_set_crossover if sparse else two_point_crossover
        mutation = sparse_portfolio_mutation if sparse else adaptive_hybrid_mutation
        if verbose:
            print(f"[GA] operators={cfg.operator_policy}")

        if getattr(cfg, 'is_rebalance', False) and getattr(cfg, 'old_weights', None) is not None:
            population = create_rebalance_population(cfg, rng)
        else:
            population = create_initial_population(cfg, rng)
        for chrom in population:
            if sparse:
                repair_sparse(chrom, preserve_valid=True)
            chrom.evaluate_fitness()

        prior_best = None
        stagnant = 0

        for generation in range(cfg.generations):
            population.sort(key=_fit, reverse=True)
            best = population[0]
            self.history.append(best.fitness)
            feasible = [c for c in population if c.lots is not None and c.budget_ok and c.diversification_ok
                        and c.profile_constraints_ok]
            self.diagnostics.append(dict(
                generation=generation, operator_policy=cfg.operator_policy, best=float(best.fitness),
                feasible_count=len(feasible), population_count=len(population),
                feasible_mean=float(np.mean([c.fitness for c in feasible])) if feasible else None,
                unique_feasible_lots=len({tuple(c.lots) for c in feasible}),
                budget_failures=sum(not c.budget_ok for c in population),
                cardinality_failures=sum(not c.diversification_ok for c in population),
                profile_failures=sum(not c.profile_constraints_ok for c in population),
            ))

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
                print(f"[GA] gen {generation:>4d} | best={best.fitness:12.8f} "
                      f"avg={avg:10.4f} | feasible={len(feasible)}/{len(population)} "
                      f"unique_feasible={self.diagnostics[-1]['unique_feasible_lots']} "
                      f"n_active={best.n_active}")

            # ---- elitism + reproduction -------------------------------
            elite_count = min(cfg.elitism_count, cfg.population_size-1) if sparse else cfg.elitism_count
            elites = population[:elite_count]
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
                    child_a, child_b = crossover(parent_a, parent_b, rng)
                else:
                    child_a, child_b = parent_a.clone(), parent_b.clone()
                    child_a.fitness = None
                    child_b.fitness = None

                for child in (child_a, child_b):
                    mutation(child, generation, cfg.generations, rng)
                    child.evaluate_fitness()
                    if len(children) < needed:
                        children.append(child)
                    else:
                        break

            population = elites + children

        population.sort(key=_fit, reverse=True)
        self.solution = population[0]
        if sparse and not (self.solution.budget_ok and self.solution.diversification_ok):
            raise PortfolioFeasibilityError('GA tidak menemukan portofolio sesuai modal dan jumlah saham.')
        if cfg.profile_policy != "legacy":
            self.solution.evaluate_fitness()
            if not (self.solution.profile_constraints_ok and self.solution.budget_ok and self.solution.diversification_ok):
                violations = (self.solution.profile_metrics or {}).get('violations', [])
                raise ProfileConstraintError(
                    "GA belum menemukan portofolio yang memenuhi composition_v1; "
                    "bukan bukti bahwa solusi mustahil. Pelanggaran kandidat terbaik: " + ', '.join(violations))
        if verbose:
            print(f"[GA] Final best fitness = {self.solution.fitness:.4f}")
        return self.solution
