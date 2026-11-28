"""Opt-in research GA with profile-specific risk ceilings.

This module is selected explicitly by the API; the legacy GA remains intact.
Call ``run(config)`` with an existing GAConfig whose MarketData has
``profile_features`` attached (build_market_data(..., include_profile_features=True)).
The shared stock screen, existing fitness, and sparse operators are reused.
All limits and reward coefficients below are uncalibrated research hypotheses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .chromosome import Chromosome
from .constrained_operators import (
    check_affordability, repair_sparse, sparse_portfolio_mutation,
    sparse_set_crossover,
)
from .operators import tournament_select
from .population import create_initial_population, create_rebalance_population
from .profile_policy import ProfileConstraintError, ProfilePolicy, _project
from .progress import report_candidates, report_generation
from .fitness_context import fitness_inputs_scope
from .ga_config import RiskProfile


@dataclass(frozen=True)
class Limits:
    min_vol_ratio: float
    max_vol_ratio: float
    min_core_weight: float
    max_stock_weight: float
    max_sector_weight: float
    upside_coefficient: float


POLICY_VERSION = "composition-v2-risk-caps-v3"
ADAPTIVE_POLICY_VERSION = "composition-v2-conservative-adaptive-v1"

LIMITS = {
    "Conservative": Limits(0.0, 1.10, 0.60, 0.20, 0.35, 0.0),
    "Moderate": Limits(0.0, 1.25, 0.30, 0.25, 0.45, 0.10),
    "Aggressive": Limits(0.0, 1.50, 0.00, 0.35, 0.60, 0.25),
}
ADAPTIVE_CONSERVATIVE_LIMITS = Limits(0.0, 1.15, 0.40, 0.22, 0.40, 0.10)
ADAPTIVE_VOLATILITY_PENALTY = 0.30
ADAPTIVE_DRAWDOWN_SESSIONS = 126
ADAPTIVE_DRAWDOWN_LIMIT = 0.12


def uses_adaptive_conservative(config):
    return (config.composition_v2_adaptive and config.is_rebalance
            and config.profile is RiskProfile.CONSERVATIVE
            and config.fitness_policy == "legacy_cvar_v1")


def configure_conservative_rebalance(config, market_status):
    """Freeze the signal-date fitness choice for optimization and execution."""
    if (not config.is_rebalance or config.profile is not RiskProfile.CONSERVATIVE
            or config.fitness_policy != "legacy_cvar_v1" or market_status != "normal"):
        return config
    history = np.asarray(config.data.profile_features.benchmark_returns, dtype=float)
    history = history[-ADAPTIVE_DRAWDOWN_SESSIONS:]
    if (history.size < ADAPTIVE_DRAWDOWN_SESSIONS
            or not np.isfinite(history).all() or np.any(history <= -1.0)):
        return config
    wealth = np.cumprod(1.0 + history)
    drawdown = 1.0 - float(wealth[-1]) / max(1.0, float(wealth.max()))
    if drawdown >= ADAPTIVE_DRAWDOWN_LIMIT:
        return config
    return replace(config, composition_v2_adaptive=True, dividend_bonus_override=0.45,
                   mdd_penalty_override=1.30, lambda_cvar=0.65 / config.profile.lambda_mdd)


def _prepare(config):
    if config.profile_policy != "legacy" or config.fitness_mode != "rebalance_aware":
        raise ValueError("Experiment requires profile_policy='legacy' and rebalance_aware fitness.")
    features = getattr(config.data, "profile_features", None)
    if features is None:
        raise ProfileConstraintError("Attach as-of profile_features before running composition_v2.")
    if not hasattr(features, "covariance"):
        features.prepare(config.data)
    limits = (ADAPTIVE_CONSERVATIVE_LIMITS if uses_adaptive_conservative(config)
              else LIMITS[config.profile.value])
    if config.max_stocks * limits.max_stock_weight < 1 - 1e-9:
        raise ProfileConstraintError("max_stocks is too small for the stock-weight cap.")
    if int(features.core_mask.sum()) * limits.max_stock_weight < limits.min_core_weight - 1e-9:
        raise ProfileConstraintError("Too few eligible core stocks for this profile.")
    if len(features.groups) * limits.max_sector_weight < 1 - 1e-9:
        raise ProfileConstraintError("Too few eligible sectors for this profile.")
    return features, limits


def metrics(config, lots, *, prices_per_lot=None):
    """Measure the invested basket at signal or execution prices."""
    features, limits = _prepare(config)
    prices = np.asarray(config.data.prices_per_lot if prices_per_lot is None
                        else prices_per_lot, dtype=float)
    lots = np.asarray(lots, dtype=float)
    if (lots.shape != prices.shape or not np.isfinite(lots).all()
            or not np.isfinite(prices).all() or np.any(lots < 0)
            or np.any(prices <= 0)):
        raise ProfileConstraintError("Invalid lots or prices for profile evaluation.")
    allocation = lots * prices
    if allocation.sum() <= 0:
        raise ProfileConstraintError("A profile requires a nonempty invested basket.")
    weights = allocation / allocation.sum()
    volatility = math.sqrt(max(0.0, float(weights @ features.covariance @ weights)))
    ratio = volatility / features.benchmark_vol
    core = float(weights[features.core_mask].sum())
    stock = float(weights.max())
    sector = max(float(weights[group].sum()) for group in features.groups)
    distances = {
        "min_vol_ratio": max(0.0, limits.min_vol_ratio - ratio),
        "max_vol_ratio": max(0.0, ratio - limits.max_vol_ratio),
        "min_core_weight": max(0.0, limits.min_core_weight - core),
        "max_stock_weight": max(0.0, stock - limits.max_stock_weight),
        "max_sector_weight": max(0.0, sector - limits.max_sector_weight),
    }
    return {
        "version": "composition_v2_experiment",
        "policy_version": ADAPTIVE_POLICY_VERSION if uses_adaptive_conservative(config) else POLICY_VERSION,
        "risk_basis": "invested_equity_basket",
        "profile": config.profile.value,
        "as_of": features.as_of,
        "volatility": volatility,
        "benchmark_volatility": features.benchmark_vol,
        "volatility_ratio": ratio,
        "core_weight": core,
        "max_stock_weight": stock,
        "max_sector_weight": sector,
        "limits": asdict(limits),
        "violations": [name for name, distance in distances.items() if distance > 1e-9],
        "violation_distance": sum(distances.values()),
    }


def validate_execution(config, lots, prices_per_lot):
    result = metrics(config, lots, prices_per_lot=prices_per_lot)
    if result["violations"]:
        raise ProfileConstraintError("Execution violates composition_v2: "
                                     + ", ".join(result["violations"]))
    return result


def trim_execution_lots(config, planned_lots, prices_per_lot, *, min_lots=None, _risk_search=True):
    """Reduce execution lots to restore composition after a skipped order.

    Never introduce a ticker, increase a lot, or reduce frozen lots. Unspent value remains cash.
    A bounded reduce-only search also tries to restore the volatility band.
    Final execution validation by the caller remains authoritative.
    """
    prices = np.asarray(prices_per_lot, dtype=float)
    lots = np.asarray(planned_lots, dtype=int).copy()
    floor = np.zeros_like(lots) if min_lots is None else np.asarray(min_lots, dtype=int)
    if lots.shape != prices.shape or np.any(lots < 0):
        raise ProfileConstraintError("Invalid execution lots for composition_v2 repair.")
    if floor.shape != lots.shape or np.any(floor < 0) or np.any(floor > lots):
        raise ProfileConstraintError("Frozen execution lots exceed planned composition_v2 lots.")
    features, limits = _prepare(config)
    if np.count_nonzero(lots) * limits.max_stock_weight < 1 - 1e-9:
        raise ProfileConstraintError("Too few traded holdings for the composition_v2 stock-weight cap.")
    if len({features.sectors[i] for i in np.flatnonzero(lots)}) * limits.max_sector_weight < 1 - 1e-9:
        raise ProfileConstraintError("Too few traded sectors for the composition_v2 sector cap.")
    for _ in range(1000):
        result = metrics(config, lots, prices_per_lot=prices)
        violations = result["violations"]
        repairable = {"max_stock_weight", "max_sector_weight", "min_core_weight"}
        if not repairable.intersection(violations):
            break
        values = lots * prices
        total = float(values.sum())
        if "max_stock_weight" in violations:
            index = int(np.argmax(values))
            excess = (values[index] - limits.max_stock_weight * total) / (1 - limits.max_stock_weight)
        elif "max_sector_weight" in violations:
            group = max(features.groups, key=lambda g: float(values[g].sum()))
            removable_group = [int(i) for i in group if lots[i] > floor[i]]
            if not removable_group:
                break
            index = max(removable_group, key=lambda i: values[i])
            excess = (values[group].sum() - limits.max_sector_weight * total) / (1 - limits.max_sector_weight)
        else:
            noncore = np.flatnonzero((~features.core_mask) & (lots > floor))
            if not len(noncore):
                break
            index = int(noncore[np.argmax(values[noncore])])
            excess = total - float(values[features.core_mask].sum()) / limits.min_core_weight
        keep_one = np.count_nonzero(lots) <= config.min_stocks
        removable = int(lots[index] - max(int(floor[index]), int(keep_one)))
        if removable <= 0:
            break
        reduce_by = min(removable, max(1, math.ceil(excess / prices[index] - 1e-10)))
        lots[index] -= reduce_by
        if np.count_nonzero(lots) < config.min_stocks:
            break
    else:
        raise ProfileConstraintError(
            "Execution cannot meet composition_v2 using only the traded planned buys.")
    result = metrics(config, lots, prices_per_lot=prices)
    if repairable.intersection(result["violations"]):
        raise ProfileConstraintError(
            "Execution cannot meet composition_v2 using only the traded planned buys.")
    if not _risk_search or not ({"min_vol_ratio", "max_vol_ratio"} & set(result["violations"])):
        return lots

    def risk_gap(measurement):
        ratio = measurement["volatility_ratio"]
        return max(0., limits.min_vol_ratio-ratio, ratio-limits.max_vol_ratio)

    # The signal-day GA cannot know which next-session orders will fill. At
    # execution, only cancel/reduce planned orders; never select a new ticker.
    for _ in range(30):
        current = metrics(config, lots, prices_per_lot=prices)
        if risk_gap(current) <= 1e-9:
            return lots
        candidates = []
        for index in np.flatnonzero(lots):
            for fraction in (.02, .05, .10, .20, .35, .50):
                trial = lots.copy()
                removable = int(trial[index] - floor[index])
                if removable <= 0:
                    continue
                trial[index] -= min(removable, max(1, math.ceil(trial[index]*fraction)))
                if np.count_nonzero(trial) < config.min_stocks:
                    continue
                try:
                    trial = trim_execution_lots(config, trial, prices, min_lots=floor,
                                                _risk_search=False)
                except ProfileConstraintError:
                    continue
                measurement = metrics(config, trial, prices_per_lot=prices)
                if set(measurement["violations"]) - {"min_vol_ratio", "max_vol_ratio"}:
                    continue
                gap = risk_gap(measurement)
                if gap < risk_gap(current) - 1e-6:
                    candidates.append((gap, -float(trial @ prices), trial))
        if not candidates:
            break
        candidates.sort(key=lambda row: (row[0], row[1]))
        lots = candidates[0][2]
    raise ProfileConstraintError(
        "Execution cannot meet composition_v2 volatility band by reducing "
        "only the traded planned buys; no replacement stock was selected."
    )


class ExperimentalChromosome(Chromosome):
    def clone(self):
        child = type(self)(self.config)
        child.genes = list(self.genes)
        child.exposure = self.exposure
        child.fitness = self.fitness
        return child

    def evaluate_fitness(self, lots_override=None):
        self.profile_constraints_ok = False
        self.profile_metrics = None
        base = super().evaluate_fitness(lots_override=lots_override)
        if self.lots is None or not self.budget_ok or not self.diversification_ok:
            return base
        result = metrics(self.config, self.lots)
        self.profile_metrics = result
        eligible = self.config.data.buy_eligible
        if eligible is not None and self.config.is_rebalance:
            forbidden = [code for code, quantity, can_buy in
                         zip(self.config.data.stock_codes, self.lots, eligible)
                         if not can_buy and quantity > self.config.old_lots.get(code, 0)]
            if forbidden:
                result["violations"].append("buy_ineligible")
                result["violation_distance"] += len(forbidden)
        self.profile_constraints_ok = not result["violations"]
        if not self.profile_constraints_ok:
            self.fitness = -self.config.death_penalty * (1 + result["violation_distance"])
            return self.fitness
        # Historical total-NAV return includes idle cash, as in the base score.
        # The cap prevents an anomalous trailing return from dominating Sharpe.
        signal = float(np.clip(self.roi_annual / result["benchmark_volatility"], -2, 2))
        self.upside_signal = signal
        self.fitness = float(base + result["limits"]["upside_coefficient"] * signal)
        if uses_adaptive_conservative(self.config):
            self.fitness -= ADAPTIVE_VOLATILITY_PENALTY * result["volatility_ratio"]
        return self.fitness


def _reference_weights(config):
    """Estimate a diverse low-risk direction once per population, as of signal."""
    features, limits = _prepare(config)
    available = np.arange(config.data.n_stocks)
    if config.data.buy_eligible is not None:
        available = available[np.asarray(config.data.buy_eligible, dtype=bool)]
    if not len(available):
        return np.zeros(config.data.n_stocks)
    sectors = np.asarray(features.sectors)[available]
    groups = [np.flatnonzero(sectors == sector) for sector in sorted(set(sectors))]
    core = features.core_mask[available]
    policy = ProfilePolicy(limits.min_core_weight, limits.max_stock_weight,
                           limits.max_sector_weight, limits.max_vol_ratio)
    covariance = features.covariance[np.ix_(available, available)]
    weights = _project(np.ones(len(available)), core, groups, policy)
    step = .8 / max(float(np.linalg.norm(covariance, ord=np.inf)), 1e-9)
    for _ in range(100):
        updated = _project(np.maximum(weights - step * (covariance @ weights), 1e-8),
                           core, groups, policy)
        if np.max(np.abs(updated - weights)) < 1e-7:
            break
        weights = updated
    result = np.zeros(config.data.n_stocks)
    result[available] = weights
    return result


def _guided_genes(config, rng, reference=None):
    """A bounded profile-aware proposal; discrete scoring remains authoritative.

    Reserve core slots and spread them across sectors, then project the small
    selected basket toward its risk band. Only a minority of the initial
    population uses these proposals; ordinary random and old-holding seeds stay.
    """
    features, limits = _prepare(config)
    eligible = config.data.buy_eligible
    available = np.arange(config.data.n_stocks)
    if eligible is not None:
        available = available[np.asarray(eligible, dtype=bool)]
    count = min(config.max_stocks, len(available))
    if count < config.min_stocks or count * limits.max_stock_weight < 1:
        return None
    core_slots = math.ceil(limits.min_core_weight / (limits.max_stock_weight - .005))
    core_available = available[features.core_mask[available]]
    if len(core_available) < core_slots:
        return None
    reference = _reference_weights(config) if reference is None else reference
    # Covariance-aware seeds support diversification without forcing extra risk.
    # The other 80% of proposals explore freely under the profile's own fitness.
    rank = -reference * rng.uniform(.8, 1.2, len(reference))
    selected, sector_counts = [], {}
    for core_only, needed in ((True, core_slots), (False, count - core_slots)):
        pool = core_available if core_only else available
        for _ in range(needed):
            choices = [int(i) for i in pool if int(i) not in selected]
            if not choices:
                return None
            index = min(choices, key=lambda i: rank[i] *
                        (1 if sector_counts.get(features.sectors[i], 0) < 3 else .1))
            selected.append(index)
            sector = features.sectors[index]
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    indices = np.asarray(selected, dtype=int)
    sectors = np.asarray(features.sectors)[indices]
    groups = [np.flatnonzero(sectors == sector) for sector in sorted(set(sectors))]
    core = features.core_mask[indices]
    policy = ProfilePolicy(limits.min_core_weight, limits.max_stock_weight,
                           limits.max_sector_weight, limits.max_vol_ratio)
    weights = _project(rng.uniform(.8, 1.2, count), core, groups, policy)
    covariance = features.covariance[np.ix_(indices, indices)]
    step = .5 / max(float(np.linalg.norm(covariance, ord=np.inf)), 1e-9)
    lower = (limits.min_vol_ratio + (.005 if limits.min_vol_ratio else 0)) * features.benchmark_vol
    upper = (limits.max_vol_ratio - .005) * features.benchmark_vol
    for _ in range(32):
        variance = float(weights @ covariance @ weights)
        if lower ** 2 <= variance <= upper ** 2:
            break
        gradient = covariance @ weights
        direction = -1 if variance > upper ** 2 else 1
        weights = _project(np.maximum(weights + direction * step * gradient, 1e-8),
                           core, groups, policy)
    genes = np.zeros(config.data.n_stocks)
    genes[indices] = weights
    return genes.tolist()


def _experimental_population(config, rng):
    factory = (create_rebalance_population if config.is_rebalance and
               config.old_weights is not None else create_initial_population)
    original = factory(config, rng)
    population = []
    guided_start = len(original) - max(1, len(original) // 5)
    reference = _reference_weights(config)
    report_candidates(0, len(original), evaluating=True)
    for index, candidate in enumerate(original):
        chrom = ExperimentalChromosome(config, candidate.genes)
        chrom.exposure = candidate.exposure
        if index >= guided_start:
            guided = _guided_genes(config, rng, reference)
            if guided is not None:
                chrom.genes = guided
        repair_sparse(chrom, preserve_valid=True)
        chrom.evaluate_fitness()
        population.append(chrom)
        report_candidates(index + 1, len(original), evaluating=True)
    return population


def run(config):
    """Return (best_chromosome, generation_history) without touching production.

    A failed search raises ProfileConstraintError; it is not proof that no
    feasible portfolio exists. Pass an explicit seed for reproducibility.
    """
    config = replace(config, profile_policy="legacy")
    from .runtime_performance import evolution_performance
    with evolution_performance(), fitness_inputs_scope(config):
        return _run(config)


def _run(config):
    features, limits = _prepare(config)
    check_affordability(config)
    rng = np.random.default_rng(config.seed)
    population = _experimental_population(config, rng)
    history = []
    best_feasible = None
    closest = None
    stagnant = 0
    prior_best = None
    for generation in range(config.generations):
        population.sort(key=lambda c: c.fitness, reverse=True)
        for candidate in population:
            measurement = candidate.profile_metrics
            if measurement is None or not candidate.budget_ok or not candidate.diversification_ok:
                continue
            other_violations = set(measurement["violations"]) - {"min_vol_ratio", "max_vol_ratio"}
            key = (bool(other_violations), measurement["violation_distance"])
            if closest is None or key < closest[0]:
                closest = (key, measurement)
        feasible = next((c for c in population if c.profile_constraints_ok and
                         c.budget_ok and c.diversification_ok), None)
        if feasible is not None and (best_feasible is None or feasible.fitness > best_feasible.fitness):
            best_feasible = feasible.clone()
            best_feasible.evaluate_fitness()
        leader = population[0]
        report_generation(generation + 1, config.generations, leader.fitness)
        history.append({"generation": generation, "best_fitness": leader.fitness,
                        "feasible_count": sum(c.profile_constraints_ok and c.budget_ok and
                                              c.diversification_ok for c in population)})
        if prior_best is not None and leader.fitness - prior_best < config.improvement_tol:
            stagnant += 1
        else:
            stagnant = 0
        prior_best = max(prior_best, leader.fitness) if prior_best is not None else leader.fitness
        if stagnant >= config.early_stop_patience:
            break
        elites = population[:min(config.elitism_count, config.population_size - 1)]
        children = []
        while len(elites) + len(children) < config.population_size:
            a = tournament_select(population, config.tournament_size, rng)
            b = tournament_select(population, config.tournament_size, rng)
            if rng.random() < config.crossover_rate:
                offspring = sparse_set_crossover(a, b, rng)
            else:
                offspring = (a.clone(), b.clone())
            for child in offspring:
                sparse_portfolio_mutation(child, generation, config.generations, rng)
                child.evaluate_fitness()
                children.append(child)
                if len(elites) + len(children) == config.population_size:
                    break
        population = elites + children
    if best_feasible is None:
        detail = (f" Profil {config.profile.value}: IHSG {features.benchmark_vol:.2%}, "
                  f"batas volatilitas portofolio {limits.min_vol_ratio*features.benchmark_vol:.2%}"
                  f"–{limits.max_vol_ratio*features.benchmark_vol:.2%}.")
        if closest is not None:
            near = closest[1]
            detail += (f" Kandidat terdekat yang dicoba {near['volatility']:.2%} "
                       f"(rasio {near['volatility_ratio']:.2f}); "
                       f"pelanggaran: {', '.join(near['violations'])}.")
        raise ProfileConstraintError("GA belum menemukan portofolio composition_v2 yang memenuhi "
                                     "semua batas." + detail + " Batas tidak dilonggarkan.")
    best_feasible.evaluate_fitness()
    if not best_feasible.profile_constraints_ok:
        raise ProfileConstraintError("Final lot validation failed for composition_v2.")
    return best_feasible, history
