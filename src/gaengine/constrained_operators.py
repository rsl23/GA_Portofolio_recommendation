"""Sparse asset-set operators with an affordable one-lot reserve per holding.

This policy changes proposal generation, not the investment fitness. It keeps
asset selection separate from allocation changes and preserves hard constraints
without increasing the budget or relaxing the minimum number of holdings.
"""
from __future__ import annotations

import numpy as np


class PortfolioFeasibilityError(ValueError):
    """The budget cannot buy the required minimum number of distinct stocks."""


def investment_limit(chromosome):
    cfg = chromosome.config
    exposure = (float(np.clip(chromosome.exposure, cfg.min_exposure, cfg.max_exposure))
                if cfg.allow_cash and cfg.fitness_mode != 'baseline' else 1.)
    return cfg.budget * exposure


def check_affordability(config):
    prices = np.asarray(config.data.prices_per_lot, dtype=float)
    if (prices.ndim != 1 or not np.isfinite(prices).all() or np.any(prices <= 0)
            or not np.isfinite(config.budget) or config.budget <= 0
            or not 1 <= config.min_stocks <= config.max_stocks
            or len(prices) < config.min_stocks):
        raise PortfolioFeasibilityError('Harga, modal, atau batas jumlah saham tidak valid.')
    limit = config.budget * (config.max_exposure
                            if config.allow_cash and config.fitness_mode != 'baseline' else 1.)
    if np.sort(prices)[:config.min_stocks].sum() > limit:
        raise PortfolioFeasibilityError(
            'Modal/exposure tidak cukup membeli satu lot pada minimum jumlah saham; '
            'batas modal dan jumlah saham tidak dilonggarkan.')


def repair_sparse(chromosome, preserve_valid=False):
    """Reserve one lot per chosen stock, then allocate only the residual budget.

    Genes remain continuous NAV allocations, compatible with the existing floor
    and accounting paths. Selection repair does not inspect returns or fitness.
    Composition-specific constraints remain with profile_policy's own repair.
    """
    cfg = chromosome.config
    prices = cfg.data.prices_per_lot
    limit = investment_limit(chromosome)
    if preserve_valid:
        raw = np.asarray(chromosome.genes, dtype=float)
        if np.isfinite(raw).all() and np.all(raw >= 0) and raw.sum() > 0:
            lots = chromosome.to_lots()
            if cfg.min_stocks <= np.count_nonzero(lots) <= cfg.max_stocks and lots @ prices <= limit:
                return chromosome
    cheapest = np.argsort(prices, kind='stable')
    minimum_cost = float(prices[cheapest[:cfg.min_stocks]].sum())
    if minimum_cost > limit:
        # Variable cash exposure must still accommodate the hard minimum.
        if cfg.allow_cash and cfg.fitness_mode != 'baseline':
            required = float(np.nextafter(minimum_cost / cfg.budget, 1.))
            chromosome.exposure = min(cfg.max_exposure, max(chromosome.exposure, required))
            if minimum_cost > investment_limit(chromosome):
                raise PortfolioFeasibilityError('Exposure maksimum tidak cukup untuk minimum saham.')
            limit = investment_limit(chromosome)
        else:
            raise PortfolioFeasibilityError('Modal tidak cukup untuk minimum saham.')
    raw = np.asarray(chromosome.genes, dtype=float)
    raw = np.where(np.isfinite(raw), np.maximum(raw, 0.), 0.)
    order = np.argsort(-raw, kind='stable')
    selected = [int(i) for i in order if raw[i] > 0][:cfg.max_stocks]
    for i in cheapest:
        if len(selected) >= cfg.min_stocks:
            break
        if int(i) not in selected:
            selected.append(int(i))
    # When one lot of the chosen set is too expensive, reduce cardinality first,
    # then substitute expensive positions with the cheapest available ones.
    while prices[selected].sum() > limit and len(selected) > cfg.min_stocks:
        selected.remove(max(selected, key=lambda i: prices[i]))
    while prices[selected].sum() > limit:
        expensive = max(selected, key=lambda i: prices[i])
        replacement = next((int(i) for i in cheapest if int(i) not in selected), None)
        if replacement is None or prices[replacement] >= prices[expensive]:
            raise PortfolioFeasibilityError('Tidak ditemukan kombinasi minimum saham terjangkau.')
        selected.remove(expensive)
        selected.append(replacement)
    idx = np.asarray(selected, dtype=int)
    reserve = float(prices[idx].sum())
    weights = raw[idx]
    weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(idx), 1 / len(idx))
    allocation = prices[idx] + max(0., limit - reserve) * weights
    result = np.zeros(len(prices))
    result[idx] = allocation / allocation.sum()
    chromosome.genes = result.tolist()
    chromosome.fitness = None
    return chromosome


def sparse_set_crossover(parent_a, parent_b, rng):
    """Recombine an asset set without reference to arbitrary ticker positions."""
    cfg = parent_a.config
    ga, gb = np.asarray(parent_a.genes), np.asarray(parent_b.genes)
    sa, sb = set(np.flatnonzero(ga > 0)), set(np.flatnonzero(gb > 0))
    union, common = sorted(sa | sb), sorted(sa & sb)
    children = []
    for parent in (parent_a, parent_b):
        child = parent.clone()
        low = max(cfg.min_stocks, min(len(sa), len(sb)))
        high = min(cfg.max_stocks, max(len(sa), len(sb)), len(union))
        k = int(rng.integers(low, high + 1)) if high >= low else min(len(union), cfg.max_stocks)
        chosen = list(common[:k])
        remaining = [i for i in union if i not in chosen]
        if k > len(chosen):
            p = (ga + gb)[remaining]
            chosen.extend(rng.choice(remaining, k-len(chosen), replace=False, p=p/p.sum()).tolist())
        mix = float(rng.uniform(.15, .85))
        genes = np.zeros(len(ga))
        genes[chosen] = (mix*ga + (1-mix)*gb)[chosen]
        child.genes = genes.tolist()
        children.append(child)
    return tuple(children)


def sparse_portfolio_mutation(chromosome, generation, max_generation, rng):
    """One bounded set move (SWAP/ADD/DELETE) or a pairwise weight transfer.

    Move probabilities are policy constants frozen before the paired study.
    Allocation step sizes decay in time; no claim of diversity feedback is made.
    """
    cfg = chromosome.config
    genes = np.maximum(np.asarray(chromosome.genes, dtype=float), 0.)
    active = np.flatnonzero(genes > 0)
    inactive = np.flatnonzero(genes == 0)
    actions, probabilities = ['transfer'], [.35]
    if len(active) and len(inactive):
        actions.append('swap'); probabilities.append(.35)
    if len(active) < cfg.max_stocks and len(inactive):
        actions.append('add'); probabilities.append(.15)
    if len(active) > cfg.min_stocks:
        actions.append('delete'); probabilities.append(.15)
    probabilities = np.asarray(probabilities) / sum(probabilities)
    action = rng.choice(actions, p=probabilities)
    if action == 'swap':
        old, new = int(rng.choice(active)), int(rng.choice(inactive))
        genes[new], genes[old] = genes[old], 0.
    elif action == 'add':
        new = int(rng.choice(inactive))
        share = float(rng.uniform(.03, .20))
        genes *= 1-share
        genes[new] = share
    elif action == 'delete':
        genes[int(rng.choice(active))] = 0.
    progress = generation / max(1, max_generation-1)
    active = np.flatnonzero(genes > 0)
    if len(active) >= 2 and (action == 'transfer' or rng.random() < .5):
        donor, recipient = rng.choice(active, 2, replace=False)
        fraction = float(rng.uniform(.01, .30 - .25*progress))
        delta = genes[donor]*fraction
        genes[donor] -= delta
        genes[recipient] += delta
    chromosome.genes = genes.tolist()
    if (cfg.allow_cash and cfg.fitness_mode != 'baseline'
            and cfg.min_exposure < cfg.max_exposure and rng.random() < .15):
        chromosome.exposure = float(np.clip(chromosome.exposure + rng.normal(0, .04),
                                            cfg.min_exposure, cfg.max_exposure))
    repair_sparse(chromosome)
