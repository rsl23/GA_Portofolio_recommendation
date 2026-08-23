"""
Chromosome Module - Real-Value Encoded Portfolio Solution
===========================================================

A chromosome is a real-valued vector ``genes[i] in [0,1]`` giving the weight
allocated to stock ``i`` (static index mapping).  Weights are normalised to
sum to 1.0 and converted to BEI lots (floor) for realistic evaluation.  The
chromosome computes its own multi-metric fitness:

    Fitness = Sharpe - lambda*MDD - gamma*AvgCorr + alpha*Fund
              - DeathPenalty_Budget - DeathPenalty_Diversification
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np


class Chromosome:
    """One candidate portfolio solution."""

    def __init__(self, config, genes: Optional[List[float]] = None) -> None:
        self.config = config
        n = config.data.n_stocks if config.data is not None else 0
        self.genes = list(genes) if genes is not None else [0.0] * n
        self.fitness = None
        self.lots = None
        self.cost = 0.0
        self.n_active = 0
        self.budget_ok = True
        self.diversification_ok = True

    # ------------------------------------------------------------------
    # representation helpers
    # ------------------------------------------------------------------
    def normalize(self) -> None:
        """Scale the genes so that their sum equals 1.0."""
        s = math.fsum(self.genes)
        if s <= 1e-12:
            self.genes = [0.0] * len(self.genes)
            return
        self.genes = [g / s for g in self.genes]

    def clone(self) -> "Chromosome":
        new = Chromosome(self.config)
        new.genes = list(self.genes)
        new.fitness = self.fitness
        return new

    # ------------------------------------------------------------------
    # budget / lots
    # ------------------------------------------------------------------
    def to_lots(self) -> np.ndarray:
        """Convert (normalised) weights into integer lots via floor."""
        w = np.asarray(self.genes, dtype=float)
        s = w.sum()
        if s > 1e-12:
            w = w / s
        prices = self.config.data.prices_per_lot
        return np.floor(w * self.config.budget / np.maximum(prices, 1e-9)).astype(int)

    def random_initialization(self, rng: Optional[np.random.Generator] = None) -> None:
        """Randomly allocate weights to ``[min_stocks, max_stocks]`` stocks."""
        rng = rng or np.random.default_rng()
        cfg = self.config
        n = cfg.data.n_stocks
        k = int(rng.integers(cfg.min_stocks, cfg.max_stocks + 1))
        indices = rng.choice(n, size=k, replace=False)
        weights = rng.uniform(0.05, 1.0, size=k)
        genes = np.zeros(n)
        genes[indices] = weights
        self.genes = (genes / genes.sum()).tolist()

    # ------------------------------------------------------------------
    # diversification repair: push a portfolio to the minimum number of lots
    # ------------------------------------------------------------------
    def _reactivate(self, lots: np.ndarray, cost: float):
        """While below the minimum active count, buy an extra lot of the
        strongest weighted-but-unpurchased stock that still fits the budget."""
        cfg = self.config
        data = cfg.data
        budget = cfg.budget
        prices = data.prices_per_lot
        leftover = budget - cost
        lots = lots.astype(int).copy()

        for _ in range(60):
            if int((lots > 0).sum()) >= cfg.min_stocks:
                break
            available = [int(i) for i in range(len(lots))
                         if lots[i] == 0 and prices[i] <= leftover + 1e-6]
            if not available:
                break
            best = max(available, key=lambda i: self.genes[i])
            lots[best] = 1
            leftover -= prices[best]

        cost = int((lots * prices).sum())
        return lots, cost

    # ------------------------------------------------------------------
    # fitness evaluation (multi-metric)
    # ------------------------------------------------------------------
    def evaluate_fitness(self) -> float:
        """Compute Fitness = Sharpe - lam*MDD - gamma*AvgCorr + alpha*Fund
        - DeathPenalty_Budget - DeathPenalty_Diversification."""
        cfg = self.config
        data = cfg.data

        w = np.asarray(self.genes, dtype=float)
        s = w.sum()
        if s <= 1e-12:
            self.fitness = -cfg.death_penalty
            return self.fitness
        # Ini bagian normalisasi untuk mengubah bobot menjadi proporsional 1.0
        w = w / s

        prices = data.prices_per_lot
        budget = cfg.budget

        lots = np.floor(w * budget / np.maximum(prices, 1e-9)).astype(int)
        cost = int((lots * prices).sum())
        lots, cost = self._reactivate(lots, cost)

        active = lots > 0
        n_active = int(active.sum())
        indices = np.where(active)[0]

        budget_death = cfg.death_penalty if cost > budget else 0.0
        div_death = (cfg.death_penalty
                     if n_active < cfg.min_stocks or n_active > cfg.max_stocks else 0.0)

        self.lots = lots
        self.cost = float(cost)
        self.n_active = n_active
        self.budget_ok = cost <= budget
        self.diversification_ok = cfg.min_stocks <= n_active <= cfg.max_stocks

        alloc = lots * prices
        a_sum = alloc.sum()
        if a_sum <= 0:
            self.fitness = -cfg.death_penalty
            return self.fitness
        alloc_w = alloc / a_sum
        aw = alloc_w[indices]

        # portfolio daily-return series
        R = data.returns[indices]
        p_daily = (R * aw[:, None]).sum(axis=0)
        p_daily = p_daily[~np.isnan(p_daily) & ~np.isinf(p_daily)]
        if p_daily.size == 0:
            self.fitness = -cfg.death_penalty
            return self.fitness

        # Sharpe ratio (annualised)
        ret_ann = float(p_daily.mean()) * 252.0
        vol_ann = float(p_daily.std()) * math.sqrt(252.0)
        sharpe = ((ret_ann - cfg.risk_free_rate) / vol_ann) if vol_ann > 1e-12 else 0.0

        # Maximum drawdown
        eq = np.cumprod(1.0 + p_daily)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-12)
        mdd = float(dd.min()) if dd.size else 0.0
        if math.isnan(mdd):
            mdd = 0.0

        # average pairwise correlation among active stocks
        avg_corr = 0.0
        if n_active >= 2:
            C = data.correlation[np.ix_(indices, indices)]
            iu = np.triu_indices(len(indices), k=1)
            vals = C[iu[0], iu[1]]
            vals = vals[~np.isnan(vals)]
            if vals.size:
                avg_corr = float(np.abs(vals).mean())

        # fundamental bonus (weighted by actual allocation)
        fund = data.fundamental_scores[indices]
        fund_avg = float((fund * aw).sum() / (aw.sum() + 1e-12))

        # final multi-metric fitness
        lam = cfg.lambda_mdd
        gamma = cfg.correlation_penalty
        alpha = cfg.fundamental_bonus

        self.fitness = float(
            sharpe
            - lam * abs(mdd)
            - gamma * avg_corr
            + alpha * fund_avg
            - budget_death
            - div_death
        )
        return self.fitness
