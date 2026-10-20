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
        self.exposure = config.max_exposure if config.allow_cash else 1.0
        self.fitness = None
        self.lots = None
        self.cost = 0.0
        self.n_active = 0
        self.budget_ok = True
        self.diversification_ok = True

        # --- metrik yang disimpan untuk pelaporan hasil -----------------
        self.sharpe_ratio = 0.0
        self.roi_annual = 0.0
        self.max_drawdown = 0.0
        self.avg_correlation = 0.0
        self.skor_fundamental = 0.0
        self.turnover_rate = 0.0
        self.rebalance_utility = 0.0

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
        new.exposure = self.exposure
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
        exposure = (float(np.clip(self.exposure, self.config.min_exposure,
                                  self.config.max_exposure))
                    if self.config.allow_cash and self.config.fitness_mode != "baseline" else 1.0)
        return np.floor(w * self.config.budget * exposure / np.maximum(prices, 1e-9)).astype(int)

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
        if cfg.allow_cash and cfg.fitness_mode != "baseline":
            self.exposure = (cfg.min_exposure if cfg.min_exposure == cfg.max_exposure
                             else float(rng.uniform(cfg.min_exposure, cfg.max_exposure)))

    # ------------------------------------------------------------------
    # diversification repair: push a portfolio to the minimum number of lots
    # ------------------------------------------------------------------
    def _reactivate(self, lots: np.ndarray, cost: float):
        """While below the minimum active count, buy an extra lot of the
        strongest weighted-but-unpurchased stock that still fits the budget."""
        cfg = self.config
        data = cfg.data
        budget = (cfg.budget * float(np.clip(self.exposure, cfg.min_exposure, cfg.max_exposure))
                  if cfg.allow_cash and cfg.fitness_mode != "baseline" else cfg.budget)
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

        cost = float((lots * prices).sum())
        return lots, cost

    # ------------------------------------------------------------------
    # fitness evaluation (multi-metric)
    # ------------------------------------------------------------------
    def evaluate_fitness(self, lots_override: Optional[np.ndarray] = None) -> float:
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

        if lots_override is None:
            exposure = (float(np.clip(self.exposure, cfg.min_exposure, cfg.max_exposure))
                        if cfg.allow_cash and cfg.fitness_mode != "baseline" else 1.0)
            self.exposure = exposure
            lots = np.floor(w * budget * exposure / np.maximum(prices, 1e-9)).astype(int)
            cost = float((lots * prices).sum())
            lots, cost = self._reactivate(lots, cost)
        else:
            lots = np.asarray(lots_override, dtype=int)
            if lots.shape != prices.shape or np.any(lots < 0):
                raise ValueError("lots_override harus berisi lot non-negatif untuk setiap kandidat.")
            cost = float((lots * prices).sum())

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
        # Keep the residual cash at zero return. Normalising by invested
        # capital alone makes a nearly-all-cash candidate look as risky and
        # profitable as the fully invested version of the same holdings.
        baseline_mode = getattr(cfg, "fitness_mode", "rebalance_aware") == "baseline"
        portfolio_w = alloc / (a_sum if baseline_mode else budget)
        aw = portfolio_w[indices]

        # portfolio daily-return series
        R = data.returns[indices]
        p_daily = (R * aw[:, None]).sum(axis=0)
        p_daily = p_daily[~np.isnan(p_daily) & ~np.isinf(p_daily)]
        if p_daily.size == 0:
            self.fitness = -cfg.death_penalty
            return self.fitness
        lookthrough = bool(cfg.allow_cash and not baseline_mode and cfg.lookthrough_stock_risk)
        # Cash lowers the realized portfolio drawdown, but must not make a
        # concentrated, volatile stock basket look intrinsically low-risk.
        stock_daily = ((R * (alloc[indices] / a_sum)[:, None]).sum(axis=0)
                       if lookthrough else p_daily)
        stock_daily = stock_daily[np.isfinite(stock_daily)]

        # Sharpe ratio (annualised)
        ret_ann = float(p_daily.mean()) * 252.0
        vol_ann = float(p_daily.std()) * math.sqrt(252.0)
        sharpe = ((ret_ann - cfg.risk_free_rate) / vol_ann) if vol_ann > 1e-12 else 0.0
        # With an explicit cash gene, use the invested fraction's hurdle in
        # the search score. The reported Sharpe above remains the conventional
        # full-portfolio metric; MDD and returns still include zero-return cash.
        invested_fraction = float(a_sum / budget)
        score_sharpe = (
            (ret_ann - cfg.risk_free_rate * invested_fraction) / vol_ann
            if cfg.allow_cash and not baseline_mode and vol_ann > 1e-12 else sharpe
        )
        if lookthrough:
            stock_vol = float(stock_daily.std()) * math.sqrt(252.0)
            score_sharpe = ((float(stock_daily.mean()) * 252.0 - cfg.risk_free_rate)
                            / stock_vol) if stock_vol > 1e-12 else 0.0

        # Maximum drawdown
        eq = np.cumprod(1.0 + p_daily)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-12)
        mdd = float(dd.min()) if dd.size else 0.0
        if math.isnan(mdd):
            mdd = 0.0
        score_mdd = mdd
        if lookthrough:
            stock_equity = np.cumprod(1.0 + stock_daily)
            stock_peak = np.maximum.accumulate(stock_equity)
            score_mdd = float(((stock_equity - stock_peak) /
                               np.maximum(stock_peak, 1e-12)).min())

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
        fund_avg = (float((fund * aw).sum() / (aw.sum() + 1e-12))
                    if baseline_mode or lookthrough else float((fund * aw).sum()))

        # final multi-metric fitness
        lam = cfg.lambda_mdd
        gamma = cfg.correlation_penalty
        alpha = cfg.fundamental_bonus

        # Cost basis determines the accounting label (realized gain/loss), not
        # whether a sale improves the portfolio from today's market price.
        # Using past profit as a fitness bonus would make tiny gains trigger
        # take-profit and large losses discourage rational cut-loss decisions.
        # --- Turnover penalty (rebalancing only) ---
        turnover_penalty = 0.0
        self.turnover_rate = 0.0
        self.rebalance_utility = 0.0
        if (not baseline_mode and getattr(cfg, 'is_rebalance', False) and
            getattr(cfg, 'old_lots', None)):
            # Marginal change in the *recent-vs-prior* risk/reward signal.
            # Neither realized gains nor losses enter this utility.
            old_alloc = np.asarray([
                cfg.old_lots.get(code, 0) * prices[i]
                for i, code in enumerate(data.stock_codes)
            ], dtype=float)
            self.rebalance_utility = float(np.dot(
                (alloc - old_alloc) / budget, data.rebalance_signal
            ))
        if baseline_mode and getattr(cfg, 'is_rebalance', False) and getattr(cfg, 'old_weights', None) is not None:
            self.turnover_rate = float(np.abs(portfolio_w - cfg.old_weights).sum())
            turnover_penalty = getattr(cfg, 'turnover_penalty_beta', 0.4) * self.turnover_rate
        elif getattr(cfg, 'is_rebalance', False) and getattr(cfg, 'old_lots', None) is not None:
            old_lots = cfg.old_lots
            old_prices = getattr(cfg, 'old_prices_per_lot', None) or {}
            if old_prices:
                new_lots = dict(zip(data.stock_codes, lots))
                candidate_prices = dict(zip(data.stock_codes, data.prices_per_lot))
                traded_value = sum(
                    abs(int(new_lots.get(code, 0)) - int(old_lots.get(code, 0)))
                    * float(old_prices.get(code, candidate_prices.get(code, 0.0)))
                    for code in set(new_lots) | set(old_lots)
                )
                self.turnover_rate = float(traded_value / budget)
            elif getattr(cfg, 'old_weights', None) is not None:
                # Compatibility for callers that have no previous price map.
                self.turnover_rate = float(np.abs(portfolio_w - cfg.old_weights).sum())
            turnover_penalty = getattr(cfg, 'turnover_penalty_beta', 0.4) * self.turnover_rate

        self.fitness = float(
            score_sharpe
            - lam * abs(score_mdd)
            - gamma * avg_corr
            + alpha * fund_avg
            - budget_death
            - div_death
            - turnover_penalty
            + getattr(cfg, 'rebalance_utility_weight', 0.0) * self.rebalance_utility
            + (cfg.cash_return_weight * ret_ann if cfg.allow_cash and not baseline_mode else 0.0)
        )

        # Simpan metrik lengkap agar bisa dilaporkan oleh controller
        self.sharpe_ratio = sharpe
        self.roi_annual = ret_ann
        self.max_drawdown = abs(mdd)
        self.avg_correlation = avg_corr
        self.skor_fundamental = fund_avg
        return self.fitness
