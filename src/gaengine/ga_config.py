"""
Configuration Module for Portfolio GA
=====================================

Contains:
    - RiskProfile : enum of risk tolerance levels with MDD penalty weights.
    - GAConfig    : dataclass holding every tunable hyper-parameter of the GA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


REBALANCE_FITNESS_PARAMETERS = {
    "momentum_bonus": 0.0,
    "rebalance_utility_weight": 0.0,
    "cash_return_weight": 0.0,
}


class RiskProfile(Enum):
    """Risk tolerance that sets the Maximum-Drawdown penalty weight (lambda)."""

    CONSERVATIVE = "Conservative"
    MODERATE     = "Moderate"
    AGGRESSIVE   = "Aggressive"

    @property
    def lambda_mdd(self) -> float:
        """lambda -- MDD penalty weight per risk profile."""
        return {
            "Conservative": 2.0,
            "Moderate": 1.0,
            "Aggressive": 0.3,
        }[self.value]

    @staticmethod
    def from_name(name: str) -> "RiskProfile":
        aliases = {"konservatif": RiskProfile.CONSERVATIVE,
                   "moderat": RiskProfile.MODERATE, "agresif": RiskProfile.AGGRESSIVE}
        name = str(name).strip()
        if name.lower() in aliases:
            return aliases[name.lower()]
        for profile in RiskProfile:
            if profile.value.lower() == str(name).lower():
                return profile
        return RiskProfile.MODERATE


@dataclass
class GAConfig:
    """All tunable hyper-parameters for the portfolio GA.

    Attributes
    ----------
    population_size : number of individuals in each generation.
    generations     : maximum number of generations to evolve.
    crossover_rate  : probability that two parents reproduce via crossover.
    tournament_size : number of individuals competing in a tournament.
    elitism_count   : number of top individuals carried over unchanged.
    budget          : total available capital (IDR).
    min_stocks      : minimum number of active stocks (hard constraint).
    max_stocks      : maximum number of active stocks (hard constraint).
    risk_profile    : "Conservative", "Moderate" or "Aggressive".
    risk_free_rate  : annual risk-free rate (fraction).
    correlation_penalty : coefficient gamma on the average-correlation term.
    fundamental_bonus   : coefficient alpha on the fundamental-score term.
    death_penalty       : penalty when a hard constraint is violated.
    mutation_rate_start / mutation_rate_end : adaptive mutation-rate bounds.
    creep_prob          : share of mutations done as "creep" (vs. reset).
    early_stop_patience : stagnant generations allowed before early stop.
    improvement_tol     : min. fitness improvement considered significant.
    seed                : random seed for reproducibility.
    data                : MarketData object used for evaluation.
    """

    # --- GA size / flow -------------------------------------------------
    population_size: int = 200
    generations: int = 500
    crossover_rate: float = 0.8
    tournament_size: int = 5
    elitism_count: int = 10
    operator_policy: str = "sparse_v1"

    # --- portfolio constraints -----------------------------------------
    budget: float = 100_000_000.0    # IDR
    min_stocks: int = 3
    max_stocks: int = 10

    # --- fitness coefficients (final formula) ---------------------------
    risk_profile: str = "Moderate"
    profile_policy: str = "legacy"
    fitness_policy: str = "legacy"
    risk_free_rate: float = 0.0625
    correlation_penalty: float = 0.5   # gamma
    fundamental_bonus: float = 0.8     # alpha
    momentum_bonus: float = 0.0
    dividend_bonus_override: Optional[float] = None
    dividend_yield_cap: float = 0.10
    risk_model: str = "market_stress"
    mdd_residual_weight: float = 0.25
    stress_missing_penalty: float = 0.03
    mdd_penalty_override: Optional[float] = None
    death_penalty: float = 50000.0
    
    annualization: float = math.sqrt(252.0)

    # --- adaptive hybrid mutation --------------------------------------
    mutation_rate_start: float = 0.15
    mutation_rate_end: float = 0.01
    creep_prob: float = 0.5

    # --- stopping / reproducibility -------------------------------------
    early_stop_patience: int = 100
    improvement_tol: float = 1e-4
    seed: Optional[int] = None
    fitness_mode: str = "rebalance_aware"  # "baseline" reproduces the prior scoring for evaluation

    # --- data ------------------------------------------------------------
    data: Optional["MarketData"] = None

    # --- rebalancing ----------------------------------------------------
    is_rebalance: bool = False
    rebalance_model: str = "legacy"
    min_utility_gain: float = 0.002
    scenario_shrinkage: float = 0.5
    old_weights: Optional[Any] = None
    old_lots: Optional[Any] = None
    old_prices_per_lot: Optional[Any] = None
    old_cost_basis: Optional[Any] = None  # accounting context; not a fitness bonus/penalty
    old_cash: Optional[float] = None
    turnover_penalty_beta: float = 0.4
    rebalance_utility_weight: float = 0.0
    allow_cash: bool = False
    min_exposure: float = 0.4  # fraction of NAV invested when allow_cash is enabled
    max_exposure: float = 1.0  # profile risk ceiling; residual NAV remains cash
    cash_return_weight: float = 0.0
    lookthrough_stock_risk: bool = False  # score invested basket risk before cash dilution
    forecast_horizon_days: int = 63
    cvar_tail_fraction: float = 0.10
    lambda_cvar: float = 0.5
    lambda_benchmark_downside: float = 0.5
    lambda_concentration: float = 0.02
    lambda_uncertainty: float = 0.25
    buy_fee_rate: float = 0.0
    sell_fee_rate: float = 0.0

    def __post_init__(self) -> None:
        for name, allowed in {
            'operator_policy': ('legacy', 'sparse_v1'),
            'profile_policy': ('legacy', 'composition_v1'),
            'fitness_policy': ('legacy', 'net_cvar_ihsg_v1'),
            'rebalance_model': ('legacy', 'guarded_v1'),
            'risk_model': ('mdd', 'market_stress'),
            'fitness_mode': ('baseline', 'rebalance_aware', 'decision_aware'),
        }.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f'{name} tidak dikenal: {getattr(self, name)}')
        for name in ('momentum_bonus', 'rebalance_utility_weight', 'cash_return_weight',
                     'stress_missing_penalty', 'min_utility_gain', 'lambda_cvar',
                     'lambda_benchmark_downside', 'lambda_concentration', 'lambda_uncertainty',
                     'buy_fee_rate', 'sell_fee_rate'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} harus finite dan non-negatif.')
        if (not math.isfinite(self.dividend_yield_cap) or self.dividend_yield_cap <= 0 or
            self.dividend_bonus_override is not None and
            (not math.isfinite(self.dividend_bonus_override) or self.dividend_bonus_override < 0)):
            raise ValueError('Parameter bonus dividen tidak valid.')
        if (not math.isfinite(self.mdd_residual_weight) or not 0 <= self.mdd_residual_weight <= 1 or
            not math.isfinite(self.scenario_shrinkage) or not 0 <= self.scenario_shrinkage <= 1 or
            self.forecast_horizon_days < 1 or not 0 < self.cvar_tail_fraction <= 1):
            raise ValueError('Parameter risiko/skenario tidak valid.')
        if self.profile_policy != 'legacy':
            if str(self.risk_profile).strip().lower() not in (
                'conservative','moderate','aggressive','konservatif','moderat','agresif'):
                raise ValueError('Profil risiko tidak dikenal untuk composition_v1.')
            if self.fitness_mode != 'rebalance_aware':
                raise ValueError('composition_v1 memerlukan fitness rebalance_aware.')
        if (not 0 <= self.crossover_rate <= 1 or
            not 0 <= self.mutation_rate_end <= self.mutation_rate_start <= 1 or
            not 0 <= self.creep_prob <= 1 or self.tournament_size < 1):
            raise ValueError('Parameter operator GA tidak valid.')
        if not 0 < self.min_exposure <= self.max_exposure <= 1:
            raise ValueError("Exposure harus memenuhi 0 < min_exposure <= max_exposure <= 1.")
        if self.lookthrough_stock_risk and not self.allow_cash:
            raise ValueError("lookthrough_stock_risk memerlukan allow_cash.")

    @property
    def profile(self) -> RiskProfile:
        return RiskProfile.from_name(self.risk_profile)

    @property
    def dividend_bonus_weight(self) -> float:
        return (self.dividend_bonus_override if self.dividend_bonus_override is not None else {
            RiskProfile.CONSERVATIVE: 0.15, RiskProfile.MODERATE: 0.08, RiskProfile.AGGRESSIVE: 0.03,
        }[self.profile])

    @property
    def lambda_mdd(self) -> float:
        return (self.profile.lambda_mdd if self.mdd_penalty_override is None
                else self.mdd_penalty_override)
