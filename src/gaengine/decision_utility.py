"""Experimental, point-in-time scenario utility for portfolio decisions.

The scenarios are historical *joint* rolling returns ending at the decision
date. They are not calibrated forecasts or a promise of future performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateAccount:
    initial_wealth: float
    rows: tuple


def prepare_candidate_account(stock_codes, prices_per_lot, old_lots, old_prices_per_lot, old_cash):
    """Freeze transaction lookup data for one run; retain legacy sum order."""
    current = dict(zip(stock_codes, map(float, prices_per_lot)))
    indices = dict(zip(stock_codes, range(len(stock_codes))))
    for code, qty in old_lots.items():
        if qty < 0 or code not in old_prices_per_lot or not math.isfinite(old_prices_per_lot[code]) or old_prices_per_lot[code] <= 0:
            raise ValueError(f"Harga/lot lama tidak valid untuk {code}.")
    initial_wealth = float(old_cash) + sum(qty * old_prices_per_lot[code] for code, qty in old_lots.items())
    if not math.isfinite(initial_wealth) or initial_wealth <= 0:
        raise ValueError("NAV sebelum transaksi harus positif.")
    rows = []
    for code in set(old_lots) | set(stock_codes):
        price = current.get(code, old_prices_per_lot.get(code))
        if price is None:
            raise ValueError(f"Harga transaksi tidak tersedia untuk {code}.")
        rows.append((indices.get(code), old_lots.get(code, 0), price))
    return CandidateAccount(initial_wealth, tuple(rows))


def empirical_cvar(losses: np.ndarray, tail_fraction: float) -> float:
    """Exact equal-probability upper-tail mean, including fractional VaR mass."""
    values = np.asarray(losses, dtype=float)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or not 0 < tail_fraction <= 1):
        raise ValueError("Loss/CVaR tail tidak valid.")
    ordered = np.sort(values)[::-1]
    mass = len(ordered) * tail_fraction
    whole = int(math.floor(mass))
    remainder = mass - whole
    total = float(ordered[:whole].sum())
    if remainder > 0 and whole < len(ordered):
        total += remainder * ordered[whole]
    return float(total / mass)


def joint_horizon_returns(daily_returns: np.ndarray, horizon: int) -> np.ndarray:
    """Return (scenario, stock) compounded windows from aligned past returns."""
    history = np.asarray(daily_returns, dtype=float)
    if history.ndim != 2 or horizon < 1 or history.shape[1] < 3 * horizon:
        raise ValueError("Riwayat minimal tiga horizon diperlukan untuk skenario.")
    if not np.isfinite(history).all() or np.any(history <= -1):
        raise ValueError("Return historis hilang atau tidak valid untuk skenario.")
    logs = np.log1p(history)
    cumulative = np.pad(np.cumsum(logs, axis=1), ((0, 0), (1, 0)))
    n_windows = history.shape[1] - horizon + 1
    ends = np.unique(np.linspace(0, n_windows - 1, min(64, n_windows), dtype=int))
    return np.expm1((cumulative[:, ends + horizon] - cumulative[:, ends]).T)


def candidate_utility(
    lots: np.ndarray,
    stock_codes: list[str],
    prices_per_lot: np.ndarray,
    old_lots: dict[str, int],
    old_prices_per_lot: dict[str, float],
    old_cash: float,
    scenarios: np.ndarray,
    cash_horizon_return: float,
    buy_fee_rate: float,
    sell_fee_rate: float,
    lambda_cvar: float,
    lambda_concentration: float,
    lambda_uncertainty: float,
    tail_fraction: float,
    effective_samples: int,
    *,
    diagnostics: dict | None = None,
    prepared_account: CandidateAccount | None = None,
) -> tuple[float, float, float]:
    """Return utility, post-trade cash, and transaction cost in IDR.

    Costs enter post-trade cash exactly once. CVaR and uncertainty are fractions
    of pre-trade NAV, like expected return. No acquisition cost enters utility.
    """
    raw_lots = np.asarray(lots)
    if (not np.isfinite(raw_lots).all() or
        not np.array_equal(raw_lots, raw_lots.astype(int))):
        raise ValueError("Jumlah lot harus integer dan finite.")
    q = raw_lots.astype(int)
    prices = np.asarray(prices_per_lot, dtype=float)
    samples = np.asarray(scenarios, dtype=float)
    if (q.shape != prices.shape or len(stock_codes) != len(q) or
        samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] != len(q) or
        not np.isfinite(samples).all() or not np.isfinite(prices).all() or
        np.any(prices <= 0) or np.any(q < 0) or
        not 0 < tail_fraction <= 1 or effective_samples < 1 or
        old_cash < 0 or cash_horizon_return < -1 or
        not np.isfinite([old_cash, cash_horizon_return, buy_fee_rate,
                         sell_fee_rate, lambda_cvar, lambda_concentration,
                         lambda_uncertainty]).all() or
        min(buy_fee_rate, sell_fee_rate, lambda_cvar,
            lambda_concentration, lambda_uncertainty) < 0):
        raise ValueError("Input skenario, lot, harga, atau parameter utility tidak valid.")
    cash = float(old_cash)
    fees = 0.0
    if prepared_account is None:
        current = dict(zip(stock_codes, map(float, prices)))
        next_lots = dict(zip(stock_codes, map(int, q)))
        for code, qty in old_lots.items():
            if qty < 0 or code not in old_prices_per_lot or not math.isfinite(old_prices_per_lot[code]) or old_prices_per_lot[code] <= 0:
                raise ValueError(f"Harga/lot lama tidak valid untuk {code}.")
        initial_wealth = float(old_cash) + sum(
            qty * old_prices_per_lot[code] for code, qty in old_lots.items())
        if not math.isfinite(initial_wealth) or initial_wealth <= 0:
            raise ValueError("NAV sebelum transaksi harus positif.")
        for code in set(old_lots) | set(stock_codes):
            delta = next_lots.get(code, 0) - old_lots.get(code, 0)
            price = current.get(code, old_prices_per_lot.get(code))
            if price is None:
                raise ValueError(f"Harga transaksi tidak tersedia untuk {code}.")
            traded = abs(delta) * price
            fee = traded * (buy_fee_rate if delta > 0 else sell_fee_rate)
            fees += fee
            cash -= delta * price + fee
    else:
        initial_wealth = prepared_account.initial_wealth
        for index, old_quantity, price in prepared_account.rows:
            delta = (int(q[index]) if index is not None else 0) - old_quantity
            traded = abs(delta) * price
            fee = traded * (buy_fee_rate if delta > 0 else sell_fee_rate)
            fees += fee
            cash -= delta * price + fee
    if cash < -0.01:
        raise ValueError("Kas setelah transaksi negatif.")
    cash = max(cash, 0.0)
    position_values = q * prices
    net_returns = (
        cash * (1 + cash_horizon_return)
        + (samples + 1.0) @ position_values
    ) / initial_wealth - 1.0
    losses = -net_returns
    signed_cvar = empirical_cvar(losses, tail_fraction)
    # A loss penalty earns no bonus when even the worst tail is profitable.
    downside_cvar = max(0.0, signed_cvar)
    weights = position_values / initial_wealth
    concentration = float(np.square(weights).sum())
    uncertainty = float(np.std(net_returns)) / math.sqrt(effective_samples)
    utility = (float(np.mean(net_returns)) - lambda_cvar * downside_cvar
               - lambda_concentration * concentration
               - lambda_uncertainty * uncertainty)
    if diagnostics is not None:
        diagnostics.update(mean_net_return=float(np.mean(net_returns)),
            signed_cvar_loss=signed_cvar, downside_cvar=downside_cvar,
            lambda_cvar=lambda_cvar,
            cvar_penalty=lambda_cvar*downside_cvar,
            utility=float(utility), signal_cash_after=cash, signal_transaction_costs=fees,
            scenario_count=len(net_returns), tail_fraction=tail_fraction)
    return utility, cash, fees
