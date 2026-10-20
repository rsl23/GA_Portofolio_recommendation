"""Read-only, walk-forward GA rebalance evaluation against price-only IHSG.

This is a research harness, not an execution endpoint. Signals use data ending
the previous trading day. Orders settle at the next scheduled session's Close.
No database writes, dividends, deposits, or corporate actions are made.
Optional broker fees are illustrative simulation assumptions, not actual trades.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from src.backend.controller.portfolio_controller import _add_months
from src.gaengine.data_loader import DATA_DIR, build_market_data
from src.gaengine.engine import GeneticEngine
from src.gaengine.ga_config import GAConfig
from src.gaengine.rebalancer import apply_no_trade_band, fit_buys_to_available_cash


@lru_cache(maxsize=1)
def historical_closes() -> tuple[pd.DataFrame, pd.Series]:
    prices = pd.read_parquet(
        DATA_DIR / "Master_OHLCV_15Tahun.parquet", columns=["Date", "Ticker", "Close"]
    )
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["Ticker"] = prices["Ticker"].str.removesuffix(".JK")
    stocks = prices.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last").sort_index()
    context = pd.read_parquet(
        DATA_DIR / "Master_Market_Context_15Tahun.parquet",
        columns=["Date", "Ticker", "Close"],
        filters=[("Ticker", "==", "^JKSE")],
    )
    ihsg = pd.Series(
        context["Close"].to_numpy(dtype=float), index=pd.to_datetime(context["Date"])
    ).dropna().sort_index()
    return stocks, ihsg


@lru_cache(maxsize=64)
def signal_data(day: date):
    result = build_market_data(min_price=1.0, backtest=True, date_ref=day)
    if result is None or result.n_stocks == 0:
        raise ValueError(f"MarketData kosong pada hari keputusan {day}.")
    return result


def describe_rebalance_decisions(
    old_lots: dict[str, int], new_lots: dict[str, int],
    old_basis: dict[str, float], execution_prices: dict[str, float],
    forward_signals: dict[str, float],
) -> list[dict]:
    """Explain trades using pre-trade basis; P/L never enters the GA score."""
    decisions = []
    for code in sorted(set(old_lots) | set(new_lots)):
        before, after = old_lots.get(code, 0), new_lots.get(code, 0)
        if before == 0 and after == 0:
            continue
        action = ("BUY" if before == 0 else "SELL" if after == 0 else
                  "ADD" if after > before else "REDUCE" if after < before else "HOLD")
        price = float(execution_prices[code])
        basis = old_basis.get(code) if before else None
        unrealized_pct_before = (price / basis - 1.0
                                 if basis is not None and basis > 0 else None)
        realized_delta = (((before - after) * 100 * (price - basis))
                          if after < before and basis is not None else
                          None if after < before else 0.0)
        decisions.append({
            "ticker": code, "action": action,
            "old_lots": before, "new_lots": after,
            "execution_price": price, "average_cost_before": basis,
            "unrealized_pct_before": unrealized_pct_before,
            "realized_delta": realized_delta,
            "forward_signal": forward_signals.get(code),
        })
    return decisions


def run_scenario(
    start: date, end: date, profile: str, fitness_mode: str,
    capital: float = 500_000_000.0, interval_months: int = 3,
    population: int = 40, generations: int = 80, seed: int = 42,
    fitness_parameters: dict | None = None,
    buy_fee_rate: float = 0.0, sell_fee_rate: float = 0.0,
) -> dict:
    closes, ihsg = historical_closes()
    trading_days = closes.index
    schedule = []
    scheduled = start
    while scheduled < end:
        index = trading_days.searchsorted(pd.Timestamp(scheduled), side="right") - 1
        if index <= 0:
            raise ValueError(f"Hari bursa/sinyal tidak tersedia untuk {scheduled}.")
        execution_day = trading_days[index]
        if not schedule or execution_day > schedule[-1]:
            schedule.append(execution_day)
        scheduled = _add_months(scheduled, interval_months)
    if not schedule:
        raise ValueError("Tanggal akhir harus setelah tanggal awal.")

    cash = float(capital)
    lots: dict[str, int] = {}
    basis: dict[str, float] = {}
    realized = 0.0
    cumulative_costs = 0.0
    events = []
    states = []
    for step, execution_day in enumerate(schedule):
        signal_day = trading_days[trading_days.get_loc(execution_day) - 1]
        data = signal_data(signal_day.date())
        candidates = data.stock_codes
        required = set(lots) | set(candidates)
        signal_marks = closes.loc[signal_day, list(required)]
        execution_marks = closes.loc[execution_day, list(required)]
        if (signal_marks.isna() | execution_marks.isna()).any():
            missing = (signal_marks.isna() | execution_marks.isna())
            raise ValueError(f"Harga aktual tidak tersedia pada {execution_day.date()}: {list(missing[missing].index)}")

        old_prices = {code: float(signal_marks[code]) * 100 for code in required}
        signal_nav = cash + sum(qty * old_prices[code] for code, qty in lots.items())
        if signal_nav <= 0:
            raise ValueError("NAV sebelum keputusan tidak positif.")
        old_weights = [lots.get(code, 0) * old_prices[code] / signal_nav for code in candidates]
        config = GAConfig(
            population_size=population, generations=generations,
            elitism_count=max(1, round(population * 0.05)), budget=signal_nav,
            risk_profile=profile, min_stocks=3, max_stocks=10,
            risk_free_rate=data.risk_free_rate, data=data,
            is_rebalance=step > 0, old_lots=dict(lots), old_weights=old_weights,
            old_prices_per_lot=old_prices, old_cost_basis=dict(basis), seed=seed + step,
            fitness_mode=fitness_mode,
            **(fitness_parameters or {}),
        )
        solution = GeneticEngine(config).run(verbose=False)
        if not solution.diversification_ok or not solution.budget_ok:
            raise ValueError("GA tidak menghasilkan portofolio yang memenuhi batas saham dan budget.")
        if step:
            if config.allow_cash:
                full_weights = (np.asarray(solution.genes) * solution.exposure).tolist()
                new_weights = apply_no_trade_band(
                    full_weights, old_weights, threshold=0.03,
                    min_active=config.min_stocks, max_active=config.max_stocks,
                )
                solution.exposure = float(np.clip(
                    sum(new_weights), config.min_exposure, config.max_exposure
                ))
                solution.genes = new_weights
            else:
                solution.genes = apply_no_trade_band(
                    solution.genes, old_weights, threshold=0.03,
                    min_active=config.min_stocks, max_active=config.max_stocks,
                )
            solution.normalize()
            solution.evaluate_fitness()
            if not solution.diversification_ok or not solution.budget_ok:
                raise ValueError("No-trade band melanggar batas saham atau budget.")
        next_lots = {code: int(qty) for code, qty in zip(candidates, solution.lots) if qty > 0}
        next_lots, _ = fit_buys_to_available_cash(
            lots, next_lots, cash,
            {code: float(execution_marks[code]) for code in required},
            dict(zip(candidates, solution.genes)), config.min_stocks,
            buy_fee_rate, sell_fee_rate,
        )
        solution.evaluate_fitness(
            lots_override=np.asarray([next_lots.get(code, 0) for code in candidates], dtype=int)
        )
        if not solution.diversification_ok or not solution.budget_ok:
            raise ValueError("Lot eksekusi melanggar batas saham atau budget.")
        delta = {code: next_lots.get(code, 0) - lots.get(code, 0)
                 for code in set(lots) | set(next_lots)
                 if next_lots.get(code, 0) != lots.get(code, 0)}
        decisions = describe_rebalance_decisions(
            lots, next_lots, basis,
            {code: float(execution_marks[code]) for code in required},
            dict(zip(data.stock_codes, data.rebalance_signal)),
        )
        nav_before = cash + sum(qty * 100 * float(execution_marks[code]) for code, qty in lots.items())
        next_cash = cash
        for code, change in delta.items():
            price = float(execution_marks[code])
            next_cash -= change * 100 * price
            if change < 0:
                realized += -change * 100 * (price - basis[code])
            else:
                old_qty = lots.get(code, 0)
                basis[code] = ((old_qty * basis.get(code, 0.0) + change * price)
                               / (old_qty + change))
        transaction_cost = sum(
            abs(change) * 100 * float(execution_marks[code])
            * (buy_fee_rate if change > 0 else sell_fee_rate)
            for code, change in delta.items()
        )
        next_cash -= transaction_cost
        cumulative_costs += transaction_cost
        if next_cash < -0.01:
            raise ValueError(
                f"Kas negatif pada {execution_day.date()}: Rp{next_cash:,.0f}; "
                f"NAV sinyal Rp{signal_nav:,.0f}, NAV eksekusi Rp{nav_before:,.0f}."
            )
        for code in set(lots) - set(next_lots):
            basis.pop(code, None)
        lots, cash = next_lots, next_cash
        nav_after = cash + sum(qty * 100 * float(execution_marks[code]) for code, qty in lots.items())
        if not np.isclose(nav_before - transaction_cost, nav_after, atol=0.05):
            raise AssertionError(f"NAV tidak kontinu pada {execution_day.date()}.")
        events.append({"date": execution_day.date().isoformat(), "nav": nav_after,
                       "cash": cash, "realized": realized,
                       "holdings": dict(lots), "delta_lots": dict(delta),
                       "ga_exposure": solution.exposure,
                       "actual_exposure": 1.0 - cash / nav_after,
                       "transaction_cost": transaction_cost,
                       "turnover": solution.turnover_rate, "trades": len(delta),
                       "fitness": solution.fitness, "sharpe": solution.sharpe_ratio,
                       "mdd": solution.max_drawdown, "correlation": solution.avg_correlation,
                       "fundamental": solution.skor_fundamental,
                       "decisions": decisions,
                       "rebalance_utility": solution.rebalance_utility})
        states.append((execution_day, dict(lots), cash))

    end_index = trading_days.searchsorted(pd.Timestamp(end), side="right") - 1
    final_day = trading_days[end_index]
    marks = closes.loc[final_day, list(lots)]
    if marks.isna().any():
        raise ValueError(f"Harga akhir tidak tersedia pada {final_day.date()}.")
    unrealized = sum(qty * 100 * (float(marks[code]) - basis[code]) for code, qty in lots.items())
    final_nav = cash + sum(qty * 100 * float(marks[code]) for code, qty in lots.items())
    if not np.isclose(final_nav - capital, realized + unrealized - cumulative_costs, atol=0.05):
        raise AssertionError("Realized + unrealized - biaya tidak merekonsiliasi NAV.")

    first_day = schedule[0]
    benchmark_days = pd.DatetimeIndex([first_day, final_day]).unique()
    benchmark = ihsg.reindex(ihsg.index.union(benchmark_days)).sort_index().ffill()
    ihsg_return = float(benchmark[final_day] / benchmark[first_day] - 1)
    portfolio_return = final_nav / capital - 1
    days = trading_days[(trading_days >= first_day) & (trading_days <= final_day)]
    ever_held = sorted({code for _, positions, _ in states for code in positions})
    marks_by_day = closes.loc[days, ever_held].ffill(limit=5)
    nav_history = []
    state_index = 0
    for day in days:
        while (state_index + 1 < len(states) and
               states[state_index + 1][0] <= day):
            state_index += 1
        _, day_lots, day_cash = states[state_index]
        marks = marks_by_day.loc[day, list(day_lots)]
        if marks.isna().any():
            raise ValueError(f"Valuasi harian tidak tersedia pada {day.date()}: "
                             f"{list(marks[marks.isna()].index)}")
        nav_history.append(day_cash + sum(
            qty * 100 * float(marks[code]) for code, qty in day_lots.items()
        ))
    if not np.isclose(nav_history[-1], final_nav, atol=0.05):
        raise AssertionError("NAV harian terakhir tidak cocok dengan NAV akhir.")
    daily_returns = pd.Series(nav_history, index=days).pct_change()
    daily_returns.iloc[0] = nav_history[0] / capital - 1.0
    daily_returns = daily_returns.dropna()
    rf_daily = (1.0 + float(data.risk_free_rate)) ** (1 / 252) - 1
    excess_daily = daily_returns - rf_daily
    volatility = float(daily_returns.std(ddof=0))
    downside = float(np.sqrt(np.mean(np.minimum(excess_daily, 0.0) ** 2)))
    peak = np.maximum.accumulate(np.r_[capital, nav_history])[1:]
    drawdown = 1.0 - np.asarray(nav_history) / peak
    worst_drawdown_index = int(np.argmax(drawdown))
    worst_return_day = daily_returns.idxmin()
    return {
        "start": start.isoformat(), "end": end.isoformat(), "profile": profile,
        "fitness_mode": fitness_mode, "portfolio_return": portfolio_return,
        "net_return": portfolio_return,
        "ihsg_return": ihsg_return, "beat_ihsg": portfolio_return > ihsg_return,
        "final_nav": final_nav, "realized_pnl": realized, "unrealized_pnl": unrealized,
        "transaction_costs": cumulative_costs,
        "cash": cash, "holdings": dict(lots), "events": events,
        "fitness_parameters": fitness_parameters or {},
        "sharpe": (float(excess_daily.mean()) / volatility * np.sqrt(252)
                   if volatility > 1e-12 else None),
        "sortino": (float(excess_daily.mean()) / downside * np.sqrt(252)
                    if downside > 1e-12 else None),
        "max_drawdown": float(drawdown.max()),
        "max_drawdown_date": days[worst_drawdown_index].date().isoformat(),
        "worst_daily_return": float(daily_returns.loc[worst_return_day]),
        "worst_daily_return_date": worst_return_day.date().isoformat(),
        "total_turnover": sum(event["turnover"] for event in events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2018, 3, 14))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2019, 3, 14))
    parser.add_argument("--profile", default="Moderate")
    parser.add_argument("--mode", choices=["baseline", "rebalance_aware"], default="baseline")
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--buy-fee-rate", type=float, default=0.0)
    parser.add_argument("--sell-fee-rate", type=float, default=0.0)
    parser.add_argument("--params-json", default="{}",
                        help="JSON override for GAConfig fitness coefficients")
    parser.add_argument("--suite", action="store_true", help="Run predeclared multi-year walk-forward windows")
    parser.add_argument("--suite-starts", nargs="*", type=date.fromisoformat,
                        default=[date(year, 5, 23) for year in range(2016, 2025)])
    parser.add_argument("--suite-window-months", type=int, default=24)
    parser.add_argument("--suite-profiles", nargs="*", default=["Conservative", "Moderate", "Aggressive"])
    parser.add_argument("--suite-mdd-limits", nargs=3, type=float,
                        default=[0.25, 0.35, 0.50],
                        metavar=("CONSERVATIVE", "MODERATE", "AGGRESSIVE"))
    args = parser.parse_args()
    if args.suite_window_months < 1:
        parser.error("Suite window months harus positif.")
    if not (0 < args.suite_mdd_limits[0] <= args.suite_mdd_limits[1] <= args.suite_mdd_limits[2] <= 1):
        parser.error("Suite MDD limits harus positif, <= 1, dan berurutan.")
    fitness_parameters = json.loads(args.params_json)
    permitted = {"mdd_penalty_override", "correlation_penalty",
                 "fundamental_bonus", "turnover_penalty_beta",
                 "rebalance_utility_weight", "allow_cash", "min_exposure",
                 "max_exposure", "cash_return_weight", "lookthrough_stock_risk"}
    if not isinstance(fitness_parameters, dict) or set(fitness_parameters) - permitted:
        parser.error(f"--params-json only supports {sorted(permitted)}")
    if not args.suite:
        print(json.dumps(run_scenario(args.start, args.end, args.profile, args.mode,
                                      population=args.population, generations=args.generations,
                                      fitness_parameters=fitness_parameters,
                                      buy_fee_rate=args.buy_fee_rate,
                                      sell_fee_rate=args.sell_fee_rate), indent=2))
        return

    results = []
    mdd_limits = dict(zip(("Conservative", "Moderate", "Aggressive"), args.suite_mdd_limits))
    logging.disable(logging.CRITICAL)
    for start in args.suite_starts:
        for profile in args.suite_profiles:
            for mode in ("baseline", "rebalance_aware"):
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        result = run_scenario(
                            start, _add_months(start, args.suite_window_months), profile, mode,
                            population=args.population, generations=args.generations,
                            fitness_parameters=fitness_parameters,
                            buy_fee_rate=args.buy_fee_rate,
                            sell_fee_rate=args.sell_fee_rate,
                        )
                    except Exception as exc:
                        result = {"start": start.isoformat(), "profile": profile,
                                  "fitness_mode": mode, "error": str(exc), "beat_ihsg": False}
                results.append(result)
                result["mdd_limit"] = mdd_limits[profile]
                result["within_mdd_limit"] = (
                    result["max_drawdown"] <= mdd_limits[profile]
                    if "max_drawdown" in result else None
                )
                print(json.dumps({key: result.get(key) for key in
                                  ("start", "end", "profile", "fitness_mode", "portfolio_return",
                                   "ihsg_return", "beat_ihsg", "max_drawdown", "mdd_limit",
                                   "within_mdd_limit", "realized_pnl",
                                   "unrealized_pnl", "error")} ),
                      flush=True)
    for mode in ("baseline", "rebalance_aware"):
        relevant = [result for result in results if result["fitness_mode"] == mode]
        valid = [result for result in relevant if "error" not in result]
        wins = sum(bool(result["beat_ihsg"]) for result in valid)
        within_limit = sum(bool(result["within_mdd_limit"]) for result in valid)
        joint = sum(bool(result["beat_ihsg"] and result["within_mdd_limit"]) for result in valid)
        print(json.dumps({"summary_mode": mode, "wins": wins,
                          "scenarios": len(relevant), "valid": len(valid),
                          "win_rate": wins / len(valid) if valid else None,
                          "within_mdd_limit": within_limit,
                          "joint_successes": joint,
                          "joint_success_rate": joint / len(valid) if valid else None}), flush=True)


if __name__ == "__main__":
    main()
