"""Experimental historical net-return/CVaR/IHSG objective (not a forecast).

Stock and benchmark blocks share dates and horizon. No additive Sharpe,
fundamental, dividend or turnover bonus enters this objective.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .decision_utility import candidate_utility


def align_benchmark_returns(data, marks):
    """Align past-only index Close, rejecting absent/stale observations."""
    if data.return_dates is None or data.price_date is None:
        raise ValueError("Tanggal return dan tanggal keputusan diperlukan untuk IHSG.")
    days = pd.DatetimeIndex(data.return_dates)
    asof = pd.Timestamp(data.price_date)
    if (len(days) != data.returns.shape[1] or not len(days)
            or not days.is_unique or not days.is_monotonic_increasing or days[-1] > asof):
        raise ValueError("Tanggal return tidak sejajar atau melewati tanggal keputusan.")
    marks = marks.copy()
    marks.index = pd.DatetimeIndex(marks.index)
    marks = marks.loc[marks.index <= asof].groupby(level=0).last().sort_index()
    if marks.empty or not np.isfinite(marks).all() or (marks <= 0).any():
        raise ValueError("Harga historis IHSG tidak tersedia/valid.")
    calendar = marks.index.union(days).sort_values()
    aligned = marks.reindex(calendar).ffill().reindex(days)
    seen = pd.Series(marks.index, index=marks.index).reindex(calendar).ffill().reindex(days)
    if aligned.isna().any() or ((days.to_series()-seen) > pd.Timedelta(days=7)).any():
        raise ValueError("IHSG tidak lengkap/stale pada window return saham.")
    # Matches data_loader's initial Close observation (initial return = zero).
    return aligned.pct_change(fill_method=None).fillna(0.).to_numpy(dtype=float)


def load_benchmark_data(data, data_dir):
    """Return a request-local object; never mutate startup's shared data cache."""
    if data.price_date is None or data.return_dates is None or not len(data.return_dates):
        raise ValueError("Riwayat tanggal saham tidak tersedia untuk fitness IHSG.")
    rows = pd.read_parquet(data_dir/'Master_Market_Context_15Tahun.parquet',
        filters=[('Ticker', '==', '^JKSE'),
                 ('Date', '>=', pd.Timestamp(data.return_dates[0])-pd.Timedelta(days=14)),
                 ('Date', '<=', pd.Timestamp(data.price_date))], columns=['Date', 'Close'])
    marks = rows.assign(Date=pd.to_datetime(rows.Date)).groupby('Date').Close.last()
    return replace(data, benchmark_returns=align_benchmark_returns(data, marks))


def fitness_scenarios(data, horizon):
    """Joint non-overlapping completed blocks, aligned backwards from as-of."""
    cached = getattr(data, '_benchmark_fitness_scenarios', None)
    if cached is not None and cached[0] == horizon:
        return cached[1], cached[2]
    history = np.asarray(data.returns, dtype=float)
    benchmark = np.asarray(data.benchmark_returns, dtype=float)
    if (history.ndim != 2 or benchmark.shape != (history.shape[1],)
            or not np.isfinite(history).all() or not np.isfinite(benchmark).all()
            or np.any(history <= -1) or np.any(benchmark <= -1)):
        raise ValueError("Return IHSG wajib tersedia dan sejajar dengan return saham.")
    n = history.shape[1] // horizon
    if n < 3:
        raise ValueError("Fitness IHSG memerlukan minimal tiga horizon historis lengkap.")
    joint = np.vstack([history, benchmark])[:, -n*horizon:]
    samples = np.expm1(np.log1p(joint).reshape(len(joint), n, horizon).sum(axis=2)).T
    data._benchmark_fitness_scenarios = (horizon, samples[:, :-1], samples[:, -1])
    return samples[:, :-1], samples[:, -1]


def benchmark_utility(config, lots, scenarios=None):
    data = config.data
    stocks, benchmark = (fitness_scenarios(data, config.forecast_horizon_days)
                         if scenarios is None else scenarios)
    diagnostics = dict(policy='net_cvar_ihsg_v1', horizon_days=config.forecast_horizon_days,
        risk_profile=config.risk_profile, as_of=str(data.price_date),
        history_start=str(pd.Timestamp(data.return_dates[0]).date()) if data.return_dates is not None else None,
        history_end=str(pd.Timestamp(data.return_dates[-1]).date()) if data.return_dates is not None else None,
        scenario_method='completed_nonoverlapping_blocks', shrinkage_to_zero=0.,
        forecast_status='historical_scenarios_not_validated_forecast',
        benchmark='IHSG Close (price return)', cash_horizon_return=0.)
    diagnostics['held_history'] = data.held_history_diagnostics or {}
    utility, cash, fees = candidate_utility(lots, data.stock_codes, data.prices_per_lot,
        config.old_lots or {}, {**(config.old_prices_per_lot or {}),
                               **dict(zip(data.stock_codes, data.prices_per_lot))},
        float(config.old_cash if config.old_cash is not None else config.budget),
        stocks, 0., config.buy_fee_rate, config.sell_fee_rate,
        config.lambda_cvar*config.profile.lambda_mdd, 0., 0.,
        config.cvar_tail_fraction, len(stocks), benchmark_scenarios=benchmark,
        lambda_benchmark_downside=config.lambda_benchmark_downside*config.profile.lambda_mdd,
        diagnostics=diagnostics)
    return utility, cash, fees, diagnostics
