"""Past-only, multi-episode market stress evidence for GA risk scoring.

Observed prices are never imputed into portfolio accounting. Missing historical
evidence gets a benchmark-like scoring prior plus a small explicit uncertainty
penalty. These stress scenarios are hypothetical, not a transaction replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressConfig:
    version: str = "market-stress-v2-rolling"
    lookback_years: int = 10
    peak_sessions: int = 252
    entry_drawdown: float = .10
    recovery_drawdown: float = .08
    min_event_observations: int = 3
    min_price_coverage: float = .95
    min_traded_fraction: float = .80
    max_untraded_run: int = 5
    archive_gap_warning_days: int = 7

    def __post_init__(self):
        if isinstance(self.lookback_years, bool) or not isinstance(self.lookback_years, int) or self.lookback_years < 1:
            raise ValueError("lookback_years harus bilangan bulat positif.")
        if not 0 <= self.recovery_drawdown < self.entry_drawdown < 1:
            raise ValueError("Ambang stress/recovery tidak valid.")
        if min(self.peak_sessions, self.min_event_observations, self.max_untraded_run) < 1:
            raise ValueError("Window stress harus positif.")
        if not (0 < self.min_price_coverage <= 1 and 0 < self.min_traded_fraction <= 1):
            raise ValueError("Ambang kelengkapan stress harus berada di (0, 1].")


CONFIG = StressConfig()


def stress_window_start(as_of: date, years: int) -> date:
    """Calendar-year horizon; DateOffset handles leap-day cutoffs safely."""
    return (pd.Timestamp(as_of) - pd.DateOffset(years=years)).date()


def detect_stress_events(ihsg: pd.Series, as_of: date,
                         config: StressConfig = CONFIG) -> list[dict]:
    """Find distinct peak-to-known-trough episodes using only data <= as_of.

    An ongoing episode uses its trough observed so far. Recovering to within
    8% of its fixed starting peak ends the episode. Re-entries sharing a peak
    are merged so one crisis cannot be counted several times. Earlier history
    only supplies peak/state warmup; scoring requires the entire episode peak
    to fall within the configured rolling lookback ending at the decision date.
    """
    prices = ihsg.copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = pd.to_numeric(prices, errors="coerce")
    prices = prices[(prices.index <= pd.Timestamp(as_of)) & np.isfinite(prices) & (prices > 0)]
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    events: dict[pd.Timestamp, dict] = {}
    active = None
    for i in range(config.peak_sessions - 1, len(prices)):
        day = prices.index[i]
        value = float(prices.iloc[i])
        if active is None:
            window = prices.iloc[i - config.peak_sessions + 1:i + 1]
            peak = float(window.max())
            peak_day = window[window.eq(peak)].index[-1]
            if 1 - value / peak < config.entry_drawdown:
                continue
            active = {"start": peak_day, "peak": peak, "trigger_date": day,
                      "trough": day, "trough_price": value, "closed": False}
        if value < active["trough_price"]:
            active.update(trough=day, trough_price=value)
        recovered = value >= active["peak"] * (1 - config.recovery_drawdown)
        active["closed"] = recovered
        prior = events.get(active["start"])
        if prior is None or active["trough_price"] < prior["trough_price"]:
            events[active["start"]] = dict(active)
        elif recovered:
            prior["closed"] = True
        if prior is not None:
            events[active["start"]]["closed"] = recovered
        if recovered:
            active = None
    result = []
    window_start = pd.Timestamp(stress_window_start(as_of, config.lookback_years))
    last_end = None
    for event in sorted(events.values(), key=lambda e: e["start"]):
        start, end = event["start"], event["trough"]
        if start < window_start:
            continue
        count = int(((prices.index >= start) & (prices.index <= end)).sum())
        if count < config.min_event_observations:
            continue
        # A later rolling peak can sit inside a prior decline. It must not
        # make the same observations a second equally weighted crash.
        if last_end is not None and start <= last_end:
            continue
        market_return = event["trough_price"] / event["peak"] - 1
        result.append({
            "id": f"IHSG_{start.date()}_{end.date()}",
            "start": start.date().isoformat(), "end": end.date().isoformat(),
            "trigger_date": event["trigger_date"].date().isoformat(),
            "market_return": market_return, "observations": count,
            "closed": event["closed"],
        })
        last_end = end
    return result


@dataclass
class MarketStressData:
    stock_codes: list[str]
    as_of: date
    history_end: date | None
    events: list[dict]
    stock_returns: np.ndarray  # (events, stocks); NaN means unproven, never zero return
    missing_reasons: list[list[str | None]]
    status: str = "available"
    reason: str | None = None
    version: str = CONFIG.version
    lookback_years: int = CONFIG.lookback_years

    def __post_init__(self):
        self.stock_returns = np.asarray(self.stock_returns, dtype=float)
        if self.stock_returns.shape != (len(self.events), len(self.stock_codes)):
            raise ValueError("Dimensi bukti stress tidak cocok dengan event dan saham.")
        self.known = np.isfinite(self.stock_returns)
        self.market_returns = np.asarray([e["market_return"] for e in self.events], dtype=float)
        if (not np.isfinite(self.market_returns).all() or np.any(self.market_returns >= 0)
                or np.any(self.market_returns <= -1)):
            raise ValueError("Event stress harus memiliki return IHSG negatif yang valid.")
        # This prior is only a scoring assumption, never a reported stock return.
        self.scoring_returns = np.where(self.known, self.stock_returns, self.market_returns[:, None])

    @property
    def window_start(self) -> date:
        return stress_window_start(self.as_of, self.lookback_years)

    def evaluate(self, weights: np.ndarray) -> dict:
        """Score the basket together: cash has zero return in every scenario."""
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (len(self.stock_codes),) or not np.isfinite(weights).all()
                or np.any(weights < 0) or weights.sum() > 1 + 1e-6):
            raise ValueError("Bobot stress harus fraksi NAV non-negatif, jumlah <= 1.")
        if not self.events:
            return {"risk": None, "missing_exposure": float(weights.sum()),
                    "event_returns_for_scoring": [], "evidence_coverage": 0.0}
        scenario_returns = self.scoring_returns @ weights
        # Half absolute loss, half excess loss versus IHSG. Return/gains above
        # zero do not earn an unlimited bonus that could dominate fitness.
        absolute_loss = np.maximum(-scenario_returns, 0)
        relative_loss = np.maximum(self.market_returns - scenario_returns, 0)
        risk = float(np.mean(.5 * absolute_loss + .5 * relative_loss))
        missing = float(np.mean((~self.known) @ weights))
        invested = float(weights.sum())
        return {"risk": risk, "missing_exposure": missing,
                "event_returns_for_scoring": scenario_returns.tolist(),
                "evidence_coverage": 1 - missing / invested if invested > 1e-12 else 1.0}

    def summary(self) -> dict:
        per_stock = {}
        for j, code in enumerate(self.stock_codes):
            observed = [
                {"event_id": event["id"],
                 "return": float(self.stock_returns[i, j]) if self.known[i, j] else None,
                 "market_return": event["market_return"],
                 "missing_reason": self.missing_reasons[i][j]}
                for i, event in enumerate(self.events)
            ]
            per_stock[code] = {"observed_events": int(self.known[:, j].sum()),
                               "total_events": len(self.events), "observations": observed,
                               "risk_with_explicit_prior": float(np.mean(
                                   .5 * np.maximum(-self.scoring_returns[:, j], 0)
                                   + .5 * np.maximum(self.market_returns - self.scoring_returns[:, j], 0)
                               )) if self.events else None,
                               "missing_fraction": float((~self.known[:, j]).mean()) if self.events else 1.0}
        return {"config_version": self.version, "as_of": self.as_of.isoformat(),
                "lookback_years": self.lookback_years,
                "window_start": self.window_start.isoformat(),
                "window_end": self.as_of.isoformat(),
                "window_policy": "episode peak must be within the rolling horizon; older data only warms up peak/state",
                "history_end": self.history_end.isoformat() if self.history_end else None,
                "status": self.status, "reason": self.reason, "events": self.events,
                "event_weighting": "equal_per_nonoverlapping_episode",
                "missing_policy": "benchmark-like scoring prior plus explicit uncertainty penalty; observed return stays null",
                "stocks": per_stock}


def _longest_false_run(values: np.ndarray) -> int:
    missing = ~np.asarray(values, dtype=bool)
    padded = np.r_[False, missing, False].astype(int)
    starts = np.where(np.diff(padded) == 1)[0]
    ends = np.where(np.diff(padded) == -1)[0]
    return int((ends - starts).max()) if len(starts) else 0


def build_stress_evidence(ihsg: pd.Series, closes: pd.DataFrame,
                          volumes: pd.DataFrame, stock_codes: list[str], as_of: date,
                          config: StressConfig = CONFIG) -> MarketStressData:
    # Cut all sources first. A future append must not change a historical score.
    cutoff = pd.Timestamp(as_of)
    closes = closes.copy()
    volumes = volumes.copy()
    for frame in (closes, volumes):
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    ihsg = ihsg.copy()
    ihsg.index = pd.to_datetime(ihsg.index).tz_localize(None).normalize()
    ihsg = ihsg.loc[ihsg.index <= cutoff].sort_index().dropna()
    closes = closes.loc[closes.index <= cutoff].sort_index()
    volumes = volumes.loc[volumes.index <= cutoff].sort_index()
    if ihsg.empty or closes.empty:
        return MarketStressData(stock_codes, as_of, None, [], np.empty((0, len(stock_codes))), [],
                                "unavailable", "Histori IHSG atau saham tidak tersedia sebelum tanggal keputusan.",
                                config.version, config.lookback_years)
    history_end = min(ihsg.index.max(), closes.index.max())
    ihsg = ihsg.loc[ihsg.index <= history_end]
    # Anchor the rolling horizon to the decision, never to a stale archive end.
    events = detect_stress_events(ihsg, as_of, config)
    returns = np.full((len(events), len(stock_codes)), np.nan)
    reasons = [[None] * len(stock_codes) for _ in events]
    for i, event in enumerate(events):
        days = ihsg.loc[event["start"]:event["end"]].index
        prices = closes.reindex(index=days, columns=stock_codes)
        volume = volumes.reindex(index=days, columns=stock_codes)
        for j, code in enumerate(stock_codes):
            p = pd.to_numeric(prices[code], errors="coerce")
            v = pd.to_numeric(volume[code], errors="coerce")
            valid = np.isfinite(p) & (p > 0)
            traded = valid & np.isfinite(v) & (v > 0)
            if not valid.iloc[0] or not valid.iloc[-1]:
                reasons[i][j] = "Harga awal/akhir event tidak tersedia; histori belum lengkap."
            elif valid.mean() < config.min_price_coverage:
                reasons[i][j] = "Kelengkapan harga selama event kurang dari batas minimum."
            elif (not traded.iloc[0] or not traded.iloc[-1] or
                  traded.mean() < config.min_traded_fraction or
                  _longest_false_run(traded.to_numpy()) > config.max_untraded_run):
                reasons[i][j] = "Bukti perdagangan tidak memadai; volume nol/hilang atau kemungkinan suspensi."
            else:
                returns[i, j] = float(p.iloc[-1] / p.iloc[0] - 1)
    partial = (as_of - history_end.date()).days > config.archive_gap_warning_days
    status = "archive_partial" if partial else "available"
    reason = f"Arsip bersama tersedia sampai {history_end.date()}; event setelahnya belum dinilai." if partial else None
    if not events:
        status = "no_events"
        no_events_reason = f"Belum ada event penurunan IHSG yang memenuhi ambang dalam {config.lookback_years} tahun terakhir pada histori tersedia."
        reason = f"{no_events_reason} {reason}" if reason else no_events_reason
    return MarketStressData(stock_codes, as_of, history_end.date(), events, returns, reasons,
                            status, reason, config.version, config.lookback_years)


@lru_cache(maxsize=1)
def _read_archives(price_path: str, price_mtime: int, context_path: str,
                   context_mtime: int) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    stocks = pd.read_parquet(price_path, columns=["Date", "Ticker", "Close", "Volume"])
    stocks["Date"] = pd.to_datetime(stocks["Date"])
    stocks["Ticker"] = stocks["Ticker"].str.removesuffix(".JK")
    # The market calendar must not include carried-forward holiday bars.
    traded_days = set(stocks.loc[pd.to_numeric(stocks["Volume"], errors="coerce") > 0, "Date"])
    stocks = stocks[stocks["Date"].isin(traded_days)]
    closes = stocks.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last").sort_index()
    volumes = stocks.pivot_table(index="Date", columns="Ticker", values="Volume", aggfunc="last").sort_index()
    context = pd.read_parquet(context_path, columns=["Date", "Ticker", "Close"],
                              filters=[("Ticker", "==", "^JKSE")])
    index = pd.Series(context["Close"].to_numpy(), index=pd.to_datetime(context["Date"])).sort_index()
    return index[index.index.isin(traded_days)], closes, volumes


def load_market_stress(stock_codes: list[str], as_of: date, data_dir: Path,
                       config: StressConfig = CONFIG) -> MarketStressData:
    try:
        prices = data_dir / "Master_OHLCV_15Tahun.parquet"
        context = data_dir / "Master_Market_Context_15Tahun.parquet"
        ihsg, closes, volumes = _read_archives(str(prices), prices.stat().st_mtime_ns,
                                             str(context), context.stat().st_mtime_ns)
        # Limit columns before copying frames for each decision; cached archives
        # remain immutable and all date filtering still happens before detection.
        return build_stress_evidence(ihsg, closes.reindex(columns=stock_codes),
                                     volumes.reindex(columns=stock_codes), stock_codes, as_of, config)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        return MarketStressData(stock_codes, as_of, None, [], np.empty((0, len(stock_codes))), [],
                                "unavailable", f"Arsip stress tidak dapat dibaca: {exc}",
                                config.version, config.lookback_years)
