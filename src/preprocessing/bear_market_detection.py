"""Causal IHSG regime detection; allocation policy lives in another module.

The Hanna turning-point method is ex-post and is not used as a trading signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CONTEXT_FILE = DATA_DIR / "Master_Market_Context_15Tahun.parquet"
STOCK_FILE = DATA_DIR / "Master_OHLCV_15Tahun.parquet"


@dataclass(frozen=True)
class RegimeConfig:
    version: str = "bear-v1"
    peak_sessions: int = 252
    fast_ma: int = 50
    slow_ma: int = 200
    entry_drawdown: float = 0.10
    crash_drawdown: float = 0.20
    exit_drawdown: float = 0.08
    entry_confirmation_days: int = 5
    exit_confirmation_days: int = 10
    slow_ma_slope_days: int = 20
    max_stale_days: int = 10
    depth_scale: float = 0.20
    ma_gap_scale: float = 0.08
    breadth_start: float = 0.50
    breadth_scale: float = 0.30
    vix_start: float = 20.0
    vix_scale: float = 20.0
    fx_scale: float = 0.10
    breadth_min_stocks: int = 50
    weights: dict[str, float] = field(default_factory=lambda: {
        "drawdown": 0.50, "ma_gap": 0.25, "breadth": 0.15, "stress": 0.10,
    })

    def __post_init__(self) -> None:
        if not (0 <= self.exit_drawdown < self.entry_drawdown <
                self.crash_drawdown < 1):
            raise ValueError("Ambang drawdown/hysteresis tidak valid.")
        if min(self.peak_sessions, self.fast_ma, self.slow_ma,
               self.entry_confirmation_days, self.exit_confirmation_days,
               self.slow_ma_slope_days) <= 0:
            raise ValueError("Periode indikator harus positif.")
        if (min(self.depth_scale, self.ma_gap_scale, self.breadth_scale,
                self.vix_scale, self.fx_scale) <= 0 or
            any(weight < 0 for weight in self.weights.values()) or
            sum(self.weights.values()) <= 0):
            raise ValueError("Skala/bobot severity tidak valid.")


CONFIG = RegimeConfig()


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _unknown(as_of: date, reason: str, cfg: RegimeConfig) -> dict:
    return {"status": "unknown", "severity": None, "as_of": None,
            "requested_date": as_of.isoformat(), "reason": reason,
            "config_version": cfg.version, "components": {}, "effective_weights": {},
            "missing_indicators": [], "formula": None, "parameters": asdict(cfg)}


def _clean(values: pd.Series | None, as_of: date, *, allow_zero: bool = False) -> pd.Series:
    if values is None or values.empty:
        return pd.Series(dtype=float)
    series = pd.to_numeric(values, errors="coerce").copy()
    series.index = pd.to_datetime(series.index).normalize()
    series = series[series.index <= pd.Timestamp(as_of)]
    series = series[np.isfinite(series)]
    series = series[series >= 0 if allow_zero else series > 0].dropna().sort_index()
    return series[~series.index.duplicated(keep="last")]


def detect_market_regime(
    ihsg: pd.Series, as_of: date, *, breadth: pd.Series | None = None,
    vix: pd.Series | None = None, usd_idr: pd.Series | None = None,
    config: RegimeConfig = CONFIG,
) -> dict:
    """Replay hysteresis chronologically using observations no later than as_of."""
    price = _clean(ihsg, as_of)
    required = max(config.peak_sessions, config.slow_ma + config.slow_ma_slope_days)
    if len(price) < required:
        return _unknown(as_of, f"Histori IHSG kurang dari {required} sesi.", config)
    if (as_of - price.index[-1].date()).days > config.max_stale_days:
        return _unknown(as_of, "Harga IHSG terakhir terlalu lama.", config)
    peak = price.rolling(config.peak_sessions, min_periods=config.peak_sessions).max()
    drawdown = (1 - price / peak).clip(lower=0)
    fast = price.rolling(config.fast_ma, min_periods=config.fast_ma).mean()
    slow = price.rolling(config.slow_ma, min_periods=config.slow_ma).mean()
    below = fast < slow
    above = fast > slow
    enter = below.rolling(config.entry_confirmation_days).sum().eq(config.entry_confirmation_days)
    leave = above.rolling(config.exit_confirmation_days).sum().eq(config.exit_confirmation_days)
    falling = slow < slow.shift(config.slow_ma_slope_days)
    state = "normal"
    for day in price.index[required - 1:]:
        dd = float(drawdown.loc[day])
        if state == "normal" and (dd >= config.crash_drawdown or
            (dd >= config.entry_drawdown and bool(enter.loc[day]) and bool(falling.loc[day]))):
            state = "bear"
        elif state == "bear" and dd < config.exit_drawdown and bool(leave.loc[day]):
            state = "normal"

    last = price.index[-1]
    dd = float(drawdown.loc[last])
    components = {
        "drawdown": _clip((dd - config.entry_drawdown) / config.depth_scale),
        "ma_gap": _clip(((float(slow.loc[last]) - float(fast.loc[last])) /
                          float(slow.loc[last])) / config.ma_gap_scale),
    }
    missing = []
    breadth_data = _clean(breadth, as_of, allow_zero=True)
    if last in breadth_data.index:
        components["breadth"] = _clip((float(breadth_data.loc[last]) -
                                       config.breadth_start) / config.breadth_scale)
    else:
        missing.append("breadth")
    stress = []
    vix_data = _clean(vix, as_of)
    if not vix_data.empty and vix_data.index[-1] >= last - pd.Timedelta(days=7):
        stress.append(_clip((float(vix_data.iloc[-1]) - config.vix_start) / config.vix_scale))
    else:
        missing.append("vix")
    fx_data = _clean(usd_idr, as_of)
    if len(fx_data) >= 30 and fx_data.index[-1] >= last - pd.Timedelta(days=7):
        recent = fx_data.tail(30)
        stress.append(_clip((float(recent.iloc[-1] / recent.min()) - 1) / config.fx_scale))
    else:
        missing.append("usd_idr")
    if stress:
        components["stress"] = sum(stress) / len(stress)
    weights = {key: config.weights[key] for key in components}
    if "stress" in weights:
        weights["stress"] *= len(stress) / 2
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return _unknown(as_of, "Bobot severity tidak valid.", config)
    effective = {key: weights[key] / total_weight for key in components}
    severity = sum(components[key] * effective[key] for key in components)
    return {
        "status": state, "severity": round(severity, 6),
        "as_of": last.date().isoformat(), "requested_date": as_of.isoformat(),
        "reason": (f"IHSG turun {dd:.1%} dari puncak {config.peak_sessions} sesi; "
                   f"MA50 {'di bawah' if below.loc[last] else 'di atas'} MA200."),
        "config_version": config.version, "components": components,
        "effective_weights": effective, "missing_indicators": missing,
        "formula": ("drawdown=clip((DD-entry_drawdown)/depth_scale); "
                    "ma_gap=clip(((MA200-MA50)/MA200)/ma_gap_scale); "
                    "breadth=clip((below_MA200-breadth_start)/breadth_scale); "
                    "stress=mean(clip((VIX-vix_start)/vix_scale), "
                    "clip(USDIDR_30d_change/fx_scale)); "
                    "severity=sum(component*effective_weight)"),
        "parameters": asdict(config),
    }


@lru_cache(maxsize=1)
def _historical_context() -> pd.DataFrame:
    return pd.read_parquet(CONTEXT_FILE, columns=["Date", "Ticker", "Close"])


def _series(frame: pd.DataFrame, ticker: str) -> pd.Series:
    rows = frame[frame["Ticker"] == ticker]
    return pd.Series(rows["Close"].to_numpy(), index=pd.to_datetime(rows["Date"]))


def load_regime_data(as_of: date, backtest: bool) -> dict[str, pd.Series]:
    if backtest:
        frame = _historical_context()
        # Replay hysteresis from the beginning of available IHSG history.
        # A sliding 850-day window can forget a still-active bear regime.
        selected = frame[frame["Date"] <= pd.Timestamp(as_of)]
        result = {"ihsg": _series(selected, "^JKSE"),
                  "vix": _series(selected, "^VIX"),
                  "usd_idr": _series(selected, "USDIDR=X")}
        stocks = pd.read_parquet(
            STOCK_FILE, columns=["Date", "Ticker", "Close"],
            filters=[("Date", ">=", pd.Timestamp(as_of - timedelta(days=420))),
                     ("Date", "<=", pd.Timestamp(as_of))],
        )
        closes = stocks.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
        ma = closes.rolling(CONFIG.slow_ma, min_periods=CONFIG.slow_ma).mean()
        eligible = ma.notna() & closes.notna()
        count = eligible.sum(axis=1)
        result["breadth"] = ((closes < ma) & eligible).sum(axis=1).div(
            count.where(count >= CONFIG.breadth_min_stocks))
        return result
    end = as_of + timedelta(days=1)  # yfinance end is exclusive
    result = {}
    for key, ticker in (("ihsg", "^JKSE"), ("vix", "^VIX"), ("usd_idr", "USDIDR=X")):
        try:
            start = "1990-01-01" if key == "ihsg" else (as_of - timedelta(days=850)).isoformat()
            frame = yf.download(ticker, start=start,
                                end=end.isoformat(), auto_adjust=False, progress=False)
            close = frame["Close"]
            result[key] = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        except Exception:
            result[key] = pd.Series(dtype=float)
    result["breadth"] = pd.Series(dtype=float)
    return result


def assess_market(as_of: date, backtest: bool) -> dict:
    try:
        return detect_market_regime(**load_regime_data(as_of, backtest), as_of=as_of)
    except Exception as exc:
        return _unknown(as_of, f"Data pasar tidak tersedia: {exc}", CONFIG)
