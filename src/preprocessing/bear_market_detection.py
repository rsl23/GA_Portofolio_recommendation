"""Causal IHSG regime detection; allocation policy lives in another module.

The Hanna turning-point method is ex-post and is not used as a trading signal.
Live and backtest share the same point-in-time stock-breadth calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

from src.preprocessing.market_breadth import load_market_breadth
from src.preprocessing.live_regime_cache import live_assessment

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CONTEXT_FILE = DATA_DIR / "Master_Market_Context_15Tahun.parquet"


@dataclass(frozen=True)
class RegimeConfig:
    version: str = "bear-v4"
    peak_sessions: int = 252
    fast_ma: int = 50
    slow_ma: int = 200
    entry_drawdown: float = 0.10
    sustained_drawdown: float = 0.15
    sustained_confirmation_days: int = 2
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
    breadth_min_stocks: int = 50
    breadth_min_coverage: float = 0.70
    vix_start: float = 20.0
    vix_scale: float = 20.0
    fx_scale: float = 0.10
    weights: dict[str, float] = field(default_factory=lambda: {
        "drawdown": .50, "ma_gap": .25, "breadth": .15, "stress": .10,
    })

    def __post_init__(self) -> None:
        if not (0 <= self.exit_drawdown < self.entry_drawdown <
                self.sustained_drawdown <
                self.crash_drawdown < 1):
            raise ValueError("Ambang drawdown/hysteresis tidak valid.")
        if min(self.peak_sessions, self.fast_ma, self.slow_ma,
               self.entry_confirmation_days, self.sustained_confirmation_days,
               self.exit_confirmation_days,
               self.slow_ma_slope_days) <= 0:
            raise ValueError("Periode indikator harus positif.")
        if (self.breadth_min_stocks < 1 or not 0 < self.breadth_min_coverage <= 1
                or not 0 <= self.breadth_start < 1):
            raise ValueError("Ambang cakupan breadth tidak valid.")
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
            "missing_indicators": [], "breadth_details": None,
            "formula": None, "parameters": asdict(cfg)}


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
    ihsg: pd.Series, as_of: date, *,
    vix: pd.Series | None = None, usd_idr: pd.Series | None = None,
    breadth: pd.Series | None = None, breadth_details: dict | None = None,
    config: RegimeConfig = CONFIG,
    return_history: bool = False,
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
    sustained = drawdown.ge(config.sustained_drawdown).rolling(
        config.sustained_confirmation_days
    ).sum().eq(config.sustained_confirmation_days)
    leave = above.rolling(config.exit_confirmation_days).sum().eq(config.exit_confirmation_days)
    falling = slow < slow.shift(config.slow_ma_slope_days)
    state = "normal"
    entry_reason = None
    history = []
    for day in price.index[required - 1:]:
        dd = float(drawdown.loc[day])
        if state == "normal":
            if dd >= config.crash_drawdown:
                state = "bear"
                entry_reason = f"drawdown mencapai {config.crash_drawdown:.0%}"
            elif bool(sustained.loc[day]):
                state = "bear"
                entry_reason = (f"drawdown minimal {config.sustained_drawdown:.0%} "
                                f"selama {config.sustained_confirmation_days} sesi")
            elif dd >= config.entry_drawdown and bool(enter.loc[day]) and bool(falling.loc[day]):
                state = "bear"
                entry_reason = "MA50 di bawah MA200 dan MA200 menurun"
        elif state == "bear" and dd < config.exit_drawdown and bool(leave.loc[day]):
            state = "normal"
            entry_reason = None
        if return_history:
            history.append({"date": day.date().isoformat(), "status": state})

    last = price.index[-1]
    dd = float(drawdown.loc[last])
    components = {
        "drawdown": _clip((dd - config.entry_drawdown) / config.depth_scale),
        "ma_gap": _clip(((float(slow.loc[last]) - float(fast.loc[last])) /
                          float(slow.loc[last])) / config.ma_gap_scale),
    }
    missing = []
    breadth_data = _clean(breadth, last.date(), allow_zero=True)
    if last in breadth_data.index and breadth_data.loc[last] <= 1:
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
        **({"history": history} if return_history else {}),
        "status": state, "severity": round(severity, 6),
        "as_of": last.date().isoformat(), "requested_date": as_of.isoformat(),
        "reason": (f"IHSG turun {dd:.1%} dari puncak {config.peak_sessions} sesi; "
                   f"MA50 {'di bawah' if below.loc[last] else 'di atas'} MA200."
                   + (f" Status bear dipicu oleh {entry_reason}." if entry_reason else "")),
        "config_version": config.version, "components": components,
        "effective_weights": effective, "missing_indicators": missing,
        "breadth_details": breadth_details,
        "formula": ("drawdown=clip((DD-entry_drawdown)/depth_scale); "
                    "ma_gap=clip(((MA200-MA50)/MA200)/ma_gap_scale); "
                    "breadth=clip((fraction_below_MA200-breadth_start)/breadth_scale); "
                    "stress=mean(clip((VIX-vix_start)/vix_scale), "
                    "clip(USDIDR_30d_change/fx_scale)); "
                    "severity=sum(component*effective_weight)"),
        "parameters": asdict(config),
    }


@lru_cache(maxsize=1)
def _historical_context() -> pd.DataFrame:
    return pd.read_parquet(CONTEXT_FILE, columns=["Date", "Ticker", "Close", "Volume"])


def _series(frame: pd.DataFrame, ticker: str) -> pd.Series:
    rows = frame[frame["Ticker"] == ticker]
    if ticker == "^JKSE":
        from src.preprocessing.ihsg_sessions import validated_ihsg_rows
        rows = validated_ihsg_rows(rows)
    return pd.Series(rows["Close"].to_numpy(), index=pd.to_datetime(rows["Date"]))


def _include_breadth(result: dict, as_of: date, backtest: bool) -> dict:
    sessions = _clean(result["ihsg"], as_of).index
    result["breadth"], result["breadth_details"] = load_market_breadth(
        sessions, backtest, ma_sessions=CONFIG.slow_ma,
        min_stocks=CONFIG.breadth_min_stocks, min_coverage=CONFIG.breadth_min_coverage)
    return result


def load_regime_data(as_of: date, backtest: bool) -> dict[str, pd.Series]:
    if backtest:
        frame = _historical_context()
        # Replay hysteresis from the beginning of available IHSG history.
        # A sliding 850-day window can forget a still-active bear regime.
        selected = frame[frame["Date"] <= pd.Timestamp(as_of)]
        result = {"ihsg": _series(selected, "^JKSE"),
                  "vix": _series(selected, "^VIX"),
                  "usd_idr": _series(selected, "USDIDR=X")}
        return _include_breadth(result, as_of, backtest)
    end = as_of + timedelta(days=1)  # yfinance end is exclusive
    result = {}
    for key, ticker in (("ihsg", "^JKSE"), ("vix", "^VIX"), ("usd_idr", "USDIDR=X")):
        try:
            start = "1990-01-01" if key == "ihsg" else (as_of - timedelta(days=850)).isoformat()
            frame = yf.download(ticker, start=start,
                                end=end.isoformat(), auto_adjust=False, progress=False)
            close = frame["Close"]
            close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            if key == "ihsg" and "Volume" in frame:
                volume = frame["Volume"]
                volume = volume.iloc[:, 0] if isinstance(volume, pd.DataFrame) else volume
                close = close[pd.to_numeric(volume, errors="coerce") > 0]
            result[key] = close
        except Exception:
            result[key] = pd.Series(dtype=float)
    return _include_breadth(result, as_of, backtest)


def _assess_market_uncached(as_of: date, backtest: bool) -> dict:
    try:
        inputs = load_regime_data(as_of, backtest)
        result = detect_market_regime(**inputs, as_of=as_of, config=CONFIG)
        if not backtest and result["as_of"] is not None:
            session = date.fromisoformat(result["as_of"])
            if session < as_of:
                # Keep the stale-data rejection from the first call. For valid
                # observations, ancillary indicators may not outrun the IHSG mark.
                result = detect_market_regime(**inputs, as_of=session, config=CONFIG)
                result["requested_date"] = as_of.isoformat()
        return result
    except Exception as exc:
        return _unknown(as_of, f"Data pasar tidak tersedia: {exc}", CONFIG)


def assess_market(as_of: date, backtest: bool) -> dict:
    if backtest:
        return _assess_market_uncached(as_of, backtest)
    parameters = {**asdict(CONFIG), "live_alignment_policy": "common-session-v1"}
    result = live_assessment(as_of, parameters, lambda: _assess_market_uncached(as_of, False))
    if result["as_of"] is not None:
        session = date.fromisoformat(result["as_of"])
        if session < as_of:
            # Publish the exact same observation under its actual session key.
            # Generate can then validate that preview without a fresh download
            # or a second observation set. Do this outside the cache's lock.
            canonical = {**result, "requested_date": session.isoformat()}
            return live_assessment(session, parameters, lambda: canonical)
    return result
