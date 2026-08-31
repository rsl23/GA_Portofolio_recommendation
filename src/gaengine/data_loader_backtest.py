"""
Data Loader Module - Load Real Stock Data from Local Files
==========================================================

Loads and aligns the local datasets and produces a :class:`MarketData`
object that the GA engine consumes:

    - stock universe     ``Daftar Saham - 20260401.xlsx``
    - daily OHLCV        ``Master_OHLCV_15Tahun.parquet``
    - quarterly ratios   ``Kuartalan_lengkap (1).xlsx``
    - dividend events    ``dividend_events_20_tahun.xlsx``
    - risk-free rate     ``BI-7Day-RR.xlsx``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Paths (data lives in a ``data/`` sub-folder under the project root)
# ----------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT / "data"

STOCKS_FILE   = DATA_DIR / "Daftar Saham - 20260401.xlsx"
PRICE_FILE    = DATA_DIR / "Master_OHLCV_15Tahun.parquet"
FUND_FILE     = DATA_DIR / "Kuartalan_lengkap (1).xlsx"
DIVIDEND_FILE = DATA_DIR / "dividend_events_20_tahun.xlsx"
RISKFREE_FILE = DATA_DIR / "BI-7Day-RR.xlsx"

TRADING_DAYS     = 252
MIN_HISTORY_DAYS = 252 * 5          # require ~5 years of daily returns
_FUND_QUARTERS   = ["Q4 2025", "Q3 2025", "Q2 2025", "Q1 2025"]

# Fundamental metric columns: PER, PBV, ROE, DER, Dividend-Yield.
# ``+1`` means "lower is better", ``-1`` means "higher is better".
_METRIC_DIRECTION = [+1, +1, -1, +1, -1]


# Market-data container diimpor dari file terpisah
from src.gaengine.market_data import MarketData


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------
def _parse_number(val) -> Optional[float]:
    """Parse a number like 1234.5, '1,180.67 B' or '2.50 %' to float/None."""
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    if isinstance(val, str) is False:
        return None
    s = val.strip().replace(",", "").replace(" ", "")
    if s == "":
        return None
    mult = 1.0
    if s.endswith("B"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("T"):
        mult, s = 1e12, s[:-1]
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _avg_ratio(df_fund: pd.DataFrame, ticker: str, ratio: str) -> Optional[float]:
    """Average a ratio over the four most recent quarters (None when absent)."""
    sub = df_fund[(df_fund["Kode"] == ticker) & (df_fund["Key Ratio (Rp)"] == ratio)]
    if sub.empty:
        return None
    vals = []
    for q in _FUND_QUARTERS:
        if q in sub.columns:
            v = _parse_number(sub[q].iloc[0])
            if v is not None and not math.isnan(v):
                vals.append(v)
    return float(np.mean(vals)) if vals else None

# Dividend yields & risk-free rate
def _load_dividend_yields(codes: List[str]) -> Dict[str, float]:
    """Most recent dividend yield per ticker (fraction), default 0."""
    try:
        df = pd.read_excel(DIVIDEND_FILE)
    except Exception:
        return {}
    df = df[df["Ticker"].isin(codes)]
    if df.empty:
        return {}
    dy = pd.to_numeric(df["Dividend_Yield_ExDate_%"], errors="coerce") / 100.0
    df = df.assign(Dy=dy)
    last = df.groupby("Ticker")["Ex_Dividend_Date"].transform("max")
    df = df[df["Ex_Dividend_Date"] == last]
    mean = df.groupby("Ticker")["Dy"].mean()
    return {t: float(mean[t]) if t in mean.index else 0.0 for t in codes}


def _load_risk_free() -> float:
    """Latest BI-7Day-RR figure as an annual fraction (default 6.25%)."""
    try:
        df = pd.read_excel(RISKFREE_FILE, header=None)
        col = df.iloc[:, 2].astype(str)
        has_pct = col[col.str.contains("%", na=False)]
        if len(has_pct):
            raw = str(has_pct.iloc[-1]).replace("%", "").replace(",", ".").strip()
            return float(raw) / 100.0
    except Exception:
        pass
    return 0.0625


# ----------------------------------------------------------------------
# Fundamental metrics (PER, PBV, ROE, DER, Dividend-Yield)
# ----------------------------------------------------------------------
def _build_fundamental_metrics(df_fund: pd.DataFrame, codes: List[str]) -> np.ndarray:
    """Return an (n x 5) array of raw metrics: PER, PBV, ROE, DER, DivYld."""
    div_map = _load_dividend_yields(codes)
    rows = []
    for t in codes:
        per  = _avg_ratio(df_fund, t, "PE Ratio (Quarter)")
        pbv  = _avg_ratio(df_fund, t, "Price to Book Value (Quarter)")
        roe  = _avg_ratio(df_fund, t, "Return on Equity (Quarter)")
        debt = _avg_ratio(df_fund, t, "Total Debt (Quarter)")
        eq   = _avg_ratio(df_fund, t, "Total Equity")
        der = None
        if debt is not None and eq is not None and eq > 0:
            der = debt / eq
        rows.append([per, pbv, roe, der, div_map.get(t)])
    return np.asarray(rows, dtype=float)   # NaN where missing


def _normalize_metrics(metrics: np.ndarray) -> np.ndarray:
    """Min-Max normalise each metric to 0-1 respecting its direction, then
    average per stock into a single composite fundamental score 0-1."""
    n, m = metrics.shape
    composite = np.zeros(n)
    for j in range(m):
        col = metrics[:, j]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            sub = np.full(n, 0.5)
        else:
            lo, hi = float(valid.min()), float(valid.max())
            span = hi - lo
            if span <= 1e-12:
                sub = np.full(n, 1.0)
            else:
                norm = (np.clip(col, lo, hi) - lo) / span
                if _METRIC_DIRECTION[j] > 0:   # lower is better
                    sub = 1.0 - norm
                else:                          # higher is better
                    sub = norm
            sub[np.isnan(col)] = 0.5
        composite += sub
    return composite / m


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def build_market_data(
    min_price: float = 50.0,
    max_stocks: Optional[int] = None,
    min_history_days: int = MIN_HISTORY_DAYS,
) -> MarketData:
    """Build the aligned :class:`MarketData` object consumed by the GA."""
    # 1. stock universe
    df_stocks = pd.read_excel(STOCKS_FILE)
    stock_map = dict(zip(df_stocks["Kode"], df_stocks["Papan Pencatatan"]))

    # 2. price data
    df_price = pd.read_parquet(PRICE_FILE)
    df_price["Ticker"] = df_price["Ticker"].str.replace(".JK", "", regex=False)
    df_price = df_price.sort_values(["Ticker", "Date"])

    valid_codes = set(df_stocks["Kode"].unique())
    df_price = df_price[df_price["Ticker"].isin(valid_codes)]

    latest = df_price.loc[df_price.groupby("Ticker")["Date"].idxmax()]
    close_map = dict(zip(latest["Ticker"], latest["Close"]))

    df_ret = df_price.copy()
    df_ret["Return"] = df_ret.groupby("Ticker")["Close"].pct_change()
    n_obs = df_ret.groupby("Ticker")["Return"].count()
    board_filter = {"Pemantauan Khusus", "Akselerasi", "Ekonomi Baru"}

    def _is_valid(ticker: str) -> bool:
        px = close_map.get(ticker, 0.0)
        if px < min_price:
            return False
        if stock_map.get(ticker, "") in board_filter:
            return False
        if n_obs.get(ticker, 0) < min_history_days:
            return False
        return True

    df_fund = pd.read_excel(FUND_FILE)
    fund_codes = set(df_fund["Kode"].unique())

    candidates = [t for t in (valid_codes & fund_codes) if _is_valid(t)]
    if max_stocks is not None and len(candidates) > max_stocks:
        vol = df_price.groupby("Ticker")["Volume"].mean()
        candidates = vol.loc[candidates].sort_values(ascending=False).head(max_stocks).index.tolist()
    candidates = sorted(candidates)

    # 3. aligned returns matrix (n x T), common window across candidates
    ret_wide = df_ret.pivot(index="Date", columns="Ticker", values="Return").reindex(columns=candidates)
    ret_wide = ret_wide.fillna(0.0)
    returns = ret_wide.to_numpy(dtype=float).T

    # 4. correlation
    correlation = np.corrcoef(returns)
    correlation = np.nan_to_num(correlation, nan=0.0)

    # 5. prices & fundamentals
    prices_per_lot = np.asarray([close_map[t] * 100.0 for t in candidates], dtype=float)
    metrics = _build_fundamental_metrics(df_fund, candidates)
    scores = _normalize_metrics(metrics)
    risk_free = _load_risk_free()

    return MarketData(
        stock_codes=candidates,
        prices_per_lot=prices_per_lot,
        returns=returns,
        correlation=correlation,
        fundamental_scores=scores,
        fundamental_metrics=metrics,
        risk_free_rate=risk_free,
    )
