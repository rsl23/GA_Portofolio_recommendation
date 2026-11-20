"""Validate historical index sessions without treating missing index volume as a holiday.

Zero-volume dates require contemporaneous trading evidence from the local equity
archive. No interpolation, network requests, or observations after that date.
"""
from functools import lru_cache
import logging
from pathlib import Path

import numpy as np
import pandas as pd

STOCK_FILE = Path(__file__).resolve().parents[2] / "data" / "Master_OHLCV_15Tahun.parquet"
MIN_TRADED_STOCKS = 50
logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _confirmed(path, modified_ns, size, dates, minimum):
    rows = pd.read_parquet(path, columns=["Date", "Ticker", "Volume"],
                           filters=[("Date", "in", [pd.Timestamp(day) for day in dates])])
    if not {"Date", "Ticker", "Volume"}.issubset(rows.columns):
        return frozenset()
    traded = rows.loc[pd.to_numeric(rows["Volume"], errors="coerce").gt(0)].copy()
    traded["Date"] = pd.to_datetime(traded["Date"]).dt.normalize()
    counts = traded.groupby("Date")["Ticker"].nunique()
    return frozenset(counts.index[counts >= minimum])


def confirmed_stock_sessions(dates, stock_file=None, minimum=MIN_TRADED_STOCKS):
    requested = tuple(sorted({pd.Timestamp(day).normalize().isoformat() for day in dates}))
    if not requested:
        return set()
    path = Path(stock_file) if stock_file is not None else STOCK_FILE
    try:
        identity = path.stat()
        return set(_confirmed(str(path), identity.st_mtime_ns, identity.st_size, requested, minimum))
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Kalender saham tidak dapat mengonfirmasi sesi IHSG: %s", exc)
        return set()


def validated_ihsg_rows(frame, stock_file=None):
    if frame.empty:
        return frame.copy()
    rows = frame.copy()
    rows["Date"] = pd.to_datetime(rows["Date"]).dt.normalize()
    close = pd.to_numeric(rows["Close"], errors="coerce")
    valid_price = np.isfinite(close) & close.gt(0)
    if "Volume" not in rows:
        return rows.loc[valid_price].sort_values("Date").drop_duplicates("Date", keep="last")
    observed_volume = pd.to_numeric(rows["Volume"], errors="coerce").gt(0)
    suspect_dates = rows.loc[valid_price & ~observed_volume, "Date"]
    confirmed = confirmed_stock_sessions(suspect_dates, stock_file)
    accepted = valid_price & (observed_volume | rows["Date"].isin(confirmed))
    return rows.loc[accepted].sort_values("Date").drop_duplicates("Date", keep="last")


def portfolio_session_calendar(index_dates, nav_dates, stock_file=None):
    """Also catch sessions missing from BOTH the index and the held-stock series."""
    observed = {pd.Timestamp(day).normalize() for day in index_dates}
    nav_days = {pd.Timestamp(day).normalize() for day in nav_dates}
    if not nav_days:
        return observed or None
    candidates = nav_days | set(pd.bdate_range(min(nav_days), max(nav_days)))
    observed.update(confirmed_stock_sessions(candidates - observed, stock_file))
    return observed or None
