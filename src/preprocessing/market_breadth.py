"""Point-in-time stock breadth shared by live and historical assessments.

The universe is all listed equities old enough for the MA window, independent
of GA screening. No prices are forward-filled. MA uses observed raw closes on
IHSG trading sessions; the final session also requires positive stock volume.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from threading import Lock
import time
from uuid import uuid4

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
STOCK_FILE = DATA_DIR / "Master_OHLCV_15Tahun.parquet"
CACHE_DIR = DATA_DIR / "cache" / "market_breadth"
_refresh_lock = Lock()
logger = logging.getLogger(__name__)


def _ticker(value) -> str:
    return str(value).strip().upper().removesuffix(".JK")


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=["Date", "Ticker", "Close", "Volume"])


def breadth_universe(listings: pd.DataFrame, sessions: pd.DatetimeIndex) -> list[str]:
    """Exclude future listings/delistings and recent relistings as of the window."""
    if len(sessions) == 0:
        return []
    start, end = sessions[0], sessions[-1]
    listing = pd.to_datetime(listings["listing_date"], errors="coerce")
    delisting = pd.to_datetime(listings["delisting_date"], errors="coerce")
    relisting = pd.to_datetime(listings["relisting_date"], errors="coerce")
    uninterrupted = delisting.isna() | (delisting > end) | ((relisting > delisting) & (relisting <= start))
    eligible = listing.notna() & (listing <= start) & uninterrupted
    return sorted({_ticker(t) for t in listings.loc[eligible, "ticker"] if pd.notna(t)})


def _listing_history() -> pd.DataFrame:
    # Lazy imports keep the numerical functions usable without a configured DB.
    from src.backend.models.database import SessionLocal
    from src.backend.models.stock_universe import StockUniverse
    with SessionLocal() as db:
        rows = db.query(StockUniverse.ticker, StockUniverse.listing_date,
                        StockUniverse.delisting_date, StockUniverse.relisting_date).all()
    return pd.DataFrame(rows, columns=["ticker", "listing_date", "delisting_date", "relisting_date"])


def calculate_breadth(history: pd.DataFrame, sessions: pd.DatetimeIndex,
                     universe: list[str], *, ma_sessions: int = 200,
                     min_stocks: int = 50, min_coverage: float = .70) -> tuple[pd.Series, dict]:
    """One-session fraction below MA; missing observations never count as above MA."""
    sessions = pd.DatetimeIndex(sessions).tz_localize(None).normalize().unique().sort_values()
    sessions = sessions[-ma_sessions:]
    universe = sorted(set(map(_ticker, universe)))
    details = dict(status="unavailable", as_of=sessions[-1].date().isoformat() if len(sessions) else None,
        universe_count=len(universe), eligible_count=0, below_ma_count=None,
        coverage=0., fraction_below_ma=None, ma_sessions=ma_sessions,
        min_stocks=min_stocks, min_coverage=min_coverage,
        universe_rule="listed_before_ma_window; no_delisting_gap; independent_of_GA",
        price_rule="raw_Close; IHSG_sessions; no_forward_fill; positive_final_volume",
        reason=None)
    empty = pd.Series(dtype=float)
    if len(sessions) < ma_sessions or not universe:
        details["reason"] = "Sesi IHSG atau daftar emiten belum cukup untuk MA."
        return empty, details
    data = history.copy()
    if not data.empty:
        data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None).dt.normalize()
        data["Ticker"] = data["Ticker"].map(_ticker)
        data = data[data["Date"].isin(sessions) & data["Ticker"].isin(universe)]
        for column in ("Close", "Volume"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.drop_duplicates(["Date", "Ticker"], keep="last")
    if data.empty:
        details["reason"] = "Histori saham tidak tersedia pada sesi penilaian."
        return empty, details
    close = data.pivot(index="Date", columns="Ticker", values="Close").reindex(
        index=sessions, columns=universe)
    close = close.where(np.isfinite(close) & (close > 0))
    volume = data.pivot(index="Date", columns="Ticker", values="Volume").reindex(
        index=sessions, columns=universe)
    complete = close.notna().all(axis=0)
    traded = np.isfinite(volume.iloc[-1]) & (volume.iloc[-1] > 0)
    eligible = complete & traded
    count = int(eligible.sum())
    details.update(eligible_count=count, coverage=count / len(universe),
        missing_history_count=int((~complete).sum()),
        untraded_count=int((complete & ~traded).sum()))
    if count < min_stocks or details["coverage"] < min_coverage:
        details["reason"] = (f"Cakupan breadth belum cukup: {count}/{len(universe)} saham; "
                             f"minimum {min_stocks} saham dan {min_coverage:.0%} cakupan.")
        return empty, details
    mean = close.mean(axis=0)
    below = int(((close.iloc[-1] < mean) & eligible).sum())
    fraction = below / count
    details.update(status="available", below_ma_count=below,
                   fraction_below_ma=fraction, reason=None)
    return pd.Series([fraction], index=[sessions[-1]]), details


def _download_history(universe: list[str], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    records = []
    for offset in range(0, len(universe), 80):
        codes = universe[offset:offset + 80]
        try:
            frame = yf.download([c + ".JK" for c in codes],
                start=sessions[0].date().isoformat(),
                end=(sessions[-1] + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=False, actions=False, group_by="ticker", threads=8,
                progress=False, timeout=20)
            for code in codes:
                symbol = code + ".JK"
                if isinstance(frame.columns, pd.MultiIndex):
                    if symbol in frame.columns.get_level_values(0):
                        stock = frame[symbol]
                    elif symbol in frame.columns.get_level_values(1):
                        stock = frame.xs(symbol, level=1, axis=1)
                    else:
                        continue
                elif len(codes) == 1:
                    stock = frame
                else:
                    continue
                if not {"Close", "Volume"}.issubset(stock.columns):
                    continue
                rows = stock[["Close", "Volume"]].copy()
                rows.index = pd.to_datetime(rows.index).tz_localize(None).normalize()
                rows = rows.rename_axis("Date").reset_index()
                rows["Ticker"] = code
                records.append(rows.dropna(subset=["Close"]))
        except Exception:
            logger.warning("Unduhan breadth batch %d tidak tersedia.", offset // 80 + 1)
    return pd.concat(records, ignore_index=True) if records else _empty_history()


def _live_stock_history(universe: list[str], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Disk cache survives reloads. Partial downloads retry after 15 minutes."""
    key = hashlib.sha256(("|".join(universe) + str(sessions[0]) + str(sessions[-1])).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{sessions[-1]:%Y%m%d}_{key}.parquet"
    with _refresh_lock:
        cached = None
        try:
            cached = pd.read_parquet(path)
            complete_download = cached["Ticker"].nunique() == len(universe)
            ttl = 6 * 3600 if complete_download else 15 * 60
            if 0 <= time.time() - path.stat().st_mtime < ttl:
                return cached
        except (OSError, ValueError, KeyError):
            pass
        missing = (sorted(set(universe) - set(cached["Ticker"]))
                   if cached is not None and not cached.empty else universe)
        # Retry failed symbols only; do not redownload hundreds of healthy
        # histories every time one delisted/unavailable symbol is retried.
        downloaded = _download_history(missing or universe, sessions)
        # A transient outage must not destroy a usable same-session cache.
        if cached is not None and not cached.empty:
            downloaded = pd.concat([cached, downloaded], ignore_index=True).drop_duplicates(
                ["Date", "Ticker"], keep="last")
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            downloaded.to_parquet(temporary, index=False)
            os.replace(temporary, path)
        except (OSError, ValueError):
            logger.warning("Cache breadth tidak dapat disimpan; memakai hasil unduhan saat ini.")
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning("File sementara cache breadth tidak dapat dibersihkan.")
        return downloaded


def load_market_breadth(sessions: pd.DatetimeIndex, backtest: bool, *,
                       ma_sessions: int = 200, min_stocks: int = 50,
                       min_coverage: float = .70) -> tuple[pd.Series, dict]:
    sessions = pd.DatetimeIndex(sessions).tz_localize(None).normalize().unique().sort_values()[-ma_sessions:]
    universe = []
    try:
        if len(sessions) >= ma_sessions:
            universe = breadth_universe(_listing_history(), sessions)
        if len(sessions) < ma_sessions or not universe:
            return calculate_breadth(_empty_history(), sessions, universe,
                ma_sessions=ma_sessions, min_stocks=min_stocks, min_coverage=min_coverage)
        history = (pd.read_parquet(STOCK_FILE, columns=["Date", "Ticker", "Close", "Volume"],
                    filters=[("Date", ">=", sessions[0]), ("Date", "<=", sessions[-1])])
                   if backtest else _live_stock_history(universe, sessions))
        return calculate_breadth(history, sessions, universe,
            ma_sessions=ma_sessions, min_stocks=min_stocks, min_coverage=min_coverage)
    except Exception:
        logger.warning("Data breadth tidak tersedia untuk %s.", sessions[-1] if len(sessions) else "unknown")
        values, details = calculate_breadth(_empty_history(), sessions, universe,
            ma_sessions=ma_sessions, min_stocks=min_stocks, min_coverage=min_coverage)
        details["reason"] = "Daftar emiten atau sumber histori breadth tidak tersedia."
        return values, details
