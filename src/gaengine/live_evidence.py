"""Dated live DB evidence for retained-position history and guarded rebalance.

No historical Parquet is substituted for missing current live observations.
The common alignment routines reject incomplete/stale history explicitly.
"""
from datetime import timedelta
import pandas as pd
from src.backend.models.database import SessionLocal
from src.backend.models.market_data import MarketData
from src.backend.models.stock_universe import StockUniverse
from .held_universe import extend_with_holdings


def load_live_held_universe(data, old_lots, sectors=None, frozen_tickers=()):
    outside = sorted(set(old_lots) - set(data.stock_codes))
    if not outside:
        return extend_with_holdings(data, old_lots, pd.DataFrame())
    if data.return_dates is None:
        raise ValueError("Tanggal histori live tidak tersedia untuk guarded_v1.")
    start = pd.Timestamp(data.return_dates[0]).date() - timedelta(days=14)
    with SessionLocal() as db:
        rows = db.query(MarketData.date, StockUniverse.ticker, MarketData.close, MarketData.volume).join(
            StockUniverse, StockUniverse.id_stock == MarketData.stock_id,
        ).filter(StockUniverse.ticker.in_(outside), MarketData.date >= start,
                 MarketData.date <= data.price_date).all()
    frame = pd.DataFrame(rows, columns=["Date", "Ticker", "Close", "Volume"])
    if frame.empty:
        raise ValueError("Histori live posisi lama di luar kandidat tidak tersedia untuk guarded_v1; "
                         "gunakan rebalance legacy pada kedua mode atau lengkapi histori.")
    frame["Date"] = pd.to_datetime(frame.Date)
    for code in outside:
        latest = frame.loc[(frame.Ticker == code) & (frame.Date == pd.Timestamp(data.price_date))]
        if latest.empty or (code not in frozen_tickers and not (pd.to_numeric(latest.Volume, errors="coerce") > 0).any()):
            raise ValueError(f"Bukti perdagangan live {code} pada sesi sinyal tidak tersedia; guarded_v1 dibatalkan.")
    history = frame.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    try:
        return extend_with_holdings(data, old_lots, history, sectors)
    except ValueError as exc:
        raise ValueError(f"{exc} Posisi lama tetap disimpan. Lengkapi histori atau gunakan rebalance legacy.") from exc
