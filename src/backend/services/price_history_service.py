from datetime import date

import pandas as pd
import yfinance as yf
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

import logging

from src.backend.models.market_data import MarketData
from src.backend.models.portofolio_items import PortofolioItem
from src.backend.models.portofolios import Portofolio
from src.backend.models.stock_universe import StockUniverse

logger = logging.getLogger(__name__)


def _get_holdings_since(db: Session) -> list[tuple]:
    """
    Ambil daftar (stock_id, ticker, tanggal_pembelian_terawal) untuk SEMUA saham
    yang pernah dibeli user di portofolio manapun (active maupun replaced),
    karena histori harga tetap dibutuhkan untuk backtesting & performa vs IHSG.
    """
    rows = (
        db.query(
            PortofolioItem.stock_id,
            StockUniverse.ticker,
            func.min(func.coalesce(Portofolio.created_at, func.now())).label("first_buy"),
        )
        .join(Portofolio, PortofolioItem.portofolio_id == Portofolio.id)
        .join(StockUniverse, PortofolioItem.stock_id == StockUniverse.id_stock)
        .group_by(PortofolioItem.stock_id, StockUniverse.ticker)
        .all()
    )
    return [(r.stock_id, r.ticker, r.first_buy) for r in rows]


def sync_market_data(db: Session) -> dict:
    """
    Sinkronisasi histori harga harian (OHLCV) via yfinance untuk semua saham
    yang pernah dibeli user:
    - Rentang: dari tanggal pembelian TERAWAL portofolio yang memuat saham tsb
      sampai hari ini.
    - Upsert ke tabel market_data (aman dijalankan berulang: baris yang sudah
      ada di-UPDATE — penting agar harga hari ini selalu terbarui).
    Returns: statistik {stocks, rows_upserted, errors}.
    """
    holdings = _get_holdings_since(db)
    if not holdings:
        logger.info("Sync market_data dilewati: belum ada saham yang dibeli user.")
        return {"stocks": 0, "rows_upserted": 0, "errors": []}

    errors: list[str] = []
    total_upserted = 0

    for stock_id, ticker, first_buy in holdings:
        start_date = first_buy.date() if hasattr(first_buy, "date") else first_buy
        try:
            # Ambil OHLCV sejak tanggal pembelian (end default = hari ini)
            df = yf.download(
                ticker + ".JK",
                start=start_date.isoformat(),
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
            if df is None or df.empty:
                errors.append(f"{ticker}: tidak ada data harga dari yfinance")
                continue

            # Normalisasi multi-index (yfinance kadang mengembalikan MultiIndex kolom)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            upserted = 0
            for idx, row in df.iterrows():
                row_date = idx.date() if hasattr(idx, "date") else idx
                open_p = _to_float(row.get("Open"))
                high_p = _to_float(row.get("High"))
                low_p = _to_float(row.get("Low"))
                close_p = _to_float(row.get("Close"))
                volume = _to_float(row.get("Volume"))
                if None in (open_p, high_p, low_p, close_p, volume):
                    continue  # lewati baris NaN (hari libur / data kosong)

                stmt = pg_insert(MarketData).values(
                    stock_id=stock_id,
                    date=row_date,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=int(volume),
                )
                # Upsert: baris (stock_id, date) yang sudah ada di-update dengan
                # harga terbaru (yfinance bisa merevisi harga hari berjalan)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_market_data_stock_date",
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                    },
                )
                db.execute(stmt)
                upserted += 1

            db.commit()
            total_upserted += upserted
            logger.info("market_data %s: %d baris tersinkron sejak %s", ticker, upserted, start_date)
        except Exception as e:
            db.rollback()
            errors.append(f"{ticker}: {e}")
            logger.exception("Gagal sync harga untuk %s", ticker)

    return {
        "stocks": len(holdings),
        "rows_upserted": total_upserted,
        "errors": errors,
    }


def _to_float(value) -> float | None:
    """Konversi aman ke float; None jika NaN/None (baris libur / data rusak)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check
        return None
    return f
