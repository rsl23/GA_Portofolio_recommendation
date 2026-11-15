from datetime import date, datetime, timedelta, timezone

import logging
import math

import pandas as pd
import yfinance as yf
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.backend.models.market_data import MarketData
from src.backend.models.idx_composite import IdxComposite
from src.backend.models.portofolio_items import PortofolioItem
from src.backend.models.portofolios import Portofolio
from src.backend.models.stock_universe import StockUniverse
from src.backend.models.filtered_stocks_cache import FilteredStockCache
from src.backend.services.cached_close_prices import cached_close_for_session

logger = logging.getLogger(__name__)


IDX_TICKER = "^JKSE"  # IHSG di yfinance
LIVE_PORTFOLIO_STATUSES = ("active", "replaced")


def _to_float(value) -> float | None:
    """
    Konversi aman ke float.
    Return None jika value None, bukan angka, atau NaN.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(f):
        return None

    return f


def _get_latest_trading_date() -> date | None:
    """
    Ambil tanggal perdagangan terakhir yang tersedia berdasarkan IHSG (^JKSE).

    Tujuannya agar sinkronisasi tidak memaksa yfinance mencari data pada
    tanggal kalender hari ini ketika hari ini merupakan weekend atau
    hari libur bursa.

    Contoh:
    - Jumat 25 Sep 2026 -> latest trading date = 25 Sep 2026
    - Sabtu 26 Sep 2026 -> latest trading date = 25 Sep 2026
    - Minggu 27 Sep 2026 -> latest trading date = 25 Sep 2026
    - Hari libur BEI -> latest trading date = hari perdagangan sebelumnya
    """
    try:
        df = yf.download(
            IDX_TICKER,
            period="10d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if df is None or df.empty:
            logger.warning(
                "Tidak dapat menentukan latest trading date IHSG: "
                "yfinance tidak mengembalikan data."
            )
            return None

        # Normalisasi MultiIndex jika yfinance mengembalikannya
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Hanya gunakan bar yang benar-benar memiliki data Close
        if "Close" not in df.columns:
            logger.warning(
                "Tidak dapat menentukan latest trading date IHSG: "
                "kolom Close tidak ditemukan."
            )
            return None

        df = df.dropna(subset=["Close"])
        # Match the daily pipeline's 17:00 WIB cut-off. A morning manual
        # sync must not permanently store an unfinished daily candle now
        # that subsequent syncs skip prices already present in the DB.
        now_wib = datetime.now(timezone(timedelta(hours=7)))
        if now_wib.hour < 17:
            df = df.loc[[pd.Timestamp(day).date() < now_wib.date() for day in df.index]]

        if df.empty:
            logger.warning(
                "Tidak dapat menentukan latest trading date IHSG: "
                "tidak ada sesi selesai dengan Close yang tersedia."
            )
            return None

        last_idx = df.index[-1]

        if hasattr(last_idx, "date"):
            return last_idx.date()

        return last_idx

    except Exception as e:
        logger.exception(
            "Gagal menentukan latest trading date IHSG: %s",
            e,
        )
        return None


def _get_holdings_since(db: Session) -> list[tuple]:
    """
    Ambil saham yang pernah dimiliki dalam mode LIVE (active maupun replaced).
    Portofolio backtest tidak menentukan ticker atau tanggal awal sinkronisasi.
    """
    rows = (
        db.query(
            PortofolioItem.stock_id,
            StockUniverse.ticker,
            func.min(
                func.date(func.coalesce(
                    Portofolio.execution_date,
                    PortofolioItem.start_date,
                    Portofolio.date_ref,
                    Portofolio.created_at,
                    func.now(),
                ))
            ).label("first_buy"),
        )
        .join(
            Portofolio,
            PortofolioItem.portofolio_id == Portofolio.id,
        )
        .join(
            StockUniverse,
            PortofolioItem.stock_id == StockUniverse.id_stock,
        )
        .filter(Portofolio.status_portofolio.in_(LIVE_PORTFOLIO_STATUSES))
        .group_by(
            PortofolioItem.stock_id,
            StockUniverse.ticker,
        )
        .all()
    )

    return [
        (r.stock_id, r.ticker, _calendar_date(r.first_buy))
        for r in rows
    ]


def _calendar_date(value):
    if value is None or isinstance(value, date) and not hasattr(value, 'date'):
        return value
    return value.date() if hasattr(value, 'date') else date.fromisoformat(str(value)[:10])


def _get_earliest_portfolio_date(db: Session):
    """
    Tanggal eksekusi efektif portofolio LIVE terawal dari semua pengguna,
    fallback jadwal/pembuatan (batas awal data IHSG).
    """
    dt = db.query(func.min(func.date(func.coalesce(
        Portofolio.execution_date, Portofolio.date_ref, Portofolio.created_at,
    )))).filter(Portofolio.status_portofolio.in_(LIVE_PORTFOLIO_STATUSES)).scalar()

    return _calendar_date(dt)


def _missing_price_ranges(
    start_date: date,
    end_date: date,
    existing_dates: set[date],
    session_dates: set[date],
) -> list[tuple[date, date]]:
    """Plan inclusive missing ranges without crossing a stored trading day.

    Known sessions identify interior holes. Outside stored coverage, request
    weekdays so a new earlier holding and new trading days are both covered.
    """
    if start_date > end_date:
        return []
    existing = {day for day in existing_dates if start_date <= day <= end_date}
    missing = {day for day in session_dates if start_date <= day <= end_date} - existing

    def add_weekdays(first, last):
        if first <= last:
            missing.update(day.date() for day in pd.bdate_range(first, last))

    if not existing:
        add_weekdays(start_date, end_date)
    else:
        add_weekdays(start_date, min(existing) - timedelta(days=1))
        add_weekdays(max(existing) + timedelta(days=1), end_date)
    missing.difference_update(existing)
    if not missing:
        return []

    days = sorted(missing)
    ranges = []
    first = last = days[0]
    for day in days[1:]:
        following = last + timedelta(days=1)
        while following.weekday() >= 5:
            following += timedelta(days=1)
        if day != following:
            ranges.append((first, last))
            first = day
        last = day
    ranges.append((first, last))
    return ranges


def _stored_index_dates(db: Session, start_date: date, end_date: date) -> set[date]:
    return {
        _calendar_date(row[0])
        for row in db.query(IdxComposite.date).filter(
            IdxComposite.ticker == IDX_TICKER,
            IdxComposite.date >= start_date,
            IdxComposite.date <= end_date,
            IdxComposite.open > 0, IdxComposite.high > 0,
            IdxComposite.low > 0, IdxComposite.close > 0,
        ).all()
    }


def _sync_price_ranges(
    db: Session,
    ticker: str,
    ranges: list[tuple[date, date]],
    errors: list[str],
    stock_id=None,
) -> int:
    """Download only the planned ranges; commit each range independently."""
    model = MarketData if stock_id is not None else IdxComposite
    total = 0
    for start_date, end_date in ranges:
        try:
            frame = yf.download(
                ticker, start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                interval="1d", auto_adjust=False, progress=False,
            )
            if frame is None or frame.empty:
                errors.append(f"{ticker}: tidak ada data harga antara {start_date} dan {end_date}")
                continue
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)

            inserted = 0
            cache_record = None
            cache_loaded = False
            for timestamp, row in frame.iterrows():
                day = _calendar_date(timestamp)
                if not start_date <= day <= end_date:
                    continue
                values = {name.lower(): _to_float(row.get(name))
                          for name in ("Open", "High", "Low", "Close")}
                if stock_id is not None and (values['close'] is None or values['close'] <= 0):
                    if not cache_loaded:
                        cache_record = db.query(FilteredStockCache).filter(
                            FilteredStockCache.kode == ticker.removesuffix('.JK')).first()
                        cache_loaded = True
                    replacement = cached_close_for_session(cache_record, day, row.get('Volume'))
                    if replacement is not None:
                        values['close'] = replacement
                        logger.info('%s: Close fallback filtered_stock_cache pada %s.', ticker, day)
                if any(value is None or value <= 0 for value in values.values()):
                    continue
                if stock_id is not None:
                    volume = _to_float(row.get("Volume"))
                    if volume is None or volume < 0:
                        continue
                    values.update(
                        stock_id=stock_id, date=day, volume=int(volume),
                        adj_close=_to_float(row.get("Adj Close")),
                    )
                    unique_columns = ["stock_id", "date"]
                    updated_columns = ["open", "high", "low", "close", "adj_close", "volume"]
                else:
                    values.update(ticker=IDX_TICKER, date=day)
                    unique_columns = ["ticker", "date"]
                    updated_columns = ["open", "high", "low", "close"]

                stmt = pg_insert(model).values(**values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_columns,
                    set_={column: getattr(stmt.excluded, column) for column in updated_columns},
                )
                result = db.execute(stmt)
                inserted += max(result.rowcount, 0)
            db.commit()
            total += inserted
            logger.info("%s: %d baris tersinkron pada rentang kosong %s sampai %s",
                        ticker, inserted, start_date, end_date)
        except Exception as exc:
            db.rollback()
            errors.append(f"{ticker} ({start_date} sampai {end_date}): {exc}")
            logger.exception("Gagal sync harga %s pada %s sampai %s", ticker, start_date, end_date)
    return total


def _sync_idx_composite(
    db: Session,
    start_date,
    latest_trading_date: date,
    errors: list[str],
) -> int:
    """Fill missing IHSG coverage and interior sessions seen in stored stocks."""
    start_date = _calendar_date(start_date)
    if start_date > latest_trading_date:
        return 0
    try:
        existing_dates = _stored_index_dates(db, start_date, latest_trading_date)
        # A traded stock is independent evidence that an absent IHSG row is
        # a genuine hole, rather than an exchange holiday.
        known_sessions = {
            _calendar_date(row[0])
            for row in db.query(MarketData.date).filter(
                MarketData.date >= start_date, MarketData.date <= latest_trading_date,
                MarketData.close > 0, MarketData.volume > 0,
            ).distinct().all()
        }
        ranges = _missing_price_ranges(
            start_date, latest_trading_date, existing_dates, known_sessions,
        )
        return _sync_price_ranges(db, IDX_TICKER, ranges, errors)
    except Exception as exc:
        db.rollback()
        errors.append(f"{IDX_TICKER}: {exc}")
        logger.exception("Gagal memeriksa histori IHSG")
        return 0


def sync_market_data(db: Session, stock_ids: list | None = None) -> dict:
    """Fill missing LIVE stock/IHSG history without re-downloading stored prices.

    Historical coverage starts at the earliest holding and ends at the last
    available IHSG session. Only missing prefix, tail, and known interior
    sessions are downloaded. Existing dates are not refreshed for subsequent
    vendor corrections to historical prices or adjusted closes.
    Backtest portfolios and their earlier dates are excluded; their source
    remains the local parquet archives.
    """
    holdings = _get_holdings_since(db)
    if stock_ids is not None:
        selected = {str(value) for value in stock_ids}
        holdings = [row for row in holdings if str(row[0]) in selected]
        earliest_portfolio_date = min((row[2] for row in holdings), default=None)
    else:
        earliest_portfolio_date = _get_earliest_portfolio_date(db)

    stats = {"stocks": 0, "rows_upserted": 0, "idx_rows_upserted": 0, "errors": []}
    if earliest_portfolio_date is None:
        return stats
    errors = stats["errors"]
    latest_trading_date = _get_latest_trading_date()
    # A dated IDX stock summary can prove a newer completed session even when
    # Yahoo's IHSG Close is missing. Keep stock coverage current independently;
    # the benchmark still needs its own actual Close and is never fabricated.
    now_wib = datetime.now(timezone(timedelta(hours=7)))
    cutoff = now_wib.date() if now_wib.hour >= 17 else now_wib.date() - timedelta(days=1)
    cache_day = db.query(func.max(FilteredStockCache.price_date)).filter(
        FilteredStockCache.price_date <= cutoff, FilteredStockCache.close > 0,
    ).scalar()
    if cache_day is not None:
        cache_day = _calendar_date(cache_day)
        if (cutoff - cache_day).days <= 10:
            latest_trading_date = max(latest_trading_date, cache_day) if latest_trading_date else cache_day
    if latest_trading_date is None:
        errors.append("Tidak dapat menentukan latest trading date IHSG.")
        return stats

    stats["stocks"] = len(holdings)
    stats["idx_rows_upserted"] = _sync_idx_composite(
        db, earliest_portfolio_date, latest_trading_date, errors,
    )
    session_dates = _stored_index_dates(db, earliest_portfolio_date, latest_trading_date)
    for stock_id, ticker, first_buy in holdings:
        start_date = _calendar_date(first_buy)
        if start_date > latest_trading_date:
            continue
        try:
            existing_dates = {
                _calendar_date(row[0])
                for row in db.query(MarketData.date).filter(
                    MarketData.stock_id == stock_id,
                    MarketData.date >= start_date, MarketData.date <= latest_trading_date,
                    MarketData.open > 0, MarketData.high > 0,
                    MarketData.low > 0, MarketData.close > 0, MarketData.volume >= 0,
                ).all()
            }
            ranges = _missing_price_ranges(
                start_date, latest_trading_date, existing_dates, session_dates,
            )
            stats["rows_upserted"] += _sync_price_ranges(
                db, ticker + ".JK", ranges, errors, stock_id=stock_id,
            )
        except Exception as exc:
            db.rollback()
            errors.append(f"{ticker}: {exc}")
            logger.exception("Gagal memeriksa histori saham %s", ticker)
    return stats
