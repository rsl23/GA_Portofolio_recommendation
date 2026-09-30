from datetime import date, timedelta

import logging

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

logger = logging.getLogger(__name__)


IDX_TICKER = "^JKSE"  # IHSG di yfinance


def _to_float(value) -> float | None:
    """
    Konversi aman ke float.
    Return None jika value None, bukan angka, atau NaN.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None

    if f != f:  # NaN check
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

        if df.empty:
            logger.warning(
                "Tidak dapat menentukan latest trading date IHSG: "
                "semua Close bernilai NaN."
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
    Ambil daftar (stock_id, ticker, tanggal_pembelian_terawal) untuk SEMUA saham
    yang pernah dibeli user di portofolio manapun (active maupun replaced),
    karena histori harga tetap dibutuhkan untuk backtesting & performa vs IHSG.
    """
    rows = (
        db.query(
            PortofolioItem.stock_id,
            StockUniverse.ticker,
            func.min(
                func.coalesce(
                    Portofolio.created_at,
                    func.now(),
                )
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
        .group_by(
            PortofolioItem.stock_id,
            StockUniverse.ticker,
        )
        .all()
    )

    return [
        (r.stock_id, r.ticker, r.first_buy)
        for r in rows
    ]


def _get_earliest_portfolio_date(db: Session):
    """
    Tanggal pembentukan portofolio TERAWAL dari semua user
    (batas awal data IHSG).
    """
    dt = db.query(func.min(Portofolio.created_at)).scalar()

    return (
        dt.date()
        if dt is not None and hasattr(dt, "date")
        else dt
    )


def _sync_idx_composite(
    db: Session,
    start_date,
    latest_trading_date: date,
    errors: list[str],
) -> int:
    """
    Sinkronisasi harga IHSG (^JKSE) via yfinance ke tabel idx_composite.

    Rentang:
        start_date -> latest_trading_date

    Karena parameter `end` yfinance bersifat exclusive, maka:
        end_date = latest_trading_date + 1 hari

    Returns:
        jumlah baris yang berhasil di-upsert.
    """
    try:
        # Tidak ada data yang perlu diambil jika portfolio dibuat
        # setelah latest trading date.
        if start_date > latest_trading_date:
            logger.info(
                "idx_composite dilewati: start_date=%s > "
                "latest_trading_date=%s",
                start_date,
                latest_trading_date,
            )
            return 0

        # yfinance menggunakan `end` sebagai exclusive boundary.
        # Jadi +1 hari agar latest_trading_date ikut terambil.
        end_date = latest_trading_date + timedelta(days=1)

        df = yf.download(
            IDX_TICKER,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if df is None or df.empty:
            errors.append(
                f"{IDX_TICKER}: tidak ada data harga "
                f"antara {start_date} dan {latest_trading_date}"
            )

            logger.warning(
                "%s: tidak ada data harga antara %s dan %s",
                IDX_TICKER,
                start_date,
                latest_trading_date,
            )

            return 0

        # Normalisasi multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        upserted = 0

        for idx, row in df.iterrows():
            row_date = (
                idx.date()
                if hasattr(idx, "date")
                else idx
            )

            open_p = _to_float(row.get("Open"))
            high_p = _to_float(row.get("High"))
            low_p = _to_float(row.get("Low"))
            close_p = _to_float(row.get("Close"))

            # Lewati baris yang tidak lengkap
            if None in (
                open_p,
                high_p,
                low_p,
                close_p,
            ):
                continue

            stmt = pg_insert(IdxComposite).values(
                ticker=IDX_TICKER,
                date=row_date,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
            )

            # Upsert berdasarkan (ticker, date)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_idx_composite_ticker_date",
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                },
            )

            db.execute(stmt)
            upserted += 1

        db.commit()

        logger.info(
            "idx_composite %s: %d baris tersinkron "
            "sejak %s sampai %s",
            IDX_TICKER,
            upserted,
            start_date,
            latest_trading_date,
        )

        return upserted

    except Exception as e:
        db.rollback()

        errors.append(
            f"{IDX_TICKER}: {e}"
        )

        logger.exception(
            "Gagal sync harga IHSG"
        )

        return 0


def sync_market_data(db: Session) -> dict:
    """
    Sinkronisasi histori harga harian (OHLCV) via yfinance untuk semua saham
    yang pernah dibeli user.

    Rentang:
        - dari tanggal pembelian TERAWAL portofolio yang memuat saham tsb
        - sampai latest trading date berdasarkan IHSG

    Dengan demikian, jika hari ini merupakan:
        - Sabtu/Minggu
        - hari libur BEI
        - atau hari non-trading lainnya

    maka sinkronisasi hanya dilakukan sampai hari perdagangan terakhir.

    Data di-upsert sehingga aman dijalankan berulang kali.

    Returns:
        {
            "stocks": jumlah saham,
            "rows_upserted": jumlah row market_data,
            "idx_rows_upserted": jumlah row idx_composite,
            "errors": daftar error
        }
    """

    earliest_portfolio_date = _get_earliest_portfolio_date(db)

    if earliest_portfolio_date is None:
        logger.info(
            "Sync market_data dilewati: "
            "belum ada portofolio sama sekali."
        )

        return {
            "stocks": 0,
            "rows_upserted": 0,
            "idx_rows_upserted": 0,
            "errors": [],
        }

    errors: list[str] = []
    total_upserted = 0

    # ============================================================
    # 1. Tentukan tanggal perdagangan terakhir
    # ============================================================
    latest_trading_date = _get_latest_trading_date()

    if latest_trading_date is None:
        error_msg = (
            "Tidak dapat menentukan latest trading date IHSG."
        )

        logger.error(error_msg)
        errors.append(error_msg)

        return {
            "stocks": 0,
            "rows_upserted": 0,
            "idx_rows_upserted": 0,
            "errors": errors,
        }

    logger.info(
        "Latest trading date berdasarkan IHSG: %s",
        latest_trading_date,
    )

    # ============================================================
    # 2. Sinkronkan harga IHSG
    # ============================================================
    idx_upserted = _sync_idx_composite(
        db=db,
        start_date=earliest_portfolio_date,
        latest_trading_date=latest_trading_date,
        errors=errors,
    )

    # ============================================================
    # 3. Ambil semua saham yang pernah dimiliki
    # ============================================================
    holdings = _get_holdings_since(db)

    for stock_id, ticker, first_buy in holdings:
        start_date = (
            first_buy.date()
            if hasattr(first_buy, "date")
            else first_buy
        )

        # ========================================================
        # 4. Jika portfolio dibuat setelah latest trading date,
        #    belum ada market data yang bisa di-sync.
        # ========================================================
        if start_date > latest_trading_date:
            logger.info(
                "market_data %s dilewati: "
                "start_date=%s > latest_trading_date=%s",
                ticker,
                start_date,
                latest_trading_date,
            )
            continue

        # ========================================================
        # 5. Karena `end` yfinance exclusive,
        #    tambahkan 1 hari agar latest_trading_date ikut masuk.
        # ========================================================
        end_date = latest_trading_date + timedelta(days=1)

        try:
            df = yf.download(
                ticker + ".JK",
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                interval="1d",
                auto_adjust=False,
                progress=False,
            )

            # ====================================================
            # 6. Tangani jika tidak ada data
            # ====================================================
            if df is None or df.empty:
                logger.warning(
                    "market_data %s: tidak ada data perdagangan "
                    "antara %s dan %s",
                    ticker,
                    start_date,
                    latest_trading_date,
                )
                continue

            # ====================================================
            # 7. Normalisasi MultiIndex
            # ====================================================
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # ====================================================
            # 8. Upsert setiap trading day
            # ====================================================
            upserted = 0

            for idx, row in df.iterrows():
                row_date = (
                    idx.date()
                    if hasattr(idx, "date")
                    else idx
                )

                open_p = _to_float(row.get("Open"))
                high_p = _to_float(row.get("High"))
                low_p = _to_float(row.get("Low"))
                close_p = _to_float(row.get("Close"))
                volume = _to_float(row.get("Volume"))
                adj_close_p = _to_float(row.get("Adj Close"))

                # Lewati baris yang tidak lengkap
                if None in (
                    open_p,
                    high_p,
                    low_p,
                    close_p,
                    volume,
                ):
                    continue

                stmt = pg_insert(MarketData).values(
                    stock_id=stock_id,
                    date=row_date,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    adj_close=adj_close_p,
                    volume=int(volume),
                )

                # =================================================
                # 9. Upsert berdasarkan (stock_id, date)
                # =================================================
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_market_data_stock_date",
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "adj_close": stmt.excluded.adj_close,
                        "volume": stmt.excluded.volume,
                    },
                )

                db.execute(stmt)
                upserted += 1

            db.commit()

            total_upserted += upserted

            logger.info(
                "market_data %s: %d baris tersinkron "
                "sejak %s sampai %s",
                ticker,
                upserted,
                start_date,
                latest_trading_date,
            )

        except Exception as e:
            db.rollback()

            errors.append(
                f"{ticker}: {e}"
            )

            logger.exception(
                "Gagal sync harga untuk %s",
                ticker,
            )

    return {
        "stocks": len(holdings),
        "rows_upserted": total_upserted,
        "idx_rows_upserted": idx_upserted,
        "errors": errors,
    }