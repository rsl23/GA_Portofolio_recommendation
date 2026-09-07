import logging
from datetime import date

from fastapi import BackgroundTasks
from sqlalchemy import func
from sqlalchemy.orm import Session
from src.backend.models.schemas.market_schema import MarketFilterResponse
from src.backend.models.database import SessionLocal
from src.backend.models.filtered_stocks_cache import FilteredStockCache
from src.backend.models.market_data import MarketData
from src.backend.models.idx_composite import IdxComposite
from src.backend.models.portofolio_items import PortofolioItem
from src.backend.models.portofolios import Portofolio
from src.backend.models.stock_universe import StockUniverse
from src.preprocessing.stock_filtering import run_live_preprocessing

logger = logging.getLogger(__name__)


class StockFilteringError(Exception):
    """Exception domain: pipeline seleksi saham gagal dijalankan."""


class PortfolioNotFoundError(Exception):
    """Exception domain: user belum memiliki portofolio (tidak ada histori harga)."""


def run_and_cache_stock_filtering(background_tasks: BackgroundTasks) -> MarketFilterResponse:
    """
    Controller untuk endpoint /market/filter-stocks:
    1. Menjalankan pipeline seleksi saham secara live (1-2 menit).
    2. Mengirim tugas simpan ke DB sebagai background task (asynchronous).
    3. Mengembalikan hasil domain (MarketFilterResponse) untuk dibungkus envelope oleh routes.
    """
    try:
        daftar_saham, df_lolos = run_live_preprocessing()
    except Exception as e:
        logger.exception("Gagal menjalankan pipeline filtering saham")
        raise StockFilteringError(f"Filtering saham gagal: {e}")

    # 'Kode' tersembunyi sebagai index di df_lolos, jadi di-reset dulu
    df_json = df_lolos.reset_index().to_dict(orient="records")

    background_tasks.add_task(save_filtered_stocks_to_db, df_json)

    return MarketFilterResponse(
        total_saham=len(daftar_saham),
        data=df_json,
    )


def get_last_filter_update(db: Session):
    """
    Ambil updated_at dari salah satu baris filtered_stock_cache.
    Semua baris ditulis bersamaan oleh job filtering (truncate + insert),
    sehingga updated_at sinkron untuk semua row — cukup ambil MAX.
    Returns: datetime atau None jika tabel kosong.
    """
    from sqlalchemy import func
    from src.backend.models.filtered_stocks_cache import FilteredStockCache

    return db.query(func.max(FilteredStockCache.updated_at)).scalar()


def run_daily_pipeline() -> dict:
    """
    Pipeline harian (dipanggil scheduler jam 04:00 atau lifespan saat data basi):
      1. Filtering saham hari ini -> simpan ke filtered_stock_cache.
      2. build_market_data() -> muat data pasar + fundamental ke RAM untuk GA.
      3. sync_market_data -> tarik harga terbaru (yfinance) untuk market_data
         & idx_composite milik portofolio user.
    Returns: statistik ringkas; 'market_data' berupa objek MarketData untuk
    disimpan ke app.state oleh pemanggil.
    """
    from src.gaengine.data_loader_live import build_market_data
    from src.backend.services.price_history_service import sync_market_data

    db = SessionLocal()
    try:
        # 1. Filtering & simpan cache (SINKRON, bukan background task,
        #    karena scheduler tidak punya BackgroundTasks FastAPI)
        daftar_saham, df_lolos = run_live_preprocessing()
        df_json = df_lolos.reset_index().to_dict(orient="records")
        save_filtered_stocks_to_db(df_json)
        logger.info("Pipeline harian [1/3]: filtering selesai (%d saham lolos).", len(daftar_saham))

        # 2. Perakitan MarketData ke RAM (fundamental + OHLCV 1 tahun)
        market_data = build_market_data()
        logger.info("Pipeline harian [2/3]: build_market_data selesai (%s saham).",
                    market_data.n_stocks if market_data else 0)

        # 3. Sinkronisasi harga portofolio user + IHSG
        stats = sync_market_data(db)
        logger.info("Pipeline harian [3/3]: sync_market_data selesai (%s).", stats)

        return {
            "filtered_stocks": len(daftar_saham),
            "market_data": market_data,
            "sync_stats": stats,
        }
    finally:
        db.close()


def save_filtered_stocks_to_db(data_records: list):
    """
    Fungsi ini berjalan di background (Asynchronous Task).
    Tugasnya adalah menyimpan hasil dataframe yang sudah diubah menjadi dictionary
    ke dalam tabel filtered_stock_cache di database.
    """
    # Membuat sesi DB baru khusus untuk proses background
    db = SessionLocal()
    try:
        # Hapus data cache lama agar selalu fresh (Truncate)
        db.query(FilteredStockCache).delete()
        
        # Insert data baru satu per satu
        for row in data_records:
            cache_item = FilteredStockCache(
                kode=row.get('Kode', ''),
                nama=row.get('NamaEmiten', ''), # Sesuaikan dengan nama kolom dari API IDX
                sektor=row.get('Sektor', ''),
                close=float(row.get('Close', 0.0)),
                market_cap=float(row.get('Market_Cap', 0.0)),
                eps=float(row.get('EPS', 0.0)),
                roe=float(row.get('ROE', 0.0)),
                der=float(row.get('DER', 0.0)),
                adtv_60=float(row.get('ADTV_60', 0.0))
            )
            db.add(cache_item)
            
        # Simpan ke database
        db.commit()
        print(f"Background Task: {len(data_records)} saham berhasil disimpan ke tabel FilteredStockCache.")
    
    except Exception as e:
        db.rollback()
        print(f"Background Task Error saat menyimpan ke DB: {e}")
    finally:
        # Tutup sesi agar memori tidak bocor
        db.close()


def get_price_history(db: Session, user_id: str) -> dict:
    """
    Ambil histori harga harian (OHLCV + adj_close) saham yang dimiliki user
    pada portofolio BERSTATUS ACTIVE (selalu yang terbaru) DAN harga IHSG
    (benchmark) dari tabel idx_composite,
    dari tanggal pembuatan portofolio active tersebut sampai data terbaru.

    Struktur return:
    {
        "start_date": "...",   # tanggal portofolio active user
        "end_date": "...",     # tanggal data terbaru (saham maupun IHSG)
        "stocks": [
            {
                "ticker": "BBCA",
                "stock_id": "...",
                "prices": [
                    {"date": "2026-08-01", "open": ..., "high": ...,
                     "low": ..., "close": ..., "adj_close": ..., "volume": ...},
                    ...
                ]
            },
            ...
        ],
        "benchmark": {
            "ticker": "^JKSE",
            "prices": [
                {"date": "2026-08-01", "open": ..., "high": ...,
                 "low": ..., "close": ...},
                ...
            ]
        }
    }
    Raises: PortfolioNotFoundError jika user belum punya portofolio aktif.
    """
    from uuid import UUID

    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise PortfolioNotFoundError(f"User ID pada token tidak valid: {user_id}") from e

    # Tanggal awal = portofolio BERSTATUS ACTIVE (yang paling baru).
    # Portofolio lama berstatus 'replaced' tidak disertakan agar tampilan
    # selalu mengikuti portofolio terbaru.
    start_dt = (
        db.query(func.min(Portofolio.created_at))
        .filter(
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio == "active",
        )
        .scalar()
    )
    if start_dt is None:
        raise PortfolioNotFoundError("User belum memiliki portofolio aktif.")
    start_date = start_dt.date() if hasattr(start_dt, "date") else start_dt

    # Saham-saham yang dimiliki user pada portofolio ACTIVE
    stock_ids = [
        sid for (sid,) in (
            db.query(PortofolioItem.stock_id)
            .join(Portofolio, PortofolioItem.portofolio_id == Portofolio.id)
            .filter(
                Portofolio.user_id == user_uuid,
                Portofolio.status_portofolio == "active",
            )
            .distinct()
            .all()
        )
    ]
    if not stock_ids:
        raise PortfolioNotFoundError("User belum memiliki saham di portofolio aktifnya.")

    # Histori harga untuk saham-saham tsb, sejak tanggal portofolio terawal.
    # PENTING: bandingkan Date dengan Date (start_date), BUKAN dengan
    # start_dt (DateTime) — '2026-09-04' >= '2026-09-04 14:30' akan FALSE
    # karena date di-cast jadi timestamp 00:00:00.
    rows = (
        db.query(MarketData, StockUniverse.ticker)
        .join(StockUniverse, MarketData.stock_id == StockUniverse.id_stock)
        .filter(
            MarketData.stock_id.in_(stock_ids),
            MarketData.date >= start_date,
        )
        .order_by(MarketData.stock_id, MarketData.date.asc())
        .all()
    )

    # Grouping per ticker agar mudah digambar chart di frontend
    grouped: dict[str, dict] = {}
    for md, ticker in rows:
        entry = grouped.setdefault(ticker, {"ticker": ticker, "stock_id": str(md.stock_id), "prices": []})
        entry["prices"].append({
            "date": md.date.isoformat(),
            "open": md.open,
            "high": md.high,
            "low": md.low,
            "close": md.close,
            "adj_close": md.adj_close,
            "volume": md.volume,
        })

    all_dates = [md.date for md, _ in rows if isinstance(md.date, date)]

    # Benchmark IHSG dari idx_composite (rentang sama: sejak portofolio terawal)
    idx_rows = (
        db.query(IdxComposite)
        .filter(IdxComposite.date >= start_date)
        .order_by(IdxComposite.date.asc())
        .all()
    )
    benchmark_prices = [
        {
            "date": ic.date.isoformat(),
            "open": ic.open,
            "high": ic.high,
            "low": ic.low,
            "close": ic.close,
        }
        for ic in idx_rows
    ]
    all_dates.extend(ic.date for ic in idx_rows if isinstance(ic.date, date))

    max_date = max(all_dates, default=None)

    return {
        "start_date": start_dt.date().isoformat() if hasattr(start_dt, "date") else str(start_dt),
        "end_date": max_date.isoformat() if max_date else None,
        "stocks": [grouped[t] for t in sorted(grouped)],
        "benchmark": {
            "ticker": idx_rows[0].ticker if idx_rows else None,
            "prices": benchmark_prices,
        },
    }




