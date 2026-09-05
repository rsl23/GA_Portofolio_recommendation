import logging
from datetime import date

from fastapi import BackgroundTasks
from sqlalchemy import func
from sqlalchemy.orm import Session
from src.backend.models.schemas.market_schema import MarketFilterResponse
from src.backend.models.database import SessionLocal
from src.backend.models.filtered_stocks_cache import FilteredStockCache
from src.backend.models.market_data import MarketData
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
    Ambil histori harga harian (OHLCV) semua saham yang dimiliki user,
    dari tanggal pembuatan portofolio TERAWAL milik user sampai data terbaru.

    Struktur return:
    {
        "start_date": "...",   # tanggal portofolio terawal user
        "end_date": "...",     # tanggal data terbaru
        "stocks": [
            {
                "ticker": "BBCA",
                "first_buy": "2026-08-01",
                "prices": [
                    {"date": "2026-08-01", "open": ..., "high": ...,
                     "low": ..., "close": ..., "volume": ...},
                    ...
                ]
            },
            ...
        ]
    }
    Raises: PortfolioNotFoundError jika user belum punya portofolio.
    """
    from uuid import UUID

    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise PortfolioNotFoundError(f"User ID pada token tidak valid: {user_id}") from e

    # Tanggal awal = portofolio TERAWAL milik user (aktif maupun replaced),
    # agar chart performa vs IHSG mencakup seluruh riwayat kepemilikan.
    start_dt = (
        db.query(func.min(Portofolio.created_at))
        .filter(Portofolio.user_id == user_uuid)
        .scalar()
    )
    if start_dt is None:
        raise PortfolioNotFoundError("User belum memiliki portofolio.")
    start_date = start_dt.date() if hasattr(start_dt, "date") else start_dt

    # Saham-saham yang benar-benar dimiliki user (dari portofolio manapun)
    stock_ids = [
        sid for (sid,) in (
            db.query(PortofolioItem.stock_id)
            .join(Portofolio, PortofolioItem.portofolio_id == Portofolio.id)
            .filter(Portofolio.user_id == user_uuid)
            .distinct()
            .all()
        )
    ]
    if not stock_ids:
        raise PortfolioNotFoundError("User belum memiliki saham di portofolionya.")

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
            "volume": md.volume,
        })

    max_date = max(
        (md.date for md, _ in rows if isinstance(md.date, date)),
        default=None,
    )

    return {
        "start_date": start_dt.date().isoformat() if hasattr(start_dt, "date") else str(start_dt),
        "end_date": max_date.isoformat() if max_date else None,
        "stocks": [grouped[t] for t in sorted(grouped)],
    }

