import logging

from fastapi import BackgroundTasks
from src.backend.models.schemas.market_schema import MarketFilterResponse
from src.backend.models.database import SessionLocal
from src.backend.models.filtered_stocks_cache import FilteredStockCache
from src.preprocessing.stock_filtering import run_live_preprocessing

logger = logging.getLogger(__name__)


class StockFilteringError(Exception):
    """Exception domain: pipeline seleksi saham gagal dijalankan."""


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

