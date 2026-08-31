from src.backend.models.database import SessionLocal
from src.backend.models.filtered_stocks_cache import FilteredStockCache

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

