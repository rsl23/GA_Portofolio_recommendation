from fastapi import APIRouter, BackgroundTasks
from src.preprocessing.stock_filtering import run_live_preprocessing
from src.backend.controller.market_controller import save_filtered_stocks_to_db

router = APIRouter()

@router.get("/filter-stocks")
def filter_stocks_endpoint(background_tasks: BackgroundTasks):
    """
    Endpoint ini akan menjalankan pipeline seleksi saham secara live.
    1. Mengunduh data & memfilter saham (Synchronous - butuh 1-2 menit).
    2. Mengirim hasilnya ke tabel Database secara Asynchronous (Background).
    3. Mengembalikan format JSON ke Frontend (Response instan).
    """
    try:
        # 1. Jalankan proses utama
        daftar_saham, df_lolos = run_live_preprocessing()
        
        # 2. Konversi Pandas DataFrame ke format List of Dictionary agar menjadi JSON murni
        # Kita menggunakan reset_index() karena 'Kode' tersembunyi sebagai index di df_lolos
        df_json = df_lolos.reset_index().to_dict(orient="records")
        
        # 3. Lempar tugas simpan DB ke background
        background_tasks.add_task(save_filtered_stocks_to_db, df_json)
        
        # 4. Kirim response ke Frontend langsung tanpa menunggu proses save DB selesai
        return {
            "status": "success",
            "message": "Filtering berhasil dilakukan.",
            "total_saham": len(daftar_saham),
            "data": df_json
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

