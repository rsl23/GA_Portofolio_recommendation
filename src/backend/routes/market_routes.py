from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from src.backend.models.database import get_db
from src.backend.models.schemas.market_schema import MarketFilterResponse
from src.backend.models.schemas.portfolio_schema import ApiResponse
from src.backend.controller.market_controller import (
    StockFilteringError,
    get_last_filter_update,
    run_and_cache_stock_filtering,
)
from src.backend.services.price_history_service import sync_market_data
from src.backend.core.deps import get_current_user

router = APIRouter()

@router.get("/filter-stocks", response_model=ApiResponse[MarketFilterResponse])
def filter_stocks_endpoint(background_tasks: BackgroundTasks):
    """
    Endpoint tipis: seluruh logika ada di market_controller.
    Respons dibungkus envelope seragam: {status, message, data}.
    """
    try:
        hasil = run_and_cache_stock_filtering(background_tasks)
    except StockFilteringError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ApiResponse(
        status="success",
        message="Filtering berhasil dilakukan.",
        data=hasil,
    )


@router.get("/filter-stocks/last-update", response_model=ApiResponse[dict])
def filter_stocks_last_update_endpoint(db: Session = Depends(get_db)):
    """
    Kembalikan tanggal/waktu terakhir tabel filtered_stock_cache diperbarui.
    Semua baris ditulis bersamaan oleh job filtering (truncate + insert),
    jadi updated_at sinkron di semua row — cukup ambil salah satu (MAX).
    Data berisi null jika tabel masih kosong (preprocessing belum pernah jalan).
    """
    last_update = get_last_filter_update(db)
    return ApiResponse(
        status="success",
        message="Tanggal update terakhir berhasil diambil.",
        data={
            "last_updated_at": last_update.isoformat() if last_update else None,
        },
    )


@router.post("/sync-prices", response_model=ApiResponse[dict])
def sync_prices_endpoint(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Trigger sinkronisasi histori harga harian (OHLCV via yfinance) untuk semua
    saham yang pernah dibeli user, sejak tanggal pembelian terawal hingga hari ini.
    Data di-upsert ke tabel market_data (aman dijalankan berulang).
    Dipanggil oleh job harian (cron/scheduler) atau manual oleh user login.
    """
    stats = sync_market_data(db)
    return ApiResponse(
        status="success",
        message="Sinkronisasi harga selesai.",
        data=stats,
    )

