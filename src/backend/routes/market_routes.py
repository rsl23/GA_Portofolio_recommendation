from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from src.backend.models.database import get_db
from src.backend.models.schemas.market_schema import MarketFilterResponse
from src.backend.models.schemas.portfolio_schema import ApiResponse
from src.backend.controller.market_controller import (
    PortfolioNotFoundError,
    StockFilteringError,
    get_price_history,
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


@router.get("/price-history", response_model=ApiResponse[dict])
def price_history_endpoint(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Ambil histori harga harian (OHLCV) semua saham milik user, dari tanggal
    pembuatan portofolio terawal sampai tanggal data terbaru.
    Identitas user diambil dari JWT (sub). Data di-group per ticker agar
    langsung siap digambar chart (mis. performa portofolio vs IHSG).
    404 jika user belum memiliki portofolio.
    """
    try:
        hasil = get_price_history(db, user_id=current_user["sub"])
    except PortfolioNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ApiResponse(
        status="success",
        message="Histori harga berhasil diambil.",
        data=hasil,
    )

