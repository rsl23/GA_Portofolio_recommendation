from datetime import date
import logging
from threading import Lock

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from src.backend.models.database import get_db
from src.backend.models.schemas.market_schema import MarketFilterResponse
from src.backend.models.schemas.portfolio_schema import ApiResponse
from src.backend.controller.market_controller import (
    StockFilteringError,
    get_last_filter_update,
    run_and_cache_stock_filtering,
    run_daily_pipeline,
)
from src.backend.services.price_history_service import sync_market_data
from src.backend.core.deps import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)
_manual_pipeline_lock = Lock()


@router.post("/run-daily-pipeline", response_model=ApiResponse[dict])
def run_daily_pipeline_endpoint(request: Request):
    """Manually run the existing daily pipeline from the local machine only.

    This diagnostic endpoint intentionally has no JWT dependency. It must not
    be exposed through a public reverse proxy: the pipeline is expensive and
    refreshes shared database caches. The scheduler remains unchanged.
    """
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Endpoint pipeline manual hanya tersedia dari localhost.")
    if not _manual_pipeline_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Pipeline manual masih berjalan.")
    try:
        result = run_daily_pipeline()
        market_data = result.get("market_data")
        if market_data is None:
            raise HTTPException(status_code=503, detail="Pipeline selesai tanpa MarketData yang valid.")
        request.app.state.market_data_today = market_data
        return ApiResponse(
            status="success",
            message="Daily pipeline berhasil dijalankan secara manual.",
            data={
                "filtered_stocks": result["filtered_stocks"],
                "market_data_stocks": market_data.n_stocks,
                "sync_stats": result["sync_stats"],
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Daily pipeline manual gagal")
        raise HTTPException(status_code=500, detail=f"Daily pipeline gagal: {exc}") from exc
    finally:
        _manual_pipeline_lock.release()

@router.get("/filter-stocks", response_model=ApiResponse[MarketFilterResponse])
def filter_stocks_endpoint(
    background_tasks: BackgroundTasks,
    backtest: bool = False,
    date_ref: date | None = None,
):
    """
    Endpoint tipis: seluruh logika ada di market_controller.
    Query param `backtest=true` -> pipeline dijalankan dalam mode backtest
    (filter yang bergantung tanggal hari ini di-skip; lihat run_live_preprocessing).
    Respons dibungkus envelope seragam: {status, message, data}.
    """
    try:
        hasil = run_and_cache_stock_filtering(background_tasks, backtest=backtest, date_ref=date_ref)
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

