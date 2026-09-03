from fastapi import APIRouter, BackgroundTasks, HTTPException
from src.backend.models.schemas.market_schema import MarketFilterResponse
from src.backend.models.schemas.portfolio_schema import ApiResponse
from src.backend.controller.market_controller import (
    StockFilteringError,
    run_and_cache_stock_filtering,
)

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

