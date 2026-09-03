from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from src.backend.models.database import get_db
from src.backend.models.schemas.portfolio_schema import (
    ApiResponse,
    PortfolioGenerateRequest,
    PortfolioResponse,
)
from src.backend.controller.portfolio_controller import (
    MarketDataUnavailableError,
    generate_new_portfolio,
)

router = APIRouter()

@router.post("/generate", response_model=ApiResponse[PortfolioResponse])
def api_generate_portfolio(request: Request, body: PortfolioGenerateRequest, db: Session = Depends(get_db)):
    """
    Endpoint untuk membuat/mengenerate portofolio baru bagi pengguna.
    Menerima modal dan profil risiko, mengembalikan rekomendasi portofolio beserta alokasi
    dan narasi singkat. Data market live diambil dari app.state (dimuat saat startup).
    Respons dibungkus envelope seragam: {status, message, data}.
    """
    # Data market sudah dimuat di app.state saat startup oleh lifespan (app.py)
    market_data = request.app.state.market_data_today
    try:
        hasil = generate_new_portfolio(db, body, market_data)
    except MarketDataUnavailableError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ApiResponse(
        status="success",
        message="Portofolio berhasil digenerate.",
        data=hasil,
    )
