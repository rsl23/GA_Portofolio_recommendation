from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from src.backend.models.database import get_db
from src.backend.core.deps import get_current_user
from src.backend.models.schemas.portfolio_schema import (
    ApiResponse,
    PortfolioGenerateRequest,
    PortfolioResponse,
)
from src.backend.controller.portfolio_controller import (
    MarketDataUnavailableError,
    UserNotFoundError,
    generate_new_portfolio,
)

router = APIRouter()

@router.post("/generate", response_model=ApiResponse[PortfolioResponse])
def api_generate_portfolio(
    request: Request,
    body: PortfolioGenerateRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Endpoint terproteksi untuk membuat/mengenerate portofolio baru.
    Wajib header Authorization: Bearer <access_token> — signature, exp, dan
    type token divalidasi oleh get_current_user (core/deps.py).
    user_id diambil dari payload JWT (sub), BUKAN dari body, agar user
    tidak dapat menyimpan portofolio atas nama user lain.
    Respons dibungkus envelope seragam: {status, message, data}.
    """
    # Data market sudah dimuat di app.state saat startup oleh lifespan (app.py)
    market_data = request.app.state.market_data_today
    try:
        hasil = generate_new_portfolio(
            db, body, user_id=current_user["sub"], market_data=market_data
        )
    except UserNotFoundError as e:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except MarketDataUnavailableError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Gagal menyimpan portofolio ke database: {e}"
        )
    return ApiResponse(
        status="success",
        message="Portofolio berhasil digenerate.",
        data=hasil,
    )
