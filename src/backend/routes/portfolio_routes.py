from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from src.backend.models.database import get_db
from src.backend.core.deps import get_current_user
from src.backend.models.schemas.portfolio_schema import (
    ApiResponse,
    PortfolioGenerateRequest,
    PortfolioResponse,
    PortfolioHistoryItem,
)
from src.backend.controller.portfolio_controller import (
    MarketDataUnavailableError,
    PortfolioNotFoundError,
    StockNotFoundError,
    UserNotFoundError,
    generate_new_portfolio,
    get_active_portfolio,
    list_portfolio_history,
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
    except StockNotFoundError as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(e))
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


@router.get("/my-portofolio", response_model=ApiResponse[PortfolioResponse])
def api_my_portfolio(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Ambil portofolio AKTIF terbaru milik user (identitas diambil dari JWT,
    bukan dari parameter). Dipakai halaman "My Portfolio".
    404 jika user belum pernah generate portofolio.
    """
    try:
        hasil = get_active_portfolio(db, user_id=current_user["sub"])
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PortfolioNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ApiResponse(
        status="success",
        message="Portofolio aktif berhasil diambil.",
        data=hasil,
    )


@router.get("/my-portofolio/history", response_model=ApiResponse[list[PortfolioHistoryItem]])
def api_my_portfolio_history(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Ambil seluruh histori portofolio milik user, terbaru dulu.
    Portofolio lama yang sudah superseded berstatus "replaced" ikut tampil di sini.
    """
    try:
        hasil = list_portfolio_history(db, user_id=current_user["sub"])
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ApiResponse(
        status="success",
        message="Histori portofolio berhasil diambil.",
        data=hasil,
    )
