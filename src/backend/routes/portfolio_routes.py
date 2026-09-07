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
    UpdateHargaBeliRequest,
)
from src.backend.controller.portfolio_controller import (
    ItemNotFoundError,
    MarketDataUnavailableError,
    PortfolioNotFoundError,
    StockNotFoundError,
    UserNotFoundError,
    generate_new_portfolio,
    get_active_portfolio,
    list_portfolio_history,
    update_harga_beli,
    get_portfolio_performance,
)
from src.backend.controller.market_controller import (
    get_price_history,
)
from src.backend.core.deps import get_current_user

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


@router.get("/price-history", response_model=ApiResponse[dict])
def price_history_endpoint(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Ambil histori harga harian (OHLCV + adj_close) semua saham milik user pada
    portofolio ACTIVE, dari tanggal pembuatan portofolio aktif sampai tanggal
    data terbaru, beserta harga IHSG (benchmark) dari idx_composite.
    Identitas user diambil dari JWT (sub). Data di-group per ticker agar
    langsung siap digambar chart (mis. performa portofolio vs IHSG).
    404 jika user belum memiliki portofolio aktif.
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


@router.get("/portofolio_performance", response_model=ApiResponse[dict])
def portofolio_performance_endpoint(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Hitung performa portofolio ACTIVE user vs IHSG dalam persentase return
    kumulatif harian, sejak tanggal portofolio aktif dibuat.
    Perhitungan berbasis kepemilikan lot: nilai portofolio harian =
    SUM(jumlah_lot x 100 lembar x adj_close), return = nilai_t / nilai_t0 - 1.
    404 jika user belum memiliki portofolio aktif / data harga belum tersedia.
    """
    try:
        hasil = get_portfolio_performance(db, user_id=current_user["sub"])
    except PortfolioNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ApiResponse(
        status="success",
        message="Performa portofolio berhasil dihitung.",
        data=hasil,
    )


@router.patch("/my-portofolio/items/{item_id}/harga-beli", response_model=ApiResponse[dict])
def update_harga_beli_endpoint(
    item_id: str,
    body: UpdateHargaBeliRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Ubah harga_beli (per lembar, IDR) milik user pada satu item portofolio ACTIVE.
    Kepemilikan divalidasi dari JWT — user hanya bisa mengubah item di
    portofolio aktif miliknya sendiri. total_investasi item ikut dihitung ulang:
    jumlah_lot x 100 lembar x harga_beli.
    """
    try:
        item = update_harga_beli(
            db,
            user_id=current_user["sub"],
            item_id=item_id,
            harga_beli=body.harga_beli,
        )
    except ItemNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ApiResponse(
        status="success",
        message="Harga beli berhasil diperbarui.",
        data={
            "item_id": str(item.id),
            "ticker": item.stock.ticker if item.stock else None,
            "jumlah_lot": item.jumlah_lot,
            "harga_acuan": item.harga_acuan,   # per lembar
            "harga_beli": item.harga_beli,     # per lembar (hasil edit)
            "total_investasi": item.total_investasi,
        },
    )
