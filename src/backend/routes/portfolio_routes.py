from datetime import date

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
    get_all_my_portfolios,
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

    Body (JSON):
      - budget (float, wajib), risk_profile (str, wajib)
      - backtest (bool, default false): true = jalankan GA memakai data
        HISTORIS LOKAL (bukan data pasar hari ini). Lihat data_loader.
      - date_ref (YYYY-MM-DD): WAJIB bila backtest=true; tanggal acuan
        simulasi, mis. {"backtest": true, "date_ref": "2024-06-28"}

    Endpoint: POST /api/v1/portfolios/generate

    Catatan: hasil backtest DISIMPAN sebagai portofolio berstatus
    "active_backtest" (portofolio live berstatus "active" tidak tersentuh;
    portofolio backtest lama hanya berubah status menjadi "replaced_backtest").
    Respons memakai envelope seragam: {status, message, data}.
    """
    # Data market LIVE sudah dimuat di app.state saat startup oleh lifespan
    # (app.py). Mode backtest TIDAK memakai cache live ini — controller akan
    # merakit ulang MarketData dari data historis lokal via data_loader.
    # getattr defensif: bila lifespan belum sempat mengisi app.state (startup
    # parsial / scheduler gagal), controller akan merakit MarketData live sendiri.
    backtest = body.backtest

    # date_ref sudah divalidasi & dikonversi 'YYYY-MM-DD' -> datetime.date oleh
    # Pydantic (lihat PortfolioGenerateRequest), jadi bisa dipakai langsung.
    # Konversi defensif tetap disiapkan bila nilainya datang sebagai string.
    date_ref = body.date_ref
    if isinstance(date_ref, str):
        try:
            date_ref = date.fromisoformat(date_ref)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Format date_ref tidak valid: {date_ref!r} (gunakan YYYY-MM-DD).",
            )

    market_data = None if backtest else getattr(request.app.state, "market_data_today", None)
    try:
        hasil = generate_new_portfolio(
            db,
            body,
            user_id=current_user["sub"],
            market_data=market_data,
            backtest=backtest,
            date_ref=date_ref,
        )
    except UserNotFoundError as e:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except MarketDataUnavailableError as e:
        db.rollback()
        # Param backtest yang salah/kurang -> 400 Bad Request; kegagalan data
        # live tetap 500 karena bukan kesalahan pemanggil.
        raise HTTPException(status_code=400 if backtest else 500, detail=str(e))
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
        message=(
            f"Portofolio backtest per {date_ref} berhasil digenerate (status active_backtest)."
            if backtest
            else "Portofolio berhasil digenerate."
        ),
        data=hasil,
    )


@router.get("/my-portfolio", response_model=ApiResponse[PortfolioResponse])
def api_my_portfolio(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    backtest: bool = False,
):
    """
    Ambil portofolio AKTIF terbaru milik user (identitas diambil dari JWT,
    bukan dari parameter). Dipakai halaman "My Portfolio".

    Query param:
      - backtest (bool, default false): false = portofolio LIVE (status "active"),
        true = portofolio SIMULASI terbaru (status "active_backtest").

    404 jika user belum pernah generate portofolio pada mode tsb.
    """
    try:
        hasil = get_active_portfolio(db, user_id=current_user["sub"], backtest=backtest)
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PortfolioNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ApiResponse(
        status="success",
        message=(
            "Portofolio backtest aktif berhasil diambil."
            if backtest
            else "Portofolio aktif berhasil diambil."
        ),
        data=hasil,
    )


@router.get("/my-portfolio/history", response_model=ApiResponse[list[PortfolioHistoryItem]])
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


@router.get("/my-portfolio/all", response_model=ApiResponse[list[PortfolioResponse]])
def api_my_portfolio_all(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    backtest: bool = False,
):
    """
    Ambil SEMUA portofolio milik user (identitas dari JWT) sesuai MODE yang
    dipilih lewat query param `backtest`, pemisahannya memakai kolom
    `status_portofolio` pada tabel portofolios:

      - backtest=false (default) -> portofolio LIVE
            status "active" (portofolio live terbaru) +
            status "replaced" (portofolio live yang sudah digantikan)
      - backtest=true            -> portofolio BACKTEST
            status "active_backtest" (simulasi terbaru) +
            status "replaced_backtest" (simulasi yang sudah digantikan)

    Karena mode dibedakan dari status, portofolio live tidak akan pernah
    tercampur dengan hasil simulasi backtest. Portofolio "replaced_*" tetap
    tersimpan di database dan ikut tampil di sini (tidak dihapus).
    Urutan: created_at terbaru dulu; respons memakai schema yang SAMA dengan
    GET /my-portofolio (PortfolioResponse lengkap dengan daftar alokasi item).

    Query param:
      - backtest (bool, default false): false = portofolio live, true = portofolio backtest.

    404 hanya jika user pada token tidak valid / tidak ada di database.
    User yang belum punya portofolio pada mode tsb -> data berisi list kosong [].

    Endpoint: GET /api/v1/portfolios/my-portofolio/all?backtest=false
    """
    try:
        hasil = get_all_my_portfolios(
            db,
            user_id=current_user["sub"],
            backtest=backtest,
        )
    except UserNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ApiResponse(
        status="success",
        message=(
            f"Seluruh {len(hasil)} portofolio backtest berhasil diambil."
            if backtest
            else f"Seluruh {len(hasil)} portofolio live berhasil diambil."
        ),
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


@router.get("/portfolio_performance/{portfolio_id}", response_model=ApiResponse[dict])
def portofolio_performance_endpoint(
    portfolio_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    end_date: date | None = None,
    backtest: bool = False,
):
    """
    Hitung performa portofolio TERTENTU (path param {portfolio_id}) milik user
    vs IHSG dalam persentase return kumulatif harian, dari portofolio.date_ref
    (backtest: tanggal simulasi; live: sekarang) sampai query param `end_date`
    (YYYY-MM-DD) — atau sampai data terakhir bila `end_date` tidak diberikan.
    Perhitungan berbasis kepemilikan lot PER ITEM (window start_date/end_date
    milik item dihormati), sehingga perubahan kepemilikan akibat rebalance
    ikut terhitung: nilai portofolio harian = SUM(lot aktif x 100 lembar x
    adj_close), return = nilai_t / nilai_t0 - 1.

    Path param:
      - portfolio_id (UUID, wajib): ID portofolio yang akan dihitung.
        Wajib milik user pada JWT — portofolio user lain -> 404.

    Query param:
      - end_date (YYYY-MM-DD, opsional): batas akhir perhitungan (inklusif).
      - backtest (bool, default false): true = hitung performa portofolio backtest

    404 jika portofolio tidak ditemukan / bukan milik user / data harga
    belum tersedia pada rentang tsb.
    """
    try:
        hasil = get_portfolio_performance(
            db,
            user_id=current_user["sub"],
            portfolio_id=portfolio_id,
            end_date=end_date,
            backtest=backtest,
        )
    except PortfolioNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ApiResponse(
        status="success",
        message="Performa portofolio berhasil dihitung.",
        data=hasil,
    )


@router.patch("/my-portfolio/items/{item_id}/harga-beli", response_model=ApiResponse[dict])
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
