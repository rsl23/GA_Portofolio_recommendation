from datetime import date
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from src.backend.models.database import get_db
from src.backend.models.users import User
from src.backend.core.deps import get_current_user
from src.backend.models.schemas.portfolio_schema import (
    ApiResponse,
    PortfolioGenerateRequest,
    PortfolioResponse,
    PortfolioHistoryItem,
    RebalanceSeriesGenerateRequest,
    RebalanceSeriesResponse,
    UpdateHargaBeliRequest,
    AddCapitalRequest,
    UpdateHoldingRequest,
)
from src.backend.controller.portfolio_controller import (
    ItemNotFoundError,
    MarketDataUnavailableError,
    PortfolioLineageConflictError,
    PortfolioNotFoundError,
    StockNotFoundError,
    UserNotFoundError,
    generate_new_portfolio,
    generate_rebalance_series,
    get_active_portfolio,
    get_all_my_portfolios,
    delete_portfolio_lineage,
    set_portfolio_list_activity,
    update_portfolio_identity,
    list_portfolio_history,
    update_harga_beli,
    get_portfolio_performance,
    add_live_capital,
    preview_bear_market,
    preview_bear_series,
)
from src.backend.controller.market_controller import (
    get_price_history,
)
from src.backend.core.deps import get_current_user

router = APIRouter()


@router.post("/generate/stream")
def api_generate_portfolio_stream(
    request: Request,
    body: PortfolioGenerateRequest,
    current_user: dict = Depends(get_current_user),
):
    from src.backend.models.database import SessionLocal
    from src.backend.services.portfolio_progress import portfolio_progress_response

    def operation():
        # Create and close the session on the worker that actually uses it.
        with SessionLocal() as db:
            return _generate_portfolio_response(request, body, db=db, current_user=current_user)

    return portfolio_progress_response(operation)


@router.post("/generate-rebalance-series/stream")
def api_generate_rebalance_series_stream(
    body: RebalanceSeriesGenerateRequest,
    current_user: dict = Depends(get_current_user),
):
    from src.backend.models.database import SessionLocal
    from src.backend.services.portfolio_progress import portfolio_progress_response

    def operation():
        with SessionLocal() as db:
            return _generate_rebalance_series_response(body, db=db, current_user=current_user)

    return portfolio_progress_response(operation)


class BearPolicyUpdate(BaseModel):
    policy: Literal["manual", "always_follow", "always_full"]


class BearImpactRequest(BaseModel):
    start_date: date | None = None
    end_date: date | None = None
    risk_profile: Literal["Conservative", "Moderate", "Aggressive", "Konservatif", "Moderat", "Agresif"] | None = None
    fallback_buy_fee: float = Field(.0015,ge=0,le=.05,allow_inf_nan=False)
    fallback_sell_fee: float = Field(.0025,ge=0,le=.05,allow_inf_nan=False)


@router.post("/bear-impact/{portfolio_id}", response_model=ApiResponse[dict])
def bear_impact_endpoint(portfolio_id: str, body: BearImpactRequest,
    db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    from src.backend.models.portofolios import Portofolio
    from src.backend.services.bear_impact_service import evaluate_bear_impact
    from src.backend.services.bear_allocation_replay import AllocationReplayUnavailable
    if body.start_date and body.end_date and body.start_date>body.end_date:
        raise HTTPException(422,'Tanggal mulai melewati tanggal akhir.')
    try:
        performance = get_portfolio_performance(db,user_id=current_user['sub'],portfolio_id=portfolio_id,
            end_date=body.end_date,backtest=True)
    except PortfolioNotFoundError as exc:
        raise HTTPException(404,str(exc)) from exc
    portfolio = db.query(Portofolio).filter(Portofolio.id==UUID(portfolio_id)).first()
    try:
        result = evaluate_bear_impact(db,portfolio,performance,**body.model_dump())
    except AllocationReplayUnavailable as exc:
        result = dict(status='unavailable',reason=str(exc))
    return ApiResponse(data=result)


@router.get("/evaluation/{portfolio_id}", response_model=ApiResponse[dict])
def portfolio_evaluation_endpoint(
    portfolio_id: str, start_date: date | None = None, end_date: date | None = None,
    backtest: bool = False, include_bear: bool = False,
    bear_scope: Literal["portfolio", "market_history"] = "portfolio",
    comparison_profile: Literal["Conservative", "Moderate", "Aggressive", "Konservatif", "Moderat", "Agresif"] | None = None,
    db: Session = Depends(get_db), current_user: dict = Depends(get_current_user),
):
    from src.backend.models.portofolios import Portofolio
    from src.backend.services.risk_evaluation import evaluate_risk, snapshot_profile_cap
    from src.backend.services.bear_evaluation import historical_bear_evaluation
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="Tanggal mulai melewati tanggal akhir.")
    try:
        performance = get_portfolio_performance(db, user_id=current_user["sub"],
            portfolio_id=portfolio_id, end_date=end_date, backtest=backtest)
    except PortfolioNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # Ownership is enforced by the same performance ledger used by the chart.
    portfolio = db.query(Portofolio).filter(Portofolio.id == UUID(portfolio_id)).first()
    try:
        selected_profile = comparison_profile or portfolio.risk_profile
        saved_cap = snapshot_profile_cap(getattr(portfolio, "rebalance_metadata", None), selected_profile)
        risk = evaluate_risk(performance, selected_profile, start_date, end_date,
                             profile_cap=saved_cap)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = {"risk": risk, "bear": None}
    if include_bear and risk["start_date"] and risk["end_date"]:
        try:
            result["bear"] = historical_bear_evaluation(
                date.fromisoformat(risk["start_date"]), date.fromisoformat(risk["end_date"]), bear_scope)
        except (OSError, ValueError, KeyError) as exc:
            result["bear"] = {"status": "unavailable", "reason": str(exc)}
    return ApiResponse(data=result)


@router.get("/bear-policy", response_model=ApiResponse[dict])
def api_get_bear_policy(db: Session = Depends(get_db),
                        current_user: dict = Depends(get_current_user)):
    user = db.query(User).filter(User.id == UUID(current_user["sub"])).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User tidak ditemukan.")
    return ApiResponse(data={"policy": user.bear_allocation_policy})


@router.patch("/bear-policy", response_model=ApiResponse[dict])
def api_update_bear_policy(body: BearPolicyUpdate, db: Session = Depends(get_db),
                           current_user: dict = Depends(get_current_user)):
    user = db.query(User).filter(User.id == UUID(current_user["sub"])).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User tidak ditemukan.")
    user.bear_allocation_policy = body.policy
    db.commit()
    return ApiResponse(data={"policy": body.policy})


@router.get("/bear-market-preview", response_model=ApiResponse[dict])
def api_preview_bear_market(
    request: Request, risk_profile: str, backtest: bool = False, date_ref: date | None = None,
    current_user: dict = Depends(get_current_user),
):
    try:
        # An old RAM snapshot must not pin all new previews to its old date.
        return ApiResponse(data=preview_bear_market(risk_profile, backtest, date_ref))
    except (ValueError, MarketDataUnavailableError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/bear-market-series-preview", response_model=ApiResponse[dict])
def api_preview_bear_series(
    body: RebalanceSeriesGenerateRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        return ApiResponse(data=preview_bear_series(
            body.start_date, body.end_date, body.interval_months, body.risk_profile,
        ))
    except (ValueError, MarketDataUnavailableError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/my-portfolio/add-capital", response_model=ApiResponse[PortfolioResponse])
def api_add_live_capital(
    body: AddCapitalRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Antrekan setoran pada portofolio live aktif; dana dipakai saat rebalance."""
    try:
        result = add_live_capital(db, current_user["sub"], body.amount, portfolio_id=body.portfolio_id)
    except (PortfolioNotFoundError, UserNotFoundError) as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Gagal mencatat setoran: {exc}") from exc
    return ApiResponse(message="Setoran dicatat dan tersedia pada rebalance live berikutnya.", data=result)

@router.post("/generate", response_model=ApiResponse[PortfolioResponse])
def api_generate_portfolio(
    request: Request,
    body: PortfolioGenerateRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Buat portofolio; Accept: text/event-stream mengaktifkan progres langsung."""
    # Negotiate progress on the established URL. An older running backend can
    # still return JSON to the same request, without a 404 or a second mutation.
    if "text/event-stream" in getattr(request, "headers", {}).get("accept", ""):
        return api_generate_portfolio_stream(request, body, current_user=current_user)
    return _generate_portfolio_response(request, body, db, current_user)


def _generate_portfolio_response(request, body, db, current_user):
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
    # Live checks the shared snapshot against the latest completed WIB cutoff.
    # Backtest remains isolated and reads its historical source in the controller.
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

    try:
        market_data = None
        if not backtest:
            from src.gaengine.progress import report_progress
            report_progress("market", "Menyiapkan data pasar",
                            "Memeriksa ketersediaan harga, fundamental, dan suku bunga acuan.")
            from src.backend.services.live_market_snapshot import get_live_market_snapshot
            try:
                snapshot_options = ({"profile_policy": body.profile_policy}
                                    if body.profile_policy != "legacy" else {})
                market_data = get_live_market_snapshot(request.app.state, **snapshot_options)
            except ValueError as exc:
                raise MarketDataUnavailableError(str(exc)) from exc
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
    except ValueError as e:
        # Kondisi request tidak valid (mis. rebalance tanpa portofolio aktif
        # pada mode tsb, atau date_ref backtest rebalance lebih awal dari
        # portofolio lama) -> 400, bukan 500.
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
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
            else "Portofolio live berhasil dibuat. Lot dan average beli dapat disesuaikan di MyPortfolio."
        ),
        data=hasil,
    )


@router.post("/generate-rebalance-series", response_model=ApiResponse[RebalanceSeriesResponse])
def api_generate_rebalance_series(
    body: RebalanceSeriesGenerateRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """Jalankan rebalance berantai, dengan pilihan respons JSON atau progres SSE."""
    if request is not None and "text/event-stream" in request.headers.get("accept", ""):
        return api_generate_rebalance_series_stream(body, current_user=current_user)
    return _generate_rebalance_series_response(body, db, current_user)


def _generate_rebalance_series_response(body, db, current_user):
    """
    Endpoint BACKTEST-SERIES: menjalankan generate awal + rebalance BERANTAI
    berulang pada rentang tanggal, mis. 2022-01-01 s/d 2026-12-31 tiap 2 bulan
    -> 30 portofolio (bukan angka pasti -- mengikuti rentang & interval).

    Alur di controller (generate_rebalance_series):
      1. base_portfolio_id -> gunakan snapshot tersimpan tanpa mengulang GA awal
      2. tiap +interval   -> rebalance dari snapshot terakhir yang berhasil
         (rebalance=true; NAV & lots diwarisi berantai, parent_portofolio_id
         membentuk rantai)
    Semua langkah memakai data HISTORIS (khusus backtest) -- portofolio LIVE
    tidak tersentuh. Portofolio terakhir = "active_backtest"; sisanya menjadi
    "replaced_backtest" (tetap tersimpan & bisa diambil via
    GET /my-portfolio/all?backtest=true).

    Body (JSON):
      - budget (float, wajib), risk_profile (str, wajib)
      - base_portfolio_id (UUID, wajib): snapshot backtest terbaru dalam rangkaian
      - start_date (YYYY-MM-DD, wajib): tanggal snapshot tersebut
      - end_date (YYYY-MM-DD, wajib, inklusif): tanggal terakhir series
      - interval_months (int, default 2, 1-12): jeda antar rebalance
      Contoh: {"base_portfolio_id": "...", "budget": 10000000, "risk_profile": "Moderat",
               "start_date": "2022-01-03", "end_date": "2026-12-31",
               "interval_months": 2}

    Respons {status, message, data} dengan data = RebalanceSeriesResponse:
      - scheduled_dates : tanggal rebalance baru yang dijadwalkan
      - portfolios      : snapshot lama dan baru, urut kronologis
      - skipped         : tanggal yang dilewati (data historis tak tersedia)
                          beserta alasannya

    CATATAN: series dijalankan SEKUENSIAL di satu request (bukan paralel)
    karena tiap rebalance bergantung pada hasil langkah sebelumnya. Tiap
    langkah = 1x GA penuh, jadi series panjang bisa memakan waktu lama --
    gunakan timeout HTTP yang memadai di sisi klien.

    Endpoint: POST /api/v1/portfolios/generate-rebalance-series
    """
    try:
        hasil = generate_rebalance_series(
            db,
            body,
            user_id=current_user["sub"],
        )
    except UserNotFoundError as e:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(e))
    except MarketDataUnavailableError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except StockNotFoundError as e:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        # Rentang tanggal salah / jumlah langkah melebihi batas / rebalance
        # tanpa portofolio aktif -> 400 Bad Request (bukan 500).
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Gagal menyimpan portofolio ke database: {e}"
        )
    return ApiResponse(
        status="success",
        message=(
            f"Rebalance series selesai: {len(hasil.scheduled_dates) - len(hasil.skipped)} rebalance "
            f"berhasil dari {len(hasil.scheduled_dates)} tanggal terjadwal"
            + (
                f" ({len(hasil.skipped)} tanggal dilewati)."
                if hasil.skipped
                else "."
            )
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
    GET /my-portfolio (PortfolioResponse lengkap dengan daftar alokasi item).

    Query param:
      - backtest (bool, default false): false = portofolio live, true = portofolio backtest.

    404 hanya jika user pada token tidak valid / tidak ada di database.
    User yang belum punya portofolio pada mode tsb -> data berisi list kosong [].

    Endpoint: GET /api/v1/portfolios/my-portfolio/all?backtest=false
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


class PortfolioActivityUpdate(BaseModel):
    is_active: bool


class PortfolioIdentityUpdate(BaseModel):
    name_portofolio: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)


@router.patch("/my-portfolio/{portfolio_id}/identity", response_model=ApiResponse[PortfolioResponse])
def api_update_portfolio_identity(
    portfolio_id: UUID,
    body: PortfolioIdentityUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    name = body.name_portofolio.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Nama portofolio tidak boleh kosong.")
    description = body.description.strip() or None if body.description is not None else None
    try:
        result = update_portfolio_identity(db, current_user["sub"], str(portfolio_id), name, description)
    except (PortfolioNotFoundError, UserNotFoundError) as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Gagal menyimpan nama portofolio.") from exc
    return ApiResponse(message="Nama dan deskripsi rangkaian portofolio diperbarui.", data=result)


@router.patch("/my-portfolio/{portfolio_id}/activity", response_model=ApiResponse[dict])
def api_update_portfolio_activity(
    portfolio_id: UUID,
    body: PortfolioActivityUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    try:
        result = set_portfolio_list_activity(db, current_user["sub"], str(portfolio_id), body.is_active)
    except (PortfolioNotFoundError, UserNotFoundError) as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Gagal mengubah status portofolio.") from exc
    return ApiResponse(status="success", message="Status portofolio diperbarui.", data=result)


@router.delete("/my-portfolio/{portfolio_id}", response_model=ApiResponse[dict])
def api_delete_portfolio(
    portfolio_id: UUID,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Hapus portofolio milik user, item, dan seluruh rebalance turunannya."""
    try:
        result = delete_portfolio_lineage(db, current_user["sub"], str(portfolio_id))
    except (PortfolioNotFoundError, UserNotFoundError) as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PortfolioLineageConflictError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Gagal menghapus portofolio.") from exc
    return ApiResponse(
        message=f"{result['deleted_portfolios']} portofolio dan seluruh item terkait berhasil dihapus.",
        data=result,
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
    vs IHSG dalam persentase return kumulatif harian, dari execution_date
    snapshot awal sampai query param `end_date`
    (YYYY-MM-DD) — atau sampai data terakhir bila `end_date` tidak diberikan.
    Perhitungan memakai rantai snapshot parent_portofolio_id dan selisih lot
    BUY/SELL/HOLD. date_ref adalah tanggal pembentukan/jadwal; execution_date
    adalah hari komposisi baru mulai berlaku. Window item [start_date, end_date)
    membatasi masa berlaku snapshot. NAV = kas + lot aktif x 100 x raw Close
    + piutang dividen bruto. Dividen diakui sejak ex-date dari posisi sebelumnya;
    pembayaran/reinvestasi tidak diasumsikan tanpa tanggal pembayaran.
    Untuk seluruh rangkaian rebalance, gunakan ID snapshot terakhir.

    Return tambahan (persen): twr_return_pct = total_return_pct (kompatibel),
    cash_return = NAV akhir - modal awal - setoran (Rupiah), cash_return_pct
    = cash_return / seluruh dana masuk. money_weighted_return_pct adalah
    XIRR yang dikonversi ke return periode; waktu/jumlah setoran diperhitungkan.
    money_weighted_return_annualized_pct adalah XIRR tahunan dan
    annualized_twr_return_pct adalah TWR majemuk per tahun (Actual/365).
    Kedua angka tahunan null bila performance_days < 365. Setoran live yang
    menunggu rebalance belum menjadi arus dana dalam perhitungan ini.

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
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
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


@router.patch("/my-portfolio/{portfolio_id}/holdings/{ticker}", response_model=ApiResponse[PortfolioResponse])
def update_holding_endpoint(portfolio_id: str, ticker: str, body: UpdateHoldingRequest,
                            db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    from src.backend.services.portfolio_holdings import update_live_holding
    try:
        result = update_live_holding(db, current_user["sub"], portfolio_id, ticker, body)
        return ApiResponse(message="Lot dan average beli berhasil diperbarui.", data=result)
    except ItemNotFoundError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Perubahan belum tersimpan. Silakan coba lagi.") from exc
