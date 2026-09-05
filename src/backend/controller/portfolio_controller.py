import logging
from uuid import uuid4, UUID

import numpy as np
from sqlalchemy.orm import Session

from src.backend.models.users import User
from src.backend.models.portofolios import Portofolio
from src.backend.models.portofolio_items import PortofolioItem
from src.backend.models.schemas.portfolio_schema import (
    PortfolioGenerateRequest,
    PortfolioResponse,
    PortfolioHistoryItem,
    PortfolioItem,
)
from src.gaengine.data_loader_live import build_market_data
from src.gaengine.engine import GeneticEngine
from src.gaengine.ga_config import GAConfig

logger = logging.getLogger(__name__)


class MarketDataUnavailableError(Exception):
    """Exception domain: MarketData gagal dibentuk / cache saham kosong."""


class UserNotFoundError(Exception):
    """Exception domain: user pada payload JWT tidak ditemukan di database."""


class PortfolioNotFoundError(Exception):
    """Exception domain: user belum memiliki portofolio aktif."""


def generate_new_portfolio(
    db: Session,
    request: PortfolioGenerateRequest,
    user_id: str,
    market_data=None,
) -> PortfolioResponse:
    """
    Controller Otak Utama:
    1. Memakai MarketData live (bisa reused dari app.state atau dibangun ulang).
    2. Menjalankan Algoritma Genetika sesuai modal & profil risiko pengguna.
    3. Menyimpan hasil ke tabel portofolios + portofolio_items.
    """
    logger.info(
        "Menjalankan GA untuk user %s dengan profil %s dan modal Rp%.2f",
        user_id, request.risk_profile, request.budget,
    )

    # 0. Validasi user dari payload JWT (sub). Lempar error jika sudah terhapus.
    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise UserNotFoundError(f"User ID pada token tidak valid: {user_id}") from e
    user = db.query(User).filter(User.id == user_uuid).first()
    if user is None:
        raise UserNotFoundError("User pada token tidak ditemukan di database.")

    # 1. Siapkan MarketData. Kalau tidak diberi (None) dari luar, bangun sendiri.
    if market_data is None:
        logger.info("Membentuk MarketData menggunakan data LIVE...")
        market_data = build_market_data(min_price=50.0)

    if market_data is None or market_data.n_stocks == 0:
        raise MarketDataUnavailableError(
            "Gagal membentuk MarketData. Pastikan tabel filtered_stock_cache sudah terisi."
        )

    # 2. Konfigurasi GA mengikuti profil & modal dari payload request
    config = GAConfig(
        population_size=200,        # jumlah kromosom
        generations=300,            # jumlah generasi evolusi
        budget=request.budget,      # modal pengguna (IDR)
        risk_profile=request.risk_profile,
        min_stocks=3,
        max_stocks=10,
        risk_free_rate=market_data.risk_free_rate,
        seed=42,
        data=market_data,
    )

    # 3. Evolusi genetika
    logger.info("=== MEMULAI EVOLUSI GENETIKA ===")
    engine = GeneticEngine(config)
    solution = engine.run(verbose=True)

    # 4. Susun laporan hasil
    prices = market_data.prices_per_lot
    codes = market_data.stock_codes
    lots = solution.lots
    alloc = lots * prices
    total = float(alloc.sum())

    rows = sorted(
        ((codes[i], int(lots[i]), float(prices[i])) for i in range(len(codes))),
        key=lambda r: r[1] * r[2],
        reverse=True,
    )
    allocations = [
        {
            "ticker": code,
            "lots": lot,
            "price_per_lot": price,
            "allocation": float(lot * price),
            "weight": float(lot * price / total) if total else 0.0,
        }
        for code, lot, price in rows
        if lot > 0
    ]

    # Estimasi return tahunan dari alokasi aktual (bobot = alokasi / total)
    expected_return = None
    if total > 0 and market_data.returns.shape[0] >= solution.n_active:
        aw = alloc / total
        p_daily = (market_data.returns * aw[:, None]).sum(axis=0)
        p_daily = p_daily[~np.isnan(p_daily) & ~np.isinf(p_daily)]
        if p_daily.size:
            expected_return = float(p_daily.mean() * 252.0)

    narasi = (
        f"Portofolio dengan profil risiko {request.risk_profile} memilih {solution.n_active} "
        f"saham dari {market_data.n_stocks} kandidat. Total dana terpakai "
        f"Rp{total:,.0f} dari budget Rp{request.budget:,.0f}. Skor fitness yang dicapai "
        f"{solution.fitness:.4f} dengan Sharpe ratio {solution.sharpe_ratio:.3f}. "
        f"Penurunan maksimum (max drawdown) tercatat {solution.max_drawdown:.2%} "
        f"dan korelasi rata-rata antar saham {solution.avg_correlation:.3f}. "
        f"Rekomendasi ini dihasilkan Algoritma Genetika."
    )

    # 5. Supersede (Opsi B): tandai portofolio aktif lama milik user sebagai "replaced".
    #    Riwayat tetap tersimpan dan bisa diambil via /my-portofolio/history.
    db.query(Portofolio).filter(
        Portofolio.user_id == user_uuid,
        Portofolio.status_portofolio == "active",
    ).update({"status_portofolio": "replaced"}, synchronize_session=False)

    # 6. Simpan hasil ke database: portofolios + portofolio_items
    portofolio = Portofolio(
        user_id=user_uuid,
        budget=request.budget,
        total_terpakai=total,
        sisa_budget=request.budget - total,
        fitness_score=float(solution.fitness),
        sharpe_ratio=float(solution.sharpe_ratio),
        max_drawdown=float(solution.max_drawdown),
        avg_correlation=float(solution.avg_correlation),
        skor_fundamental=float(solution.skor_fundamental),
        risk_profile=request.risk_profile,
        # bobot pengali fitness persis seperti yang dipakai GA saat evaluasi
        mdd_lambda=float(config.lambda_mdd),           # lambda MDD per profil risiko
        avg_korelasi_gamma=float(config.correlation_penalty),  # gamma = 0.5
        funda_alpha=float(config.fundamental_bonus),   # alpha = 0.3
        narasi_llm=narasi,
        status_portofolio="active",
        # field rebalance: ini portofolio baru, bukan hasil rebalance
        is_rebalance=False,
        parent_portofolio_id=None,
        turnover_rate=None,          # turnover hanya relevan saat rebalance
        turnover_penalty_beta=None,  # beta hanya dipakai saat rebalance
    )
    db.add(portofolio)
    db.flush()  # dapatkan portofolio.id sebelum commit (id di-generate default-nya)

    items = [
        PortofolioItem(
            portofolio_id=portofolio.id,
            ticker=ticker,
            bobot_persentase=float((lot * price / total) * 100.0) if total else 0.0,
            jumlah_lot=lot,
            harga_acuan=price,
            total_investasi=float(lot * price),
            action_type="buy",  # portofolio baru: semua posisi adalah pembelian awal
        )
        for ticker, lot, price in rows
        if lot > 0
    ]
    db.add_all(items)
    db.commit()
    db.refresh(portofolio)
    logger.info("Portofolio %s berhasil disimpan dengan %d item.", portofolio.id, len(items))

    return PortfolioResponse(
        id=str(portofolio.id),
        user_id=str(user_uuid),
        fitness_score=float(solution.fitness),
        sharpe_ratio=float(solution.sharpe_ratio),
        expected_return=expected_return,
        max_drawdown=float(solution.max_drawdown),
        avg_correlation=float(solution.avg_correlation),
        skor_fundamental=float(solution.skor_fundamental),
        total_terpakai=total,
        sisa_budget=request.budget - total,
        n_active=int(solution.n_active),
        allocated_budget_ok=bool(solution.budget_ok),
        risk_profile=request.risk_profile,
        status_portofolio=portofolio.status_portofolio,
        created_at=portofolio.created_at,
        budget=request.budget,
        allocations=allocations,
        narasi_llm=narasi,
    )


def _parse_user_uuid(user_id: str, db: Session) -> UUID:
    """Validasi user_id dari JWT: harus UUID valid dan ada di tabel users."""
    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise UserNotFoundError(f"User ID pada token tidak valid: {user_id}") from e
    if db.query(User).filter(User.id == user_uuid).first() is None:
        raise UserNotFoundError("User pada token tidak ditemukan di database.")
    return user_uuid


def _to_item_response(item: PortofolioItem) -> PortfolioItem:
    """Konversi PortofolioItem (ORM) ke schema PortfolioItem (response API)."""
    total = item.total_investasi or 0.0
    return PortfolioItem(
        ticker=item.ticker,
        lots=item.jumlah_lot,
        price_per_lot=item.harga_acuan,
        allocation=total,
        weight=(item.bobot_persentase or 0.0) / 100.0,
    )


def get_active_portfolio(db: Session, user_id: str) -> PortfolioResponse:
    """
    Ambil portofolio AKTIF terbaru milik user (dari payload JWT).
    Ordering: created_at terbaru, dengan id sebagai tiebreaker.
    Respons memakai schema yang SAMA dengan hasil generate (PortfolioResponse):
    field yang tidak disimpan di DB (expected_return) bernilai None,
    n_active dihitung dari jumlah item, allocated_budget_ok dihitung ulang.
    Raises: UserNotFoundError jika user tidak valid / tidak ada.
            PortfolioNotFoundError jika user belum pernah generate.
    """
    user_uuid = _parse_user_uuid(user_id, db)
    portofolio = (
        db.query(Portofolio)
        .filter(
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio == "active",
        )
        .order_by(Portofolio.created_at.desc(), Portofolio.id.desc())
        .first()
    )
    if portofolio is None:
        raise PortfolioNotFoundError("User belum memiliki portofolio aktif.")

    items = (
        db.query(PortofolioItem)
        .filter(PortofolioItem.portofolio_id == portofolio.id)
        .order_by(PortofolioItem.total_investasi.desc())
        .all()
    )

    # Hitung ulang field yang tidak disimpan di DB
    total_terpakai = portofolio.total_terpakai
    allocated_budget_ok = (
        (total_terpakai <= portofolio.budget)
        if total_terpakai is not None
        else None
    )

    return PortfolioResponse(
        id=str(portofolio.id),
        user_id=str(portofolio.user_id),
        fitness_score=portofolio.fitness_score,
        sharpe_ratio=portofolio.sharpe_ratio,
        expected_return=None,  # tidak disimpan di DB (hanya dihitung saat generate)
        max_drawdown=portofolio.max_drawdown,
        avg_correlation=portofolio.avg_correlation,
        skor_fundamental=portofolio.skor_fundamental,
        total_terpakai=total_terpakai,
        sisa_budget=portofolio.sisa_budget,
        n_active=len(items),
        allocated_budget_ok=allocated_budget_ok,
        risk_profile=portofolio.risk_profile,
        status_portofolio=portofolio.status_portofolio,
        created_at=portofolio.created_at,
        budget=portofolio.budget,
        allocations=[_to_item_response(i) for i in items],
        narasi_llm=portofolio.narasi_llm,
    )


def list_portfolio_history(db: Session, user_id: str) -> list[PortfolioHistoryItem]:
    """
    Ambil seluruh histori portofolio milik user (semua status, terbaru dulu).
    Dipakai endpoint /my-portofolio/history — histori "replaced" ikut tampil di sini.
    """
    user_uuid = _parse_user_uuid(user_id, db)
    rows = (
        db.query(Portofolio)
        .filter(Portofolio.user_id == user_uuid)
        .order_by(Portofolio.created_at.desc(), Portofolio.id.desc())
        .all()
    )
    return [
        PortfolioHistoryItem(
            id=str(p.id),
            budget=p.budget,
            total_terpakai=p.total_terpakai,
            sisa_budget=p.sisa_budget,
            fitness_score=p.fitness_score,
            sharpe_ratio=p.sharpe_ratio,
            max_drawdown=p.max_drawdown,
            risk_profile=p.risk_profile,
            status_portofolio=p.status_portofolio,
            created_at=p.created_at,
        )
        for p in rows
    ]

def get_portfolio_by_id(db: Session, portfolio_id: str) -> PortfolioResponse:
    """
    Ambil portofolio beserta itemnya dari database berdasarkan ID.
    Raises: ValueError jika ID tidak valid atau portofolio tidak ditemukan.
    """
    try:
        portfolio_uuid = UUID(portfolio_id)
    except (ValueError, TypeError) as e:
        raise ValueError(f"ID portofolio tidak valid: {portfolio_id}") from e

    portfolio = db.query(Portofolio).filter(Portofolio.id == portfolio_uuid).first()
    if not portfolio:
        raise ValueError(f"Portofolio dengan ID {portfolio_id} tidak ditemukan.")

    items = db.query(PortofolioItem).filter(PortofolioItem.portofolio_id == portfolio_uuid).all()

    allocations = [
        {
            "ticker": item.ticker,
            "lots": item.jumlah_lot,
            "price_per_lot": item.harga_acuan,
            "allocation": item.total_investasi,
            "weight": item.bobot_persentase / 100.0,
        }
        for item in items
    ]

    return PortfolioResponse(
        id=str(portfolio.id),
        user_id=str(portfolio.user_id),
        fitness_score=portfolio.fitness_score,
        sharpe_ratio=portfolio.sharpe_ratio,
        expected_return=None,  # Tidak disimpan di DB, bisa dihitung ulang jika perlu
        max_drawdown=portfolio.max_drawdown,
        avg_correlation=portfolio.avg_correlation,
        skor_fundamental=portfolio.skor_fundamental,
        total_terpakai=portfolio.total_terpakai,
        sisa_budget=portfolio.sisa_budget,
        n_active=len(items),
        allocated_budget_ok=(
            (portfolio.total_terpakai <= portfolio.budget)
            if portfolio.total_terpakai is not None
            else None
        ),
        risk_profile=portfolio.risk_profile,
        status_portofolio=portfolio.status_portofolio,
        created_at=portfolio.created_at,
        budget=portfolio.budget,
        allocations=allocations,
        narasi_llm=portfolio.narasi_llm,
    )