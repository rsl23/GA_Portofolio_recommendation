import logging
import os
from datetime import date
from uuid import UUID

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.backend.models.users import User
from src.backend.models.portofolios import Portofolio
from src.backend.models.portofolio_items import PortofolioItem
from src.backend.models.stock_universe import StockUniverse
from src.backend.models.market_data import MarketData
from src.backend.models.idx_composite import IdxComposite
from src.backend.models.schemas.portfolio_schema import (
    PortfolioGenerateRequest,
    PortfolioResponse,
    PortfolioHistoryItem,
    PortfolioItem,
)
from src.backend.services.gemini_llm_caller import explain_portfolio
from src.gaengine.data_loader import build_market_data
from src.gaengine.engine import GeneticEngine
from src.gaengine.ga_config import GAConfig

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Status portofolio
# ----------------------------------------------------------------------
# Setiap user bisa memiliki DUA jenis portofolio terpisah:
#   LIVE     : "active"             -> sedang dipakai; digantikan -> "replaced"
#   BACKTEST : "active_backtest"    -> simulasi terbaru; digantikan -> "replaced_backtest"
# Keduanya TIDAK saling menimpa. Portofolio juga TIDAK PERNAH dihapus —
# generate ulang hanya mengubah STATUS portofolio lama.
STATUS_ACTIVE = "active"                          # portofolio live terbaru
STATUS_REPLACED = "replaced"                      # portofolio live yang digantikan
STATUS_ACTIVE_BACKTEST = "active_backtest"        # hasil simulasi backtest terbaru
STATUS_REPLACED_BACKTEST = "replaced_backtest"    # hasil simulasi yang digantikan


class MarketDataUnavailableError(Exception):
    """Exception domain: MarketData gagal dibentuk / cache saham kosong."""


class UserNotFoundError(Exception):
    """Exception domain: user pada payload JWT tidak ditemukan di database."""


class PortfolioNotFoundError(Exception):
    """Exception domain: user belum memiliki portofolio aktif."""


class StockNotFoundError(Exception):
    """Exception domain: ticker dari market_data tidak terdaftar di stock_universe."""


def generate_new_portfolio(
    db: Session,
    request: PortfolioGenerateRequest,
    user_id: str,
    market_data=None,
    backtest: bool = False,
    date_ref: date | None = None,
) -> PortfolioResponse:
    """
    Controller Otak Utama:
    1. Menyiapkan MarketData:
         - LIVE     : pakai market_data dari app.state (bila ada) atau rakit
                      sendiri dari data live (DB cache + yfinance + ZAPI).
         - BACKTEST : WAJIB rakit dari data historis lokal pada `date_ref`
                      (lihat src/gaengine/data_loader.build_market_data).
    2. Menjalankan Algoritma Genetika sesuai modal & profil risiko pengguna.
    3. Menyimpan hasil ke tabel portofolios + portofolio_items.
       Status portofolio mengikuti mode:
         - LIVE     : baru = "active";            lama = "replaced"
         - BACKTEST : baru = "active_backtest";   lama = "replaced_backtest"
       Jadi portofolio LIVE dan BACKTEST berdampingan (tidak saling menimpa),
       dan portofolio lama TIDAK PERNAH dihapus — hanya berubah status.
    """
    logger.info(
        "Menjalankan GA untuk user %s dengan profil %s dan modal Rp%.2f (backtest=%s, date=%s)",
        user_id, request.risk_profile, request.budget, backtest, date_ref,
    )

    # 0. Validasi user dari payload JWT (sub). Lempar error jika sudah terhapus.
    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise UserNotFoundError(f"User ID pada token tidak valid: {user_id}") from e
    user = db.query(User).filter(User.id == user_uuid).first()
    if user is None:
        raise UserNotFoundError("User pada token tidak ditemukan di database.")

    # 1. Siapkan MarketData.
    #    - Mode BACKTEST: SELALU rakit ulang dari data historis lokal pada
    #      `date_ref`; market_data live dari app.state tidak boleh dipakai.
    #    - Mode LIVE: pakai market_data yang diberikan (app.state) bila ada,
    #      kalau None baru rakit sendiri dari data live.
    if backtest:
        if date_ref is None:
            raise MarketDataUnavailableError(
                "Mode backtest memerlukan parameter 'date_ref' (format YYYY-MM-DD)."
            )
        logger.info("Membentuk MarketData BACKTEST per %s...", date_ref)
        try:
            market_data = build_market_data(
                min_price=1.0, backtest=True, date_ref=date_ref
            )
        except Exception as e:
            raise MarketDataUnavailableError(
                f"Gagal merakit MarketData backtest untuk {date_ref}: {e}"
            ) from e
    elif market_data is None:
        logger.info("Membentuk MarketData menggunakan data LIVE...")
        market_data = build_market_data(min_price=1.0)

    if market_data is None or market_data.n_stocks == 0:
        raise MarketDataUnavailableError(
            "Gagal membentuk MarketData. Pastikan tabel filtered_stock_cache sudah terisi "
            "(live) atau tanggal backtest memiliki data (backtest)."
        )

    # 2. Konfigurasi GA mengikuti profil & modal dari payload request
    config = GAConfig(
        population_size=200,        # jumlah kromosom
        generations=500,            # jumlah generasi evolusi
        budget=request.budget,      # modal pengguna (IDR)
        risk_profile=request.risk_profile,
        min_stocks=3,
        max_stocks=10,
        risk_free_rate=market_data.risk_free_rate,
        # seed=42,
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
            "item_id": "",  # diisi setelah item tersimpan (lihat langkah 6)
            "ticker": code,
            "lots": lot,
            "price_per_lot": price,
            "harga_beli": price,  # default: user bisa edit lewat PATCH
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

    # 4a. Narasi TEMPLATE (selalu dibuat): dipakai apa adanya bila narasi LLM
    #     tidak tersedia / gagal, sehingga endpoint generate tidak pernah gagal
    #     hanya karena layanan LLM bermasalah.
    narasi = (
        f"Portofolio dengan profil risiko {request.risk_profile} memilih {solution.n_active} "
        f"saham dari {market_data.n_stocks} kandidat. Total dana terpakai "
        f"Rp{total:,.0f} dari budget Rp{request.budget:,.0f}. Skor fitness yang dicapai "
        f"{solution.fitness:.4f} dengan Sharpe ratio {solution.sharpe_ratio:.3f}. "
        f"Penurunan maksimum (max drawdown) tercatat {solution.max_drawdown:.2%} "
        f"dan korelasi rata-rata antar saham {solution.avg_correlation:.3f}. "
        f"Rekomendasi ini dihasilkan Algoritma Genetika."
    )

    # 4b. Narasi LLM (Gemini): menjelaskan MENGAPA & BAGAIMANA portofolio ini
    #     terpilih oleh GA -- metrik fitness function (Sharpe, MDD, korelasi,
    #     bonus fundamental), batasan budget/diversifikasi, serta peran tiap
    #     emiten pemenang. Bahan/prompt dirakit src/gaengine/prompt_builder.py,
    #     pemanggilan model ada di src/backend/services/gemini_llm_caller.py.
    #     Bila gagal (tanpa API key, timeout, dsb) -> None, narasi template dipakai.
    narasi_llm = explain_portfolio(
        solution=solution,
        market_data=market_data,
        config=config,
        allocations=allocations,
        mode="backtest" if backtest else "live",
        date_ref=date_ref,
        budget=request.budget,
        expected_return=expected_return,
        generations_run=len(engine.history),
        fallback=None,
    )
    if narasi_llm:
        narasi = narasi_llm
    logger.info(
        "Narasi portofolio memakai %s.",
        "penjelasan LLM (Gemini)" if narasi_llm else "template GA (fallback)",
    )

    if backtest:
        narasi += (
            f" Hasil ini berasal dari SIMULASI BACKTEST per {date_ref} "
            f"(data historis, bukan kondisi pasar hari ini) dan disimpan sebagai "
            f"portofolio simulasi (status {STATUS_ACTIVE_BACKTEST}) sehingga tidak "
            f"mengubah portofolio live Anda."
        )

    # 5a. date_ref acuan portofolio (kolom Portofolio.date_ref):
    #     - BACKTEST : tanggal simulasi dari parameter.
    #     - LIVE     : sekarang (selaras created_at).
    from datetime import datetime as _dt
    if backtest:
        date_ref_dt = _dt.combine(date_ref, _dt.min.time())
    else:
        date_ref_dt = _dt.now()

    # 5. Supersede: portofolio lama DIGANTIKAN STATUSNYA (tidak dihapus).
    #    Mode LIVE     : active            -> replaced
    #    Mode BACKTEST : active_backtest   -> replaced_backtest
    #    Riwayat tetap tersimpan dan bisa diambil via /my-portofolio/history.
    if backtest:
        status_baru = STATUS_ACTIVE_BACKTEST
        status_lama = STATUS_ACTIVE_BACKTEST
        status_digantikan = STATUS_REPLACED_BACKTEST
    else:
        status_baru = STATUS_ACTIVE
        status_lama = STATUS_ACTIVE
        status_digantikan = STATUS_REPLACED

    replaced = (
        db.query(Portofolio)
        .filter(
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio == status_lama,
        )
        .update({"status_portofolio": status_digantikan}, synchronize_session=False)
    )
    if replaced:
        logger.info(
            "Supersede (%s): %d portofolio '%s' -> '%s'.",
            "backtest" if backtest else "live", replaced, status_lama, status_digantikan,
        )

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
        funda_alpha=float(config.fundamental_bonus),   # alpha (bonus fundamental) per profil
        narasi_llm=narasi,
        # "active" untuk mode live, "active_backtest" untuk mode simulasi
        status_portofolio=status_baru,
        # field rebalance: ini portofolio baru, bukan hasil rebalance
        is_rebalance=False,
        parent_portofolio_id=None,
        turnover_rate=None,          # turnover hanya relevan saat rebalance
        turnover_penalty_beta=None,  # beta hanya dipakai saat rebalance
        # acuan tanggal portofolio: tanggal simulasi (backtest) / sekarang (live)
        date_ref=date_ref_dt,
    )
    db.add(portofolio)
    db.flush()  # dapatkan portofolio.id sebelum commit (id di-generate default-nya)

    # Map ticker -> id_stock dari stock_universe (wajib terdaftar karena
    # sync harian API ZPI; ticker asing = bug preprocessing, gagal keras).
    needed_tickers = list({t for t, lot, _ in rows if lot > 0})
    stock_rows = (
        db.query(StockUniverse)
        .filter(StockUniverse.ticker.in_(needed_tickers))
        .all()
    )
    ticker_to_id = {s.ticker: s.id_stock for s in stock_rows}
    missing = [t for t in needed_tickers if t not in ticker_to_id]
    if missing:
        db.rollback()
        raise StockNotFoundError(
            f"Ticker berikut tidak terdaftar di stock_universe: {', '.join(missing)}"
        )

    items = [
        PortofolioItem(
            portofolio_id=portofolio.id,
            stock_id=ticker_to_id[ticker],
            bobot_persentase=float((lot * price / total) * 100.0) if total else 0.0,
            jumlah_lot=lot,
            # price dari market_data adalah harga PER LOT (100 lembar);
            # kolom harga_acuan & harga_beli disimpan PER LEMBAR
            harga_acuan=price / 100.0,
            harga_beli=price / 100.0,  # default: user bisa edit lewat PATCH
            total_investasi=float(lot * price),
            action_type="buy",  # portofolio baru: semua posisi adalah pembelian awal
            # window kepemilikan: mulai pada date_ref portofolio;
            # end_date NULL = masih dipegang (diisi saat rebalance/sell).
            start_date=date_ref_dt,
            end_date=None,
        )
        for ticker, lot, price in rows
        if lot > 0
    ]
    db.add_all(items)
    db.commit()
    db.refresh(portofolio)
    logger.info("Portofolio %s berhasil disimpan dengan %d item.", portofolio.id, len(items))

    # Isi item_id pada allocations (item sudah tersimpan -> id tersedia)
    saved_rows = [r for r in rows if r[1] > 0]
    item_id_map = {r[0]: str(item.id) for r, item in zip(saved_rows, items)}
    for a in allocations:
        a["item_id"] = item_id_map[a["ticker"]]

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
        date_ref=portofolio.date_ref,
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


def _to_item_response(item: PortofolioItem, ticker: str) -> PortfolioItem:
    """Konversi PortofolioItem (ORM) + ticker (dari join stock_universe) ke schema response.
    harga_acuan/harga_beli di DB per LEMBAR; price_per_lot response = x100."""
    total = item.total_investasi or 0.0
    return PortfolioItem(
        item_id=str(item.id),
        ticker=ticker,
        lots=item.jumlah_lot,
        price_per_lot=(item.harga_acuan or 0.0) * 100.0,
        harga_beli=item.harga_beli,
        allocation=total,
        weight=(item.bobot_persentase or 0.0) / 100.0,
    )


def _query_items_with_ticker(db: Session, portofolio_id) -> list[tuple[PortofolioItem, str]]:
    """Query item portofolio di-join ke stock_universe agar ticker tersedia untuk response."""
    return (
        db.query(PortofolioItem, StockUniverse.ticker)
        .join(StockUniverse, PortofolioItem.stock_id == StockUniverse.id_stock)
        .filter(PortofolioItem.portofolio_id == portofolio_id)
        .order_by(PortofolioItem.total_investasi.desc())
        .all()
    )


def _to_portfolio_response(db: Session, portofolio: Portofolio) -> PortfolioResponse:
    """Bangun PortfolioResponse dari satu baris Portofolio + item-itemnya.

    Dipakai BERSAMA oleh get_active_portfolio & get_all_my_portfolios agar bentuk
    respons selalu identik. Field yang tidak disimpan di DB (expected_return)
    bernilai None, n_active dihitung dari jumlah item, sedangkan total_terpakai &
    sisa_budget dihitung ULANG dari total_investasi item (berbasis harga_beli
    milik user) karena kolom di tabel portofolios bisa basi setelah user
    mengedit harga_beli lewat PATCH.
    """
    item_pairs = _query_items_with_ticker(db, portofolio.id)
    items = [pair[0] for pair in item_pairs]

    total_terpakai = sum((it.total_investasi or 0.0) for it in items)

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
        sisa_budget=portofolio.budget - total_terpakai,
        n_active=len(items),
        allocated_budget_ok=total_terpakai <= portofolio.budget,
        risk_profile=portofolio.risk_profile,
        status_portofolio=portofolio.status_portofolio,
        created_at=portofolio.created_at,
        date_ref=portofolio.date_ref,
        budget=portofolio.budget,
        allocations=[_to_item_response(item, ticker) for item, ticker in item_pairs],
        narasi_llm=portofolio.narasi_llm,
    )


def get_active_portfolio(db: Session, user_id: str, backtest: bool = False) -> PortfolioResponse:
    """
    Ambil portofolio AKTIF terbaru milik user (dari payload JWT).
    `backtest=False` (default) -> portofolio LIVE berstatus "active".
    `backtest=True`            -> portofolio SIMULASI berstatus "active_backtest".
    Ordering: created_at terbaru, dengan id sebagai tiebreaker.
    Respons memakai schema yang SAMA dengan hasil generate (PortfolioResponse):
    field yang tidak disimpan di DB (expected_return) bernilai None,
    n_active dihitung dari jumlah item, allocated_budget_ok dihitung ulang,
    date_ref diambil dari kolom Portofolio.date_ref (backtest = tanggal
    simulasi, live = waktu portofolio digenerate).
    Raises: UserNotFoundError jika user tidak valid / tidak ada.
            PortfolioNotFoundError jika user belum pernah generate.
    """
    user_uuid = _parse_user_uuid(user_id, db)
    status_target = STATUS_ACTIVE_BACKTEST if backtest else STATUS_ACTIVE
    portofolio = (
        db.query(Portofolio)
        .filter(
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio == status_target,
        )
        .order_by(Portofolio.created_at.desc(), Portofolio.id.desc())
        .first()
    )
    if portofolio is None:
        raise PortfolioNotFoundError(
            "User belum memiliki portofolio backtest aktif."
            if backtest
            else "User belum memiliki portofolio aktif."
        )

    # Respons dibangun lewat helper bersama _to_portfolio_response (bentuk
    # respons identik dengan get_all_my_portfolios): total_terpakai & sisa_budget
    # dihitung ulang dari item (total_investasi berbasis harga_beli milik user)
    # — nilai kolom portofolio bisa basi setelah user mengedit harga_beli lewat PATCH.
    return _to_portfolio_response(db, portofolio)


def get_all_my_portfolios(
    db: Session,
    user_id: str,
    backtest: bool = False,
) -> list[PortfolioResponse]:
    """
    Ambil SEMUA portofolio milik user (identitas dari payload JWT) yang dipisah
    berdasarkan MODE lewat kolom `status_portofolio`:

      backtest=False -> mode LIVE     : status "active"  + "replaced"
      backtest=True  -> mode BACKTEST : status "active_backtest" + "replaced_backtest"

    Karena pemisahan mode memakai status_portofolio, TIDAK ada portofolio yang
    tertukar antar mode: portofolio live yang sudah digantikan ("replaced") tetap
    tampil di mode live, dan hasil simulasi lama ("replaced_backtest") tetap
    tampil di mode backtest. Portofolio berstatus "replaced_*" TIDAK dihapus dari DB.
    Urutan: created_at terbaru dulu (id sebagai tiebreaker).
    Respons memakai schema yang SAMA dengan GET /my-portofolio
    (PortfolioResponse, lengkap dengan daftar alokasi itemnya), termasuk
    `date_ref` (kolom Portofolio.date_ref) sehingga frontend bisa membedakan
    tanggal simulasi portofolio backtest dan tanggal dibuat portofolio live.
    User yang belum punya portofolio pada mode tsb -> list kosong (bukan error).
    Raises: UserNotFoundError jika user pada token tidak valid / tidak ada.
    """
    user_uuid = _parse_user_uuid(user_id, db)

    # Filter status sesuai mode: LIVE (active/replaced) vs BACKTEST (active/replaced _backtest)
    status_targets = (
        (STATUS_ACTIVE_BACKTEST, STATUS_REPLACED_BACKTEST)
        if backtest
        else (STATUS_ACTIVE, STATUS_REPLACED)
    )

    rows = (
        db.query(Portofolio)
        .filter(
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio.in_(status_targets),
        )
        .order_by(Portofolio.created_at.desc(), Portofolio.id.desc())
        .all()
    )
    logger.info(
        "Mengambil %d portofolio %s untuk user %s (status: %s).",
        len(rows), "backtest" if backtest else "live", user_uuid,
        ", ".join(status_targets),
    )

    return [_to_portfolio_response(db, portofolio) for portofolio in rows]


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

    item_pairs = _query_items_with_ticker(db, portfolio_uuid)
    items = [pair[0] for pair in item_pairs]

    allocations = [
        {
            "item_id": str(item.id),
            "ticker": ticker,
            "lots": item.jumlah_lot,
            "price_per_lot": (item.harga_acuan or 0.0) * 100.0,  # DB per lembar -> response per lot
            "harga_beli": item.harga_beli,
            "allocation": item.total_investasi,
            "weight": item.bobot_persentase / 100.0,
        }
        for item, ticker in item_pairs
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
        date_ref=portfolio.date_ref,  # acuan tanggal: tanggal simulasi (backtest) / waktu generate (live)
        budget=portfolio.budget,
        allocations=allocations,
        narasi_llm=portfolio.narasi_llm,
    )


class ItemNotFoundError(Exception):
    """Exception domain: item portofolio tidak ditemukan / bukan milik user."""


def update_harga_beli(db: Session, user_id: str, item_id: str, harga_beli: float) -> PortofolioItem:
    """
    Ubah harga_beli (per lembar) pada satu item portofolio milik user.
    Validasi kepemilikan: item harus berada di portofolio milik user (dari JWT)
    dan berstatus active - portofolio lama (replaced) tidak boleh diedit.
    total_investasi ikut dihitung ulang: jumlah_lot * 100 * harga_beli.
    Raises: ItemNotFoundError.
    """
    from uuid import UUID

    try:
        user_uuid = UUID(user_id)
        item_uuid = UUID(item_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise ItemNotFoundError(f"ID tidak valid: {item_id}") from e

    # Cek kepemilikan: item -> portofolio -> user (active saja)
    item = (
        db.query(PortofolioItem)
        .join(Portofolio, PortofolioItem.portofolio_id == Portofolio.id)
        .filter(
            PortofolioItem.id == item_uuid,
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio == "active",
        )
        .first()
    )
    if item is None:
        raise ItemNotFoundError(
            "Item portofolio tidak ditemukan, bukan milik Anda, atau portofolionya sudah tidak aktif."
        )

    item.harga_beli = float(harga_beli)
    # total_investasi dihitung ulang berdasar harga beli user (per lembar x 100)
    item.total_investasi = float(item.jumlah_lot) * 100.0 * float(harga_beli)
    db.flush()

    # Rekapulasi level portofolio: total_terpakai & sisa_budget mengacu ke
    # total_investasi item yang sudah berbasis harga_beli (bukan harga_acuan).
    new_total = (
        db.query(func.sum(PortofolioItem.total_investasi))
        .filter(PortofolioItem.portofolio_id == item.portofolio_id)
        .scalar()
    ) or 0.0
    portofolio_row = db.query(Portofolio).filter(Portofolio.id == item.portofolio_id).first()
    if portofolio_row is not None:
        portofolio_row.total_terpakai = float(new_total)
        portofolio_row.sisa_budget = float(portofolio_row.budget) - float(new_total)

    db.commit()
    db.refresh(item)
    logger.info(
        "harga_beli item %s diubah menjadi Rp%.2f (total investasi Rp%.2f, total terpakai Rp%.2f)",
        item_id, harga_beli, item.total_investasi, new_total,
    )
    return item


def _query_price_rows(db: Session, stock_ids: list, start_date, end_date):
    """
    Query baris harga (date, adj_close, close) untuk saham portofolio pada
    rentang [start_date, end_date]; `end_date=None` berarti tanpa batas atas.
    Dipisah dari pemanggil agar bisa diulang setelah auto-sync berjalan.
    """
    price_query = (
        db.query(MarketData.stock_id, MarketData.date, MarketData.adj_close, MarketData.close)
        .filter(
            MarketData.stock_id.in_(stock_ids),
            MarketData.date >= start_date,
        )
    )
    if end_date is not None:
        price_query = price_query.filter(MarketData.date <= end_date)
    return price_query.all()


def _latest_available_price_date(db: Session, stock_ids: list, end_date=None):
    """
    Tanggal harga TERAKHIR yang sudah tersimpan di `market_data` untuk saham
    tsb (dibatasi `end_date` bila diberikan); None bila belum ada sama sekali.

    Dipakai untuk MENYESUAIKAN awal rentang performa ketika `date_ref`
    portofolio jatuh pada hari NON-BURSA (weekend / libur BEI) sehingga
    tanggalnya lebih baru daripada tanggal perdagangan terakhir yang tersedia.
    `sync_market_data` sengaja berhenti pada `latest_trading_date` (lihat
    src/backend/services/price_history_service.py), jadi tanpa penyesuaian ini
    rentang [start_date, ...] tidak pernah berisi data -> endpoint 404.
    """
    query = (
        db.query(func.max(MarketData.date))
        .filter(MarketData.stock_id.in_(stock_ids))
    )
    if end_date is not None:
        query = query.filter(MarketData.date <= end_date)

    latest = query.scalar()
    if latest is None:
        return None
    return latest.date() if hasattr(latest, "date") else latest


def _sync_prices_on_demand(db: Session, reason: str) -> dict | None:
    """
    Jalankan sync_market_data (yfinance -> tabel market_data + idx_composite)
    SATU KALI untuk mengisi data harga yang belum ada.

    Sengaja TIDAK melempar exception: kegagalan sync (mis. jaringan / yfinance
    menolak) hanya dicatat di log, lalu pemanggil memeriksa ulang datanya dan
    melempar PortfolioNotFoundError yang informatif bila tetap kosong.

    Returns:
        dict statistik sync ({stocks, rows_upserted, idx_rows_upserted, errors}),
        atau None bila auto-sync dimatikan / gagal.
    """

    from src.backend.services.price_history_service import sync_market_data

    try:
        logger.info("Auto-sync harga dimulai (%s): menarik histori harian via yfinance...", reason)
        stats = sync_market_data(db)
        logger.info(
            "Auto-sync harga selesai (%s): %s saham, %s baris harga, %s baris IHSG, %s error.",
            reason, stats.get("stocks"), stats.get("rows_upserted"),
            stats.get("idx_rows_upserted"), len(stats.get("errors") or []),
        )
        return stats
    except Exception as e:  # noqa: BLE001 - kegagalan sync tidak boleh mematikan endpoint
        logger.exception("Auto-sync harga (%s) gagal: %s", reason, e)
        return None


def _describe_sync(stats: dict | None) -> str:
    """Ringkasan hasil auto-sync untuk pesan error / info pada respons."""
    if stats is None:
        return ("Sinkronisasi harga otomatis tidak dijalankan "
                "(AUTO_SYNC_PRICES=false atau proses sync gagal).")
    return (
        "Sinkronisasi harga otomatis sudah dijalankan "
        f"({stats.get('stocks', 0)} saham diproses, {stats.get('rows_upserted', 0)} baris "
        f"harga & {stats.get('idx_rows_upserted', 0)} baris IHSG tersimpan, "
        f"{len(stats.get('errors') or [])} error)."
    )



def get_portfolio_performance(
    db: Session,
    user_id: str,
    portfolio_id: str,
    end_date: date | None = None,
    backtest: bool = False,
) -> dict:
    """
    Hitung performa (persentase return kumulatif harian) portofolio TERTENTU
    milik user (identifikasi via portfolio_id dari path) vs IHSG.

    Rentang perhitungan:
      mulai  : portofolio.date_ref (kolom acuan; fallback created_at)
      akhir  : parameter `end_date` (YYYY-MM-DD) bila diberikan,
               selain itu sampai data harga terakhir yang tersedia.

    Penyesuaian hari bursa (agar endpoint tetap jalan pada hari non-bursa):
      - Bila `date_ref`/`item.start_date` jatuh pada weekend / libur BEI
        (mis. portofolio digenerate hari Minggu) sehingga lebih baru daripada
        tanggal perdagangan terakhir yang datanya tersedia -- sync memang
        berhenti di `latest_trading_date` pada price_history_service.py --
        awal rentang digeser MUNDUR ke tanggal perdagangan terakhir tersebut
        (lihat `_latest_available_price_date`), dan item dianggap aktif sejak
        tanggal itu. Tanggal yang diminta tetap dilaporkan pada respons
        sebagai `requested_start_date` (+ flag `start_date_adjusted`).
      - Item dengan `start_date` di luar cakupan data juga disesuaikan:
        sebelum data pertama -> aktif sejak data pertama; setelah data
        terakhir -> aktif sejak tanggal perdagangan terakhir, kecuali
        `end_date` yang diminta memang lebih awal dari tanggal pembelian item.

    Metode (berbasis KEPEMILIKAN LOT per item, bukan bobot investasi):
      nilai_portofolio(t) = SUM over item(
          jumlah_lot_i(t) * adj_close_i(t) )   bila t di dalam window
                                               [item.start_date, item.end_date]
      return(t)           = nilai_portofolio(t) / nilai_portofolio(t0) - 1
    Karena tiap item punya start_date/end_date sendiri, kepemilikan bisa
    berubah sepanjang waktu (efek rebalance: buy/sell/add/reduce) dan semua
    item milik ticker yang sama diakumulasi lewat interval masing-masing.

    Args:
        portfolio_id : ID portofolio (path param). Portofolio HARUS milik
                       user pada JWT — portofolio user lain -> 404.
        end_date     : batas akhir perhitungan (inklusif); None = data terakhir.

    Raises: PortfolioNotFoundError jika ID tidak valid / bukan milik user /
            data harga belum tersedia pada rentang tsb.
    """
    import pandas as pd

    try:
        user_uuid = UUID(user_id)
        portfolio_uuid = UUID(portfolio_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise PortfolioNotFoundError(f"ID tidak valid: {e}") from e

    # 1. Portofolio by ID — WAJIB milik user dari JWT (privasi antar user:
    #    portofolio milik orang lain diperlakukan seperti tidak ada).
    portfolio = (
        db.query(Portofolio)
        .filter(Portofolio.id == portfolio_uuid, Portofolio.user_id == user_uuid)
        .first()
    )
    if portfolio is None:
        raise PortfolioNotFoundError(
            f"Portofolio {portfolio_id} tidak ditemukan (atau bukan milik Anda)."
        )

    # Deteksi mode: dari PARAMETER atau otomatis dari status portofolio,
    # sehingga portofolio backtest selalu dihitung dari data parquet historis
    # walau frontend lupa mengirim ?backtest=true.
    is_backtest = (
        backtest
        or portfolio.status_portofolio in (STATUS_ACTIVE_BACKTEST, STATUS_REPLACED_BACKTEST)
    )

    # Awal hitung = date_ref portofolio (backtest: tanggal simulasi;
    # live: sekarang). Fallback created_at untuk baris lama tanpa date_ref.
    start_dt = portfolio.date_ref or portfolio.created_at
    start_date = start_dt.date() if hasattr(start_dt, "date") else start_dt

    # Tanggal yang DIMINTA (date_ref portofolio) disimpan terpisah karena
    # `start_date` bisa digeser ke tanggal perdagangan terakhir yang tersedia
    # bila date_ref jatuh pada hari non-bursa (lihat langkah 3a).
    requested_start = start_date

    # 2. Item kepemilikan (window start_date/end_date ikut diambil)
    holdings_rows = (
        db.query(
            PortofolioItem.stock_id,
            StockUniverse.ticker,
            PortofolioItem.jumlah_lot,
            PortofolioItem.start_date,
            PortofolioItem.end_date,
        )
        .join(Portofolio, PortofolioItem.portofolio_id == Portofolio.id)
        .join(StockUniverse, PortofolioItem.stock_id == StockUniverse.id_stock)
        .filter(PortofolioItem.portofolio_id == portfolio.id)
        .all()
    )
    if not holdings_rows:
        raise PortfolioNotFoundError("User belum memiliki saham di portofolio aktifnya.")

    stock_ids = [r.stock_id for r in holdings_rows]
    id_to_ticker = {str(r.stock_id): r.ticker for r in holdings_rows}

    # Info auto-sync harga (diisi hanya bila endpoint menarik data via
    # sync_market_data karena data harga/IHSG belum tersedia di DB).
    sync_stats: dict | None = None

    if not is_backtest:
        # 3. Harga per (date, stock_id) sejak start_date (<= end_date bila ada) (LIVE)
        price_rows = _query_price_rows(db, stock_ids, start_date, end_date)
        if not price_rows:
            # Data harga belum ada di DB. Daripada langsung melempar error,
            # tarik dulu histori harian via sync_market_data (yfinance) lalu
            # ulangi query yang sama — sync bersifat upsert, aman diulang.
            logger.info(
                "Performa portofolio %s: data harga belum tersedia -> menjalankan "
                "auto-sync harga (sync_market_data).", portfolio_id,
            )
            sync_stats = _sync_prices_on_demand(
                db, reason=f"data harga portofolio {portfolio_id} belum ada"
            )
            price_rows = _query_price_rows(db, stock_ids, start_date, end_date)
        if not price_rows:
            # 3a. Masih kosong: kemungkinan besar date_ref portofolio jatuh pada
            #     hari NON-BURSA (weekend/libur BEI) sehingga lebih baru daripada
            #     latest_trading_date. sync_market_data sengaja berhenti pada
            #     tanggal perdagangan terakhir (price_history_service.py),
            #     sehingga awal rentang digeser mundur ke tanggal perdagangan
            #     terakhir yang datanya benar-benar tersedia di market_data.
            fallback_start = _latest_available_price_date(db, stock_ids, end_date)
            if fallback_start is not None and fallback_start < start_date:
                logger.info(
                    "Performa portofolio %s: tidak ada harga sejak %s (hari non-bursa?) -> "
                    "awal rentang digeser ke hari bursa terakhir yang tersedia: %s.",
                    portfolio_id, start_date, fallback_start,
                )
                start_date = fallback_start
                price_rows = _query_price_rows(db, stock_ids, start_date, end_date)

        if not price_rows:
            raise PortfolioNotFoundError(
                "Belum ada data harga tersimpan untuk saham portofolio aktif. "
                + _describe_sync(sync_stats)
                + " Bila portofolio dibuat pada hari non-bursa (weekend/libur BEI) "
                "dan sahamnya baru pertama kali dibeli, tunggu hari bursa "
                "berikutnya lalu sync harga kembali. Periksa koneksi/ticker "
                "saham lalu jalankan sync-prices secara manual."
            )

        # 4. Pivot date x stock (adj_close; fallback close bila adj kosong)
        recs = [
            {"date": r.date, "stock_id": str(r.stock_id),
             "price": r.adj_close if r.adj_close is not None else r.close}
            for r in price_rows
            if (r.adj_close is not None or r.close is not None)
        ]
        if not recs:
            raise PortfolioNotFoundError("Data harga (adj_close/close) belum tersedia untuk saham portofolio aktif.")

        df_price = pd.DataFrame(recs).pivot_table(
            index="date", columns="stock_id", values="price", aggfunc="last"
        ).sort_index()
    else:
        # 3 & 4. Harga historis dari Parquet (BACKTEST)
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent.parent.parent
        PRICE_FILE = ROOT / "data" / "Master_OHLCV_15Tahun.parquet"

        # Karena di Parquet masih ada ".JK", kita tambahkan untuk keperluan filter
        tickers = list(id_to_ticker.values())
        tickers_jk = [t + ".JK" for t in tickers]
        
        # Susun filter Predicate Pushdown untuk PyArrow
        parquet_filters = [
            ("Date", ">=", pd.Timestamp(start_date)),
            ("Ticker", "in", tickers_jk)
        ]
        if end_date is not None:
            parquet_filters.append(("Date", "<=", pd.Timestamp(end_date)))

        try:
            # Pandas hanya akan menarik baris yang lolos filter dari hard disk (RAM sangat aman)
            window = pd.read_parquet(PRICE_FILE, filters=parquet_filters)
        except Exception as e:
            raise PortfolioNotFoundError(f"Gagal membaca data Parquet: {e}")

        # Rapihkan formatnya sesuai dengan kebutuhan kode di bawahnya
        window["Date"] = pd.to_datetime(window["Date"])
        window["Ticker"] = window["Ticker"].astype(str).str.replace(".JK", "", regex=False)

        if window.empty:
            # Pesan informatif: tunjukkan rentang yang diminta vs cakupan data
            # agar jelas apakah date_ref di luar cakupan parquet. Kolom Date
            # dibaca ulang (satu kolom saja, murah) karena `window` sudah kosong
            # sehingga min/max-nya tidak bisa dipakai (bernilai NaT).
            cakupan = "tidak diketahui"
            try:
                all_dates = pd.to_datetime(
                    pd.read_parquet(PRICE_FILE, columns=["Date"])["Date"]
                )
                if not all_dates.empty:
                    cakupan = f"{all_dates.min().date()} s/d {all_dates.max().date()}"
            except Exception as exc:  # noqa: BLE001 - pesan error tetap dibuat
                logger.warning("Gagal membaca cakupan tanggal parquet: %s", exc)

            raise PortfolioNotFoundError(
                f"Belum ada data harga tersimpan di rentang simulasi ini "
                f"(diminta {start_date} s/d {end_date or cakupan}; "
                f"data parquet hanya mencakup {cakupan}). "
                f"Pilih date_ref/backtest dalam cakupan data."
            )
            
        ticker_to_id = {t: str(i) for i, t in id_to_ticker.items()}
        window["stock_id"] = window["Ticker"].map(ticker_to_id)
        
        if "Adj Close" in window.columns:
            window["price"] = window["Adj Close"].fillna(window["Close"])
        else:
            window["price"] = window["Close"]
            
        df_price = window.pivot_table(
            index="Date", columns="stock_id", values="price", aggfunc="last"
        ).sort_index()

    # Normalisasi: MarketData.date bisa berupa datetime.date -> Timestamp,
    # agar perbandingan window item (pd.Timestamp) tidak TypeError.
    df_price.index = pd.to_datetime(df_price.index)
    df_price = df_price.ffill()            # isi hari bolong per saham (suspensi)
    df_price = df_price.dropna(how="all")  # buang baris libur bersama
    if df_price.empty:
        raise PortfolioNotFoundError("Data harga belum cukup untuk menghitung performa.")

    df_price.columns = [id_to_ticker[c] for c in df_price.columns]

    # 5. Nilai portofolio harian = SUM(lot aktif pada t x harga(t) x 100).
    #    Lot aktif ditentukan window [item.start_date, item.end_date] MILIK ITEM,
    #    sehingga perubahan kepemilikan akibat rebalance ikut terhitung.
    #    Penyesuaian hari bursa:
    #      - item_start sebelum data pertama -> aktif sejak data pertama;
    #      - item_start SETELAH data terakhir (mis. portofolio dibuat pada
    #        weekend/libur bursa) -> dihitung aktif sejak tanggal perdagangan
    #        terakhir, KECUALI `end_date` yang diminta memang lebih awal dari
    #        tanggal pembelian item tersebut (item belum boleh ikut dihitung).
    LOTS_PER_SHARE = 100
    series_start = df_price.index.min()
    series_last = df_price.index.max()

    # Rentang EFEKTIF tiap item dipakai untuk nilai harian MAUPUN snapshot
    # holdings, agar keduanya selalu konsisten.
    effective_items = []
    missing_price_tickers: set[str] = set()
    for r in holdings_rows:
        ticker = r.ticker
        if ticker not in df_price.columns:
            # Saham tanpa SATU PUN baris harga pada rentang ini (mis. baru
            # pertama kali dibeli pada hari non-bursa sehingga sync_market_data
            # belum pernah menarik harganya) dicatat agar nilai portofolio yang
            # kurang lengkap tidak lolos tanpa jejak.
            missing_price_tickers.add(ticker)
            continue

        item_start = (
            pd.Timestamp(r.start_date) if r.start_date is not None
            else pd.Timestamp(start_date)
        )
        if item_start < series_start:
            item_start = series_start
        elif item_start > series_last:
            if end_date is not None and item_start > pd.Timestamp(end_date):
                continue  # dibeli setelah batas akhir yang diminta -> belum aktif
            item_start = series_last

        item_end = pd.Timestamp(r.end_date) if r.end_date is not None else None
        effective_items.append((ticker, int(r.jumlah_lot), item_start, item_end))

    portfolio_value = pd.Series(0.0, index=df_price.index)
    for ticker, lot, item_start, item_end in effective_items:
        mask = df_price.index >= item_start
        if item_end is not None:
            mask &= df_price.index <= item_end
        # Harga yang belum tersedia (saham belum tercatat / suspensi) dihitung 0
        # agar tidak menghasilkan NaN yang mematikan seluruh seri nilai.
        contribution = (lot * LOTS_PER_SHARE * df_price[ticker]).where(mask, 0.0)
        portfolio_value += contribution.fillna(0.0)

    # t0 = tanggal pertama dengan nilai > 0 (fallback: tanggal pertama).
    nonzero = portfolio_value[portfolio_value > 0]
    if nonzero.empty:
        raise PortfolioNotFoundError("Tidak ada kepemilikan aktif pada rentang perhitungan.")
    portfolio_return = (portfolio_value / nonzero.iloc[0] - 1).fillna(0.0)

    # 6. Return IHSG pada rentang yang sama
    ihsg_return = None
    
    if not is_backtest:
        idx_query = (
            db.query(IdxComposite)
            .filter(IdxComposite.date >= df_price.index.min())
        )
        if end_date is not None:
            idx_query = idx_query.filter(IdxComposite.date <= pd.Timestamp(end_date))
        idx_rows = idx_query.order_by(IdxComposite.date.asc()).all()
        if not idx_rows and sync_stats is None:
            # Benchmark IHSG belum tersimpan -> satu panggilan sync mengisi
            # market_data + idx_composite sekaligus, lalu query diulang.
            logger.info(
                "Performa portofolio %s: data IHSG belum tersedia -> menjalankan "
                "auto-sync harga (benchmark).", portfolio_id,
            )
            sync_stats = _sync_prices_on_demand(db, reason="data IHSG (benchmark) belum ada")
            idx_rows = idx_query.order_by(IdxComposite.date.asc()).all()
        if not idx_rows:
            logger.warning(
                "Performa portofolio %s: data IHSG tetap kosong -> garis benchmark tidak "
                "ditampilkan. %s", portfolio_id, _describe_sync(sync_stats),
            )
        if idx_rows:
            idx_series = pd.Series(
                {r.date: r.close for r in idx_rows if r.close is not None}
            ).sort_index()
            ihsg_return = idx_series / idx_series.iloc[0] - 1
    else:
        # BACKTEST: Baca dari context parquet
        CONTEXT_FILE = ROOT / "data" / "Master_Market_Context_15Tahun.parquet"
        try:
            ihsg_filters = [
                ("Ticker", "==", "^JKSE"),
                ("Date", ">=", df_price.index.min())
            ]
            if end_date is not None:
                ihsg_filters.append(("Date", "<=", pd.Timestamp(end_date)))
                
            # Tarik HANYA data IHSG pada rentang tanggal yang diminta
            ihsg_data = pd.read_parquet(CONTEXT_FILE, filters=ihsg_filters)
            
            if not ihsg_data.empty:
                ihsg_data["Date"] = pd.to_datetime(ihsg_data["Date"])
                
            if not ihsg_data.empty:
                ihsg_data = ihsg_data.set_index("Date").sort_index()
                idx_series = ihsg_data["Close"].dropna()
                if not idx_series.empty:
                    ihsg_return = idx_series / idx_series.iloc[0] - 1
        except Exception as e:
            logger.warning(f"Gagal membaca IHSG dari Parquet: {e}")

    # 7. Gabungkan (inner join tanggal agar kedua garis selalu punya nilai)
    combined = pd.DataFrame({"portfolio_return": portfolio_return})
    if ihsg_return is not None:
        combined["ihsg_return"] = ihsg_return
        combined = combined.dropna()

    series = [
        {
            "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
            "portfolio_value": float(portfolio_value.loc[d]),
            "portfolio_return": float(row["portfolio_return"]),
            "ihsg_return": (
                float(row["ihsg_return"])
                if "ihsg_return" in row and pd.notna(row["ihsg_return"])
                else None
            ),
        }
        for d, row in combined.iterrows()
    ]

    end = combined.index.max() if len(combined) else df_price.index.max()

    # Snapshot lot aktif per ticker pada akhir rentang (untuk respons holdings).
    # Memakai rentang EFEKTIF hasil penyesuaian hari bursa di langkah 5 agar
    # konsisten dengan nilai portofolio yang dihitung.
    final_lots: dict[str, int] = {}
    for ticker, lot, _item_start, item_end in effective_items:
        if item_end is not None and item_end < end:
            continue  # sudah dijual sebelum akhir rentang
        final_lots[ticker] = final_lots.get(ticker, 0) + lot
    final_lots = {t: l for t, l in final_lots.items() if l > 0}

    if missing_price_tickers:
        logger.warning(
            "Performa portofolio %s: %d saham tidak punya data harga pada rentang "
            "ini sehingga TIDAK ikut dihitung (nilai portofolio lebih rendah dari "
            "seharusnya): %s",
            portfolio_id, len(missing_price_tickers), sorted(missing_price_tickers),
        )

    start_iso = start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date)
    requested_start_iso = (
        requested_start.isoformat() if hasattr(requested_start, "isoformat")
        else str(requested_start)
    )
    end_iso = end.isoformat() if hasattr(end, "isoformat") else str(end)

    return {
        # start_date = rentang yang BENAR-BENAR dipakai (bisa tergeser mundur ke
        # hari bursa terakhir bila date_ref portofolio jatuh pada hari non-bursa)
        "start_date": start_iso,
        # Tanggal yang DIMINTA (date_ref portofolio) + penanda penyesuaian, agar
        # frontend bisa menjelaskan bila awal garis berbeda dari tanggal generate.
        "requested_start_date": requested_start_iso,
        "start_date_adjusted": start_iso != requested_start_iso,
        "end_date": end_iso,
        "holdings": {t: final_lots[t] for t in sorted(final_lots)},
        # Saham yang tidak punya data harga pada rentang ini (tidak ikut
        # dihitung) -- biasanya baru pertama kali dibeli pada hari non-bursa
        # sehingga belum pernah ditarik oleh sync_market_data.
        "missing_price_tickers": sorted(missing_price_tickers),
        "series": series,
    }