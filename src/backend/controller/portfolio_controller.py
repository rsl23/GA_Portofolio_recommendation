import logging
from uuid import uuid4, UUID

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


class StockNotFoundError(Exception):
    """Exception domain: ticker dari market_data tidak terdaftar di stock_universe."""


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
            "item_id": "",  # diisi setelah item tersimpan (lihat langkah 6)
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

    item_pairs = _query_items_with_ticker(db, portofolio.id)
    items = [pair[0] for pair in item_pairs]

    # total_terpakai & sisa_budget dihitung ulang dari item (total_investasi
    # berbasis harga_beli milik user) — nilai kolom portofolio bisa basi
    # setelah user mengedit harga_beli lewat PATCH.
    total_terpakai = sum((it.total_investasi or 0.0) for it in items)
    sisa_budget = portofolio.budget - total_terpakai
    allocated_budget_ok = total_terpakai <= portofolio.budget

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
        sisa_budget=sisa_budget,
        n_active=len(items),
        allocated_budget_ok=allocated_budget_ok,
        risk_profile=portofolio.risk_profile,
        status_portofolio=portofolio.status_portofolio,
        created_at=portofolio.created_at,
        budget=portofolio.budget,
        allocations=[_to_item_response(item, ticker) for item, ticker in item_pairs],
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


def get_portfolio_performance(db: Session, user_id: str) -> dict:
    """
    Hitung performa (persentase return kumulatif harian) portofolio ACTIVE user
    vs IHSG, sejak tanggal portofolio aktif dibuat.

    Metode (berbasis KEPEMILIKAN LOT, bukan bobot investasi):
      nilai_portofolio(t) = SUM( jumlah_lot_i * adj_close_i(t) )
      return(t)           = nilai_portofolio(t) / nilai_portofolio(t0) - 1
    Kontribusi tiap emiten otomatis mengikuti pergerakan harganya, sehingga
    return portofolio total lebih akurat.

    Raises: PortfolioNotFoundError jika user belum punya portofolio aktif.
    """
    import pandas as pd
    from uuid import UUID

    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise PortfolioNotFoundError(f"User ID pada token tidak valid: {user_id}") from e

    # 1. Portofolio ACTIVE user — ambil yang TERBARU secara eksplisit
    #    (bukan MIN/MAX sembarang active), jadi basis perhitungan jelas.
    portfolio = (
        db.query(Portofolio)
        .filter(
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio == "active",
        )
        .order_by(Portofolio.created_at.desc(), Portofolio.id.desc())
        .first()
    )
    if portfolio is None:
        raise PortfolioNotFoundError("User belum memiliki portofolio aktif.")
    start_dt = portfolio.created_at
    start_date = start_dt.date() if hasattr(start_dt, "date") else start_dt

    # 2. Kepemilikan lot per emiten dari portofolio active tersebut (by id)
    holdings_rows = (
        db.query(
            PortofolioItem.stock_id,
            StockUniverse.ticker,
            PortofolioItem.jumlah_lot,
        )
        .join(Portofolio, PortofolioItem.portofolio_id == Portofolio.id)
        .join(StockUniverse, PortofolioItem.stock_id == StockUniverse.id_stock)
        .filter(PortofolioItem.portofolio_id == portfolio.id)
        .all()
    )
    if not holdings_rows:
        raise PortfolioNotFoundError("User belum memiliki saham di portofolio aktifnya.")

    stock_ids = [r.stock_id for r in holdings_rows]
    lots_map = {r.ticker: int(r.jumlah_lot) for r in holdings_rows}

    # 3. Harga per (date, stock_id) sejak start_date
    price_rows = (
        db.query(MarketData.stock_id, MarketData.date, MarketData.adj_close, MarketData.close)
        .filter(
            MarketData.stock_id.in_(stock_ids),
            MarketData.date >= start_date,
        )
        .all()
    )
    if not price_rows:
        raise PortfolioNotFoundError(
            "Belum ada data harga tersimpan untuk saham portofolio aktif. "
            "Jalankan sync-prices terlebih dahulu."
        )

    # 4. Vectorization: pivot date x stock (adj_close; fallback close bila adj kosong)
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
    # dropna eksplisit bertahap:
    # a) ffill: isi tanggal bolong per saham selama saham tsb sudah punya harga
    #    sebelumnya (mis. saham suspension 1 hari)
    df_price = df_price.ffill()
    # b) buang baris yang KOSONG SEMUA (hari libur bersama sebelum saham manapun
    #    punya harga — biasanya di awal rentang)
    df_price = df_price.dropna(how="all")
    # c) basis t0 harus tanggal di mana SEMUA saham portofolio sudah punya harga,
    #    supaya nilai portofolio t0 benar-benar mencakup seluruh kepemilikan
    df_price = df_price.dropna(axis=0, how="any")
    if df_price.empty:
        raise PortfolioNotFoundError("Data harga belum cukup untuk menghitung performa.")

    id_to_ticker = {str(r.stock_id): r.ticker for r in holdings_rows}
    df_price.columns = [id_to_ticker[c] for c in df_price.columns]

    # 5. Nilai portofolio harian = SUM(lots * price).
    #    PENTING: adj_close dari yfinance adalah harga PER LEMBAR, sedangkan
    #    jumlah_lot = 100 lembar, jadi konversi lot -> lembar (x 100).
    LOTS_PER_SHARE = 100
    lots = pd.Series({t: lots_map[t] * LOTS_PER_SHARE for t in df_price.columns if t in lots_map})
    df_price = df_price[list(lots.index)]
    portfolio_value = df_price.mul(lots, axis=1).sum(axis=1)
    portfolio_return = portfolio_value / portfolio_value.iloc[0] - 1

    # 6. Return IHSG sejak tanggal yang sama
    idx_rows = (
        db.query(IdxComposite)
        .filter(IdxComposite.date >= df_price.index.min())
        .order_by(IdxComposite.date.asc())
        .all()
    )
    ihsg_return = None
    if idx_rows:
        idx_series = pd.Series(
            {r.date: r.close for r in idx_rows if r.close is not None}
        ).sort_index()
        ihsg_return = idx_series / idx_series.iloc[0] - 1

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

    return {
        "start_date": start_dt.date().isoformat() if hasattr(start_dt, "date") else str(start_dt),
        "end_date": end.isoformat() if hasattr(end, "isoformat") else str(end),
        "holdings": {t: lots_map[t] for t in sorted(lots_map)},
        "series": series,
    }