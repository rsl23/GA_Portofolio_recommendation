import calendar
import logging
import os
from datetime import date, timedelta
from decimal import Decimal
import math
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
    RebalanceSeriesGenerateRequest,
    RebalanceSeriesResponse,
    SkippedRebalanceStep,
)
from src.backend.services.gemini_llm_caller import explain_portfolio
from src.gaengine.data_loader import build_market_data
from src.gaengine.engine import GeneticEngine
from src.gaengine.ga_config import GAConfig
from src.gaengine.rebalancer import (
    compute_nav,
    align_old_weights_to_market,
    apply_no_trade_band,
    fit_buys_to_available_cash,
    compute_rebalance_deltas,
)
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


def _money(value) -> Decimal:
    return Decimal(str(value))


def _as_date(value) -> date:
    """Normalize persisted DATE/DATETIME values before lineage comparisons."""
    if hasattr(value, "date"):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise ValueError(f"Tanggal snapshot tidak valid: {value!r}")


def _execution_close_prices(db: Session, tickers: set[str], requested: date, backtest: bool) -> tuple[date, dict[str, float]]:
    """Return one common trading day and actual (not forward-filled) raw closes."""
    import pandas as pd
    if not tickers:
        raise MarketDataUnavailableError("Tidak ada ticker untuk dieksekusi.")
    if backtest:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent.parent
        try:
            frame = pd.read_parquet(
                root / "data" / "Master_OHLCV_15Tahun.parquet",
                columns=["Date", "Ticker", "Close"],
                filters=[("Date", ">=", pd.Timestamp(requested - timedelta(days=14))),
                         ("Date", "<=", pd.Timestamp(requested)),
                         ("Ticker", "in", [f"{ticker}.JK" for ticker in tickers])],
            )
        except Exception as exc:
            raise MarketDataUnavailableError(f"Harga Close backtest tidak dapat dibaca: {exc}") from exc
        if frame.empty:
            raise MarketDataUnavailableError(f"Tidak ada harga Close hingga {requested}.")
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
        effective = frame["Date"].max()
        day_rows = frame[frame["Date"] == effective]
        prices = {str(row.Ticker).removesuffix(".JK"): float(row.Close)
                  for row in day_rows.itertuples() if pd.notna(row.Close)}
    else:
        stocks = db.query(StockUniverse).filter(StockUniverse.ticker.in_(tickers)).all()
        ids = {stock.id_stock: stock.ticker for stock in stocks}
        if len(ids) != len(tickers):
            raise StockNotFoundError("Ticker eksekusi tidak lengkap dalam stock_universe.")
        rows = db.query(MarketData).filter(
            MarketData.stock_id.in_(ids), MarketData.date <= requested
        ).order_by(MarketData.date.desc()).all()
        if not rows:
            raise MarketDataUnavailableError("Harga Close live belum tersimpan.")
        effective = rows[0].date
        prices = {ids[row.stock_id]: float(row.close) for row in rows
                  if row.date == effective and row.close is not None}
    missing = sorted(ticker for ticker in tickers
                     if ticker not in prices or not math.isfinite(prices[ticker]) or prices[ticker] <= 0)
    if missing:
        raise MarketDataUnavailableError(
            f"Harga Close aktual pada {effective} tidak tersedia untuk: {', '.join(missing)}."
        )
    return effective, prices


def _backtest_trading_day(requested: date) -> date:
    """Resolve the EOD decision/close-fill date before loading any GA inputs."""
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent.parent
    try:
        days = pd.read_parquet(
            root / "data" / "Master_OHLCV_15Tahun.parquet",
            columns=["Date"],
            filters=[("Date", ">=", pd.Timestamp(requested - timedelta(days=14))),
                     ("Date", "<=", pd.Timestamp(requested))],
        )
    except Exception as exc:
        raise MarketDataUnavailableError(f"Kalender perdagangan backtest tidak dapat dibaca: {exc}") from exc
    if days.empty:
        raise MarketDataUnavailableError(f"Tidak ada hari bursa hingga {requested}.")
    return pd.to_datetime(days["Date"]).dt.date.max()


def _close_previous_items(db: Session, items: list[tuple[PortofolioItem, str]], execution_day: date) -> None:
    """End all old snapshot positions atomically with the replacement snapshot."""
    from datetime import datetime

    execution_dt = datetime.combine(execution_day, datetime.min.time())
    for item, ticker in items:
        if item.end_date is not None:
            db.rollback()
            raise ValueError(f"Item snapshot sebelumnya {ticker} sudah memiliki end_date.")
    for item, _ in items:
        item.end_date = execution_dt


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
        # Signals use the previous completed session. The scheduled session's
        # Close is an execution fill, never an input to the GA decision.
        execution_target = _backtest_trading_day(date_ref)
        decision_day = _backtest_trading_day(execution_target - timedelta(days=1))
        logger.info("Membentuk MarketData BACKTEST per hari bursa %s...", decision_day)
        try:
            market_data = build_market_data(
                min_price=1.0, backtest=True, date_ref=decision_day
            )
        except Exception as e:
            raise MarketDataUnavailableError(
                f"Gagal merakit MarketData backtest untuk {decision_day}: {e}"
            ) from e
    elif market_data is None:
        logger.info("Membentuk MarketData menggunakan data LIVE...")
        market_data = build_market_data(min_price=1.0)

    if market_data is None or market_data.n_stocks == 0:
        raise MarketDataUnavailableError(
            "Gagal membentuk MarketData. Pastikan tabel filtered_stock_cache sudah terisi "
            "(live) atau tanggal backtest memiliki data (backtest)."
        )

    # --- REBALANCING LOGIC ---
    old_portfolio_id = None
    old_lots = {}
    old_weights_aligned = None
    old_signal_prices_per_lot: dict[str, float] = {}
    old_cost_basis: dict[str, float | None] = {}
    eliminated_stocks = []
    execution_day = None
    all_close_prices: dict[str, float] = {}
    ga_budget = request.budget
    
    if getattr(request, 'rebalance', False):
        try:
            old_portfolio_res = get_active_portfolio(db, str(user_uuid), backtest=backtest)
            old_portfolio_id = old_portfolio_res.id
            for alloc in old_portfolio_res.allocations:
                old_lots[alloc.ticker] = alloc.lots
            
            # Validasi khusus BACKTEST: tanggal simulasi baru tidak boleh lebih
            # awal dari portofolio backtest aktif yang akan di-rebalance.
            # Snapshot lama ditutup pada execution_date, bukan pada tanggal
            # jadwal/date_ref yang bisa jatuh pada hari non-bursa.
            # Mode LIVE aman secara alami karena date_ref_dt = sekarang.
            if backtest and date_ref is not None:
                old_dt = old_portfolio_res.date_ref or old_portfolio_res.created_at
                old_date = old_dt.date() if hasattr(old_dt, "date") else old_dt
                if old_date is not None and date_ref < old_date:
                    raise ValueError(
                        f"Tanggal backtest ({date_ref}) lebih awal dari portofolio "
                        f"backtest aktif ({old_date}). Rebalancing backtest hanya "
                        f"boleh pada tanggal yang sama atau setelah tanggal "
                        f"pembuatan portofolio lama."
                    )

            if (old_portfolio_res.sisa_budget is None or
                not math.isfinite(old_portfolio_res.sisa_budget) or
                old_portfolio_res.sisa_budget < -0.01):
                raise ValueError("Kas portofolio lama tidak valid untuk rebalance.")
            leftover_cash = float(old_portfolio_res.sisa_budget)
            execution_day, all_close_prices = _execution_close_prices(
                db, set(old_lots) | set(market_data.stock_codes),
                execution_target if backtest else date.today(), backtest,
            )
            if backtest and execution_day != execution_target:
                raise MarketDataUnavailableError("Hari harga eksekusi berbeda dari jadwal bursa efektif.")
            old_snapshot_row = db.query(Portofolio).filter(
                Portofolio.id == UUID(str(old_portfolio_id)), Portofolio.user_id == user_uuid
            ).first()
            if old_snapshot_row is not None:
                old_cost_basis = _rebalance_cost_basis(db, old_snapshot_row, user_uuid)
            previous_execution_day = (
                old_snapshot_row.execution_date if old_snapshot_row is not None else None
            ) or old_portfolio_res.date_ref or old_portfolio_res.created_at
            previous_execution_day = _as_date(previous_execution_day)
            if execution_day <= previous_execution_day:
                raise MarketDataUnavailableError(
                    f"Hari eksekusi {execution_day} harus sesudah snapshot sebelumnya ({previous_execution_day})."
                )
            current_prices = {code: price * 100.0 for code, price in all_close_prices.items()}
            
            # Find eliminated stocks
            for code in old_lots:
                if code not in market_data.stock_codes:
                    eliminated_stocks.append(code)
            
            old_weights_aligned, new_nav = align_old_weights_to_market(
                old_lots, current_prices, leftover_cash, market_data.stock_codes
            )
            if backtest:
                signal_day, signal_closes = _execution_close_prices(
                    db, set(old_lots) | set(market_data.stock_codes), decision_day, True,
                )
                if signal_day != decision_day:
                    raise MarketDataUnavailableError("Harga sinyal tidak tersedia pada hari keputusan.")
                old_signal_prices_per_lot = {
                    code: price * 100.0 for code, price in signal_closes.items()
                }
                old_weights_aligned, ga_budget = align_old_weights_to_market(
                    old_lots, old_signal_prices_per_lot,
                    leftover_cash, market_data.stock_codes,
                )
            else:
                old_signal_prices_per_lot = {
                    code: price * 100.0 for code, price in all_close_prices.items()
                }
                ga_budget = new_nav
            
            # Override budget with NAV
            request.budget = new_nav
            logger.info("Rebalancing mode. NAV computed as Rp%.2f", new_nav)
            
        except PortfolioNotFoundError:
            # Pesan disesuaikan mode agar user tahu portofolio MODE MANA yang
            # kurang (live vs backtest) -- keduanya bisa direbalance.
            raise ValueError(
                "Tidak dapat melakukan rebalancing karena tidak ada portofolio "
                + ("backtest aktif" if backtest else "live aktif")
                + ". Silakan generate portofolio terlebih dahulu pada mode ini."
            )

    if execution_day is None:
        execution_day, all_close_prices = _execution_close_prices(
            db, set(market_data.stock_codes),
            execution_target if backtest else date.today(), backtest,
        )
        if backtest and execution_day != execution_target:
            raise MarketDataUnavailableError("Hari harga eksekusi berbeda dari jadwal bursa efektif.")
    # Backtest GA sizes positions with the prior session's prices, while the
    # simulated order and post-trade allocation use the execution-day Close.
    ga_prices = ({code: _money(all_close_prices[code]) for code in market_data.stock_codes}
                 if backtest else
                 {code: _money(price) / 100 for code, price in
                  zip(market_data.stock_codes, market_data.prices_per_lot)})
    if any(not price.is_finite() or price <= 0 for price in ga_prices.values()):
        raise MarketDataUnavailableError("Harga kandidat GA tidak valid untuk eksekusi.")

    # 2. Konfigurasi GA mengikuti profil & modal dari payload request
    config = GAConfig(
        population_size=200,        # jumlah kromosom
        generations=500,            # jumlah generasi evolusi
        budget=ga_budget,            # NAV known when the GA decision is made
        risk_profile=request.risk_profile,
        min_stocks=3,
        max_stocks=10,
        risk_free_rate=market_data.risk_free_rate,
        # seed=42,
        data=market_data,
        is_rebalance=getattr(request, 'rebalance', False),
        old_weights=old_weights_aligned,
        old_lots=old_lots,
        old_prices_per_lot=old_signal_prices_per_lot,
        old_cost_basis=old_cost_basis,
    )

    # 3. Evolusi genetika
    logger.info("=== MEMULAI EVOLUSI GENETIKA ===")
    engine = GeneticEngine(config)
    solution = engine.run(verbose=True)
    if not solution.diversification_ok or not solution.budget_ok:
        raise ValueError("GA tidak menghasilkan portofolio yang memenuhi batas saham dan budget.")
    
    # --- APPLY NO TRADE BAND ---
    if getattr(request, 'rebalance', False):
        if config.allow_cash:
            full_weights = (np.asarray(solution.genes) * solution.exposure).tolist()
            new_weights = apply_no_trade_band(
                full_weights, old_weights_aligned, threshold=0.03,
                min_active=config.min_stocks, max_active=config.max_stocks,
            )
            solution.exposure = float(np.clip(
                sum(new_weights), config.min_exposure, config.max_exposure
            ))
            solution.genes = new_weights
        else:
            solution.genes = apply_no_trade_band(
                solution.genes, old_weights_aligned, threshold=0.03,
                min_active=config.min_stocks, max_active=config.max_stocks,
            )
        solution.normalize()
        # Re-evaluate fitness to update metrics after applying no-trade band
        solution.evaluate_fitness()
        if not solution.diversification_ok or not solution.budget_ok:
            raise ValueError("No-trade band melanggar batas saham atau budget.")

    # The GA only knows decision-day prices in backtests. A higher execution
    # Close can make the intended BUYs unaffordable; retain every planned sale
    # and trim lowest-priority BUY lots before the snapshot is persisted.
    desired_lots = {code: int(qty) for code, qty in
                    zip(market_data.stock_codes, solution.lots) if qty > 0}
    affordable_lots, _ = fit_buys_to_available_cash(
        old_lots, desired_lots,
        float(old_portfolio_res.sisa_budget) if old_portfolio_id else float(request.budget),
        all_close_prices,
        dict(zip(market_data.stock_codes, solution.genes)), config.min_stocks,
    )
    if affordable_lots != desired_lots:
        logger.info("Menyesuaikan BUY lot ke kas tersedia pada harga eksekusi %s", execution_day)
    solution.evaluate_fitness(lots_override=np.asarray(
        [affordable_lots.get(code, 0) for code in market_data.stock_codes], dtype=int
    ))
    if not solution.diversification_ok or not solution.budget_ok:
        raise ValueError("Lot eksekusi melanggar batas saham atau budget.")

    # 4. Susun laporan hasil
    codes = market_data.stock_codes
    prices = (np.asarray([float(ga_prices[code] * 100) for code in codes], dtype=float)
              if backtest else market_data.prices_per_lot)
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
            "harga_beli": price / 100.0,  # per lembar, sama seperti kolom DB/PATCH
            "allocation": float(lot * price),
            "weight": float(lot * price / total) if total else 0.0,
        }
        for code, lot, price in rows
        if lot > 0
    ]
    
    rebalance_deltas = None
    if getattr(request, 'rebalance', False):
        new_lots_dict = {code: int(lot) for code, lot in zip(codes, lots)}
        new_weights_dict = {code: float(w) for code, w in zip(codes, solution.genes)}
        old_weights_dict = {code: w for code, w in zip(market_data.stock_codes, old_weights_aligned)}
        
        rebalance_deltas = compute_rebalance_deltas(
            old_lots=old_lots,
            new_lots=new_lots_dict,
            old_weights=old_weights_dict,
            new_weights=new_weights_dict,
            eliminated_stocks=eliminated_stocks,
            old_cost_basis=old_cost_basis,
            execution_prices=all_close_prices,
        )

    # Estimasi return tahunan dari alokasi terhadap modal termasuk sisa kas.
    expected_return = None
    if total > 0 and market_data.returns.shape[0] >= solution.n_active:
        aw = alloc / float(request.budget)
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
        rebalance_deltas=rebalance_deltas,
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

    new_lots_for_accounting = {code: int(lot) for code, lot, _ in rows if lot > 0}
    changed_tickers = {code for code in set(old_lots) | set(new_lots_for_accounting)
                       if old_lots.get(code, 0) != new_lots_for_accounting.get(code, 0)}
    execution_prices = {
        code: str(ga_prices.get(code, _money(all_close_prices[code])))
        for code in sorted(changed_tickers)
    }
    cash_before = _money(old_portfolio_res.sisa_budget if old_portfolio_id else request.budget)
    cash_after = cash_before + sum(
        (_money(old_lots.get(code, 0)) - _money(new_lots_for_accounting.get(code, 0)))
        * 100 * _money(execution_prices[code]) for code in changed_tickers
    )
    holdings_after = sum(_money(lot) * 100 * ga_prices[code]
                         for code, lot in new_lots_for_accounting.items())
    if cash_after < Decimal("-0.01"):
        raise ValueError("Rebalance menghasilkan kas negatif; snapshot dibatalkan.")
    # budget is the pre-trade NAV at the valuation mark. If an actual/simulated
    # execution differs from that mark, the difference is trading slippage,
    # not an external cash flow or an invalid snapshot.

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
        total_terpakai=float(holdings_after),
        sisa_budget=float(cash_after),
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
        is_rebalance=getattr(request, 'rebalance', False),
        parent_portofolio_id=old_portfolio_id if getattr(request, 'rebalance', False) else None,
        turnover_rate=float(solution.turnover_rate) if getattr(request, 'rebalance', False) else None,
        turnover_penalty_beta=float(config.turnover_penalty_beta) if getattr(request, 'rebalance', False) else None,
        # acuan tanggal portofolio: tanggal simulasi (backtest) / sekarang (live)
        date_ref=date_ref_dt,
        execution_date=execution_day,
        execution_prices=execution_prices,
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

    delta_map = {d["kode"]: d for d in (rebalance_deltas or [])}
    
    # Close every item in the previous position snapshot on the effective
    # execution day, including HOLD and partial SELL. The new snapshot starts
    # on that same day; windows are interpreted as [start_date, end_date).
    old_item_map = {}
    if getattr(request, 'rebalance', False) and old_portfolio_id:
        old_items_with_ticker = _query_items_with_ticker(db, old_portfolio_id)
        _close_previous_items(db, old_items_with_ticker, execution_day)
        for old_item, tick in old_items_with_ticker:
            old_item_map[tick] = old_item

    items = []
    for ticker, lot, price in rows:
        if lot > 0:
            action = "buy"
            start_dt = _dt.combine(execution_day, _dt.min.time())
            harga_beli = price / 100.0
            
            if getattr(request, 'rebalance', False) and ticker in delta_map:
                action = delta_map[ticker]["action"].lower()
                if ticker in old_item_map:
                    # A held position gets a new snapshot window too; only its
                    # purchase basis may carry over from the old item.
                    old_item = old_item_map[ticker]
                    if action == "hold" or action == "sell":
                        harga_beli = old_item.harga_beli
            
            items.append(
                PortofolioItem(
                    portofolio_id=portofolio.id,
                    stock_id=ticker_to_id[ticker],
                    bobot_persentase=float((lot * price / total) * 100.0) if total else 0.0,
                    jumlah_lot=lot,
                    # price dari market_data adalah harga PER LOT (100 lembar);
                    # kolom harga_acuan & harga_beli disimpan PER LEMBAR
                    harga_acuan=price / 100.0,
                    harga_beli=harga_beli,
                    total_investasi=float(lot * price),
                    action_type=action,
                    start_date=start_dt,
                    end_date=None,
                )
            )
            
    db.add_all(items)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(portofolio)
    logger.info("Portofolio %s berhasil disimpan dengan %d item.", portofolio.id, len(items))

    # Isi item_id pada allocations (item sudah tersimpan -> id tersedia)
    saved_rows = [r for r in rows if r[1] > 0]
    saved_item_map = {r[0]: item for r, item in zip(saved_rows, items)}
    for a in allocations:
        item = saved_item_map[a["ticker"]]
        a["item_id"] = str(item.id)
        a["harga_beli"] = item.harga_beli

    return PortfolioResponse(
        id=str(portofolio.id),
        user_id=str(user_uuid),
        fitness_score=float(solution.fitness),
        sharpe_ratio=float(solution.sharpe_ratio),
        expected_return=expected_return,
        max_drawdown=float(solution.max_drawdown),
        avg_correlation=float(solution.avg_correlation),
        skor_fundamental=float(solution.skor_fundamental),
        total_terpakai=float(holdings_after),
        sisa_budget=float(cash_after),
        n_active=int(solution.n_active),
        allocated_budget_ok=cash_after >= Decimal("-0.01"),
        risk_profile=request.risk_profile,
        status_portofolio=portofolio.status_portofolio,
        created_at=portofolio.created_at,
        date_ref=portofolio.date_ref,
        budget=request.budget,
        allocations=allocations,
        narasi_llm=narasi,
        rebalance_deltas=rebalance_deltas,
    )


# ----------------------------------------------------------------------
# Rebalance series: generate awal + rebalance berantai tiap N bulan
# ----------------------------------------------------------------------
# Batas jumlah tanggal (langkah) per series. Tiap langkah menjalankan GA penuh
# (200 populasi x 500 generasi) + narasi LLM, sehingga series yang sangat
# panjang butuh waktu lama -- batas ini mencegah request tak terkendali.
MAX_SERIES_STEPS = 60


def _add_months(d: date, months: int) -> date:
    """Tambah N bulan ke tanggal, aman terhadap akhir-bulan
    (mis. 31 Jan + 1 bulan -> 28/29 Feb, bukan error/rollover)."""
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def build_rebalance_schedule(
    start_date: date, end_date: date, interval_months: int
) -> list[date]:
    """Susun jadwal tanggal series: start_date, +interval, +2*interval, ...
    sampai melewati end_date (inklusif).

    Jumlah tanggal TIDAK angka pasti -- tergantung rentang dan interval, mis.
    2022-01-01 s/d 2026-12-31 tiap 2 bulan -> 30 tanggal (30 portofolio).

    Raises: ValueError bila start_date > end_date atau interval < 1.
    """
    if start_date > end_date:
        raise ValueError(
            f"start_date ({start_date}) tidak boleh setelah end_date ({end_date})."
        )
    if interval_months < 1:
        raise ValueError("interval_months harus >= 1.")

    schedule: list[date] = []
    current = start_date
    while current <= end_date:
        schedule.append(current)
        current = _add_months(current, interval_months)
    return schedule


def generate_rebalance_series(
    db: Session,
    request: RebalanceSeriesGenerateRequest,
    user_id: str,
) -> RebalanceSeriesResponse:
    """
    Controller REBALANCE SERIES (khusus BACKTEST): menjalankan generate awal
    lalu rebalance BERANTAI berulang pada banyak tanggal sekaligus.

    Alur (eksekusi SEKUENSIAL, bukan paralel -- tiap langkah bergantung pada
    hasil langkah sebelumnya: NAV, old_lots, dan status active_backtest):
      1. langkah pertama (start_date) : generate portofolio AWAL (rebalance=false)
      2. langkah berikutnya (+interval tiap N bulan sampai end_date):
         rebalance (rebalance=true) dari portofolio aktif hasil langkah
         sebelumnya -- NAV dihitung ulang dari harga pada tanggal tsb, rantai
         induk tercatat di parent_portofolio_id.

    Semua langkah memakai mode backtest (date_ref = tanggal jadwal, data
    historis parquet) sehingga portofolio LIVE tidak tersentuh. Portofolio
    terakhir berstatus "active_backtest", langkah sebelumnya menjadi
    "replaced_backtest" (tetap tersimpan di DB).

    Kegagalan per langkah:
      - langkah AWAL gagal -> seluruh series gagal (tidak ada yang direbalance)
      - langkah berikutnya gagal MarketDataUnavailableError (mis. tanggal di
        luar cakupan parquet) -> dicatat pada `skipped`, series LANJUT dari
        portofolio terakhir yang berhasil
      - error lain (ticker tak terdaftar, SQL, dll.) -> series berhenti dan
        error diteruskan; portofolio yang sudah terlanjur commit TIDAK dihapus

    Returns: RebalanceSeriesResponse berisi SEMUA portofolio berhasil, urut
    kronologis dari start_date (jumlahnya mengikuti jadwal, BUKAN angka pasti),
    plus daftar tanggal yang dilewati beserta alasannya.

    Raises: ValueError (rentang salah / jumlah langkah > MAX_SERIES_STEPS),
            UserNotFoundError, MarketDataUnavailableError (langkah awal),
            StockNotFoundError.
    """
    schedule = build_rebalance_schedule(
        request.start_date, request.end_date, request.interval_months
    )
    if len(schedule) > MAX_SERIES_STEPS:
        raise ValueError(
            f"Jumlah langkah series ({len(schedule)}) melebihi batas "
            f"{MAX_SERIES_STEPS}. Perbesar interval_months atau perpendek rentang."
        )

    logger.info(
        "=== REBALANCE SERIES (backtest): %d tanggal, %s s/d %s, interval %d bulan ===",
        len(schedule), request.start_date, request.end_date, request.interval_months,
    )

    portfolios: list[PortfolioResponse] = []
    skipped: list[SkippedRebalanceStep] = []

    for index, tanggal in enumerate(schedule):
        is_initial = index == 0
        # Request BARU tiap langkah: generate_new_portfolio memutasi budget
        # (override NAV) pada objek request, jadi tidak boleh dipakai ulang.
        step_request = PortfolioGenerateRequest(
            budget=request.budget,
            risk_profile=request.risk_profile,
            backtest=True,
            date_ref=tanggal,
            rebalance=not is_initial,  # langkah 1 = generate awal, sisanya rebalance
        )
        try:
            hasil = generate_new_portfolio(
                db,
                step_request,
                user_id=user_id,
                market_data=None,
                backtest=True,
                date_ref=tanggal,
            )
        except MarketDataUnavailableError as e:
            if is_initial:
                raise  # tanpa portofolio awal, series mustahil jalan
            db.rollback()
            logger.warning(
                "Rebalance series: langkah %d/%d (%s) dilewati -> %s",
                index + 1, len(schedule), tanggal, e,
            )
            skipped.append(SkippedRebalanceStep(date=tanggal, reason=str(e)))
            continue

        portfolios.append(hasil)
        logger.info(
            "Rebalance series: langkah %d/%d selesai (%s, rebalance=%s, id=%s).",
            index + 1, len(schedule), tanggal, not is_initial, hasil.id,
        )

    logger.info(
        "Rebalance series selesai: %d portofolio berhasil, %d tanggal dilewati.",
        len(portfolios), len(skipped),
    )
    return RebalanceSeriesResponse(
        start_date=request.start_date,
        end_date=request.end_date,
        interval_months=request.interval_months,
        scheduled_dates=schedule,
        portfolios=portfolios,
        skipped=skipped,
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
    harga_acuan/harga_beli di DB per LEMBAR; allocation memakai mark GA,
    bukan total_investasi yang dapat berubah ketika harga_beli diedit."""
    total = (item.jumlah_lot or 0) * 100.0 * (item.harga_acuan or 0.0)
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
    respons selalu identik. Kas dan alokasi snapshot harus berasal dari kolom
    Portofolio yang disimpan saat eksekusi; perubahan harga_beli milik user
    tidak merupakan BUY/SELL atau perubahan kas.
    """
    item_pairs = _query_items_with_ticker(db, portofolio.id)
    items = [pair[0] for pair in item_pairs]

    return PortfolioResponse(
        id=str(portofolio.id),
        user_id=str(portofolio.user_id),
        fitness_score=portofolio.fitness_score,
        sharpe_ratio=portofolio.sharpe_ratio,
        expected_return=None,  # tidak disimpan di DB (hanya dihitung saat generate)
        max_drawdown=portofolio.max_drawdown,
        avg_correlation=portofolio.avg_correlation,
        skor_fundamental=portofolio.skor_fundamental,
        total_terpakai=portofolio.total_terpakai,
        sisa_budget=portofolio.sisa_budget,
        n_active=len(items),
        allocated_budget_ok=portofolio.sisa_budget >= -0.01,
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

    # Kas dan alokasi tetap memakai snapshot tersimpan, bukan cost edit user.
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
            "allocation": (item.jumlah_lot or 0) * 100.0 * (item.harga_acuan or 0.0),
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
        allocated_budget_ok=portfolio.sisa_budget >= -0.01,
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

    db.commit()
    db.refresh(item)
    logger.info(
        "harga_beli item %s diubah menjadi Rp%.2f (basis pengguna Rp%.2f; kas tidak berubah)",
        item_id, harga_beli, item.total_investasi,
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



# Retained temporarily as a reference for the prior item-window calculation.
# The public ``get_portfolio_performance`` defined below supersedes it with
# snapshot/NAV accounting; keeping this private function avoids an unrelated,
# high-risk deletion while the rebalance feature is being integrated.
def _get_portfolio_performance_legacy(
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


# ----------------------------------------------------------------------
# NAV accounting performance (portfolio snapshots / rebalance chain)
# ----------------------------------------------------------------------
def _portfolio_lineage(db: Session, portfolio: Portofolio, user_uuid: UUID) -> list[Portofolio]:
    """Return the root-to-leaf snapshot chain for ``portfolio``.

    A rebalance creates a new ``Portofolio`` whose ``parent_portofolio_id``
    points to the previous snapshot.  The item rows are therefore *position
    snapshots*, not an append-only transaction ledger.  Walking this chain is
    the only reliable history currently stored by the schema.
    """
    chain: list[Portofolio] = []
    seen: set[UUID] = set()
    current: Portofolio | None = portfolio
    target_backtest = portfolio.status_portofolio in (STATUS_ACTIVE_BACKTEST, STATUS_REPLACED_BACKTEST)
    while current is not None:
        if current.id in seen:
            raise PortfolioNotFoundError("Rantai parent_portofolio_id mengandung siklus.")
        if current.user_id != user_uuid:
            raise PortfolioNotFoundError("Rantai portofolio tidak valid untuk pengguna ini.")
        if (current.status_portofolio in (STATUS_ACTIVE_BACKTEST, STATUS_REPLACED_BACKTEST)) != target_backtest:
            raise PortfolioNotFoundError("Rantai portofolio mencampur mode live dan backtest.")
        seen.add(current.id)
        chain.append(current)
        if current.parent_portofolio_id is None:
            break
        current = (
            db.query(Portofolio)
            .filter(
                Portofolio.id == current.parent_portofolio_id,
                Portofolio.user_id == user_uuid,
            )
            .first()
        )
        if current is None:
            raise PortfolioNotFoundError("Induk portofolio pada riwayat rebalance tidak ditemukan.")
    return list(reversed(chain))


def _snapshot_positions(db: Session, portfolio_id: UUID) -> dict[str, dict]:
    """Read GA marks and identifiable live cost-basis edits."""
    rows = _query_items_with_ticker(db, portfolio_id)
    return {
        ticker: {
            "stock_id": item.stock_id,
            "lots": int(item.jumlah_lot or 0),
            # harga_acuan is the allocation mark saved when GA created this
            # snapshot. PATCH changes total_investasi to lots * harga_beli;
            # generated snapshots keep it at lots * harga_acuan instead.
            "reference_price": float(item.harga_acuan) if item.harga_acuan is not None else None,
            "purchase_price": float(item.harga_beli) if item.harga_beli is not None else None,
            "item_total": float(item.total_investasi) if item.total_investasi is not None else None,
            "start_date": _as_date(item.start_date) if item.start_date is not None else None,
            "end_date": _as_date(item.end_date) if item.end_date is not None else None,
        }
        for item, ticker in rows
        if int(item.jumlah_lot or 0) > 0
    }


def _snapshot_date(portfolio: Portofolio) -> date:
    value = portfolio.date_ref or portfolio.created_at
    return _as_date(value)


def _apply_position_snapshot(
    positions: dict[str, int],
    cost_basis: dict[str, float | None],
    next_positions: dict[str, int],
    execution_prices: dict[str, float | None],
    realized_pnl: float | None,
) -> tuple[dict[str, float | None], float | None]:
    """Apply a position snapshot using weighted-average cost basis.

    Lots are converted to shares only when computing P/L.  A missing execution
    price/cost basis makes P/L unknown (``None``), rather than reporting a
    financially misleading zero.
    """
    next_basis = dict(cost_basis)
    for ticker in set(positions) | set(next_positions):
        previous_lots, current_lots = positions.get(ticker, 0), next_positions.get(ticker, 0)
        delta_lots = current_lots - previous_lots
        price = execution_prices.get(ticker)
        if delta_lots < 0:
            basis = next_basis.get(ticker)
            if price is None or basis is None or realized_pnl is None:
                realized_pnl = None
            else:
                realized_pnl += (-delta_lots) * 100.0 * (price - basis)
        if delta_lots > 0:
            old_basis = next_basis.get(ticker)
            if price is None:
                next_basis[ticker] = None
            elif previous_lots <= 0:
                next_basis[ticker] = price
            elif old_basis is None:
                next_basis[ticker] = None
            else:
                next_basis[ticker] = (
                    previous_lots * old_basis + delta_lots * price
                ) / current_lots
        if current_lots == 0:
            next_basis.pop(ticker, None)
    return next_basis, realized_pnl


def _rebalance_cost_basis(
    db: Session, portfolio: Portofolio, user_uuid: UUID,
) -> dict[str, float | None]:
    """Replay recorded lot deltas to recover the held positions' average cost.

    Legacy fills without execution prices stay unknown. Live PATCH corrections
    are applied to the remaining lots, matching the performance endpoint.
    This is advisory GA input; it never rewrites cash or historical snapshots.
    """
    is_backtest = portfolio.status_portofolio in (
        STATUS_ACTIVE_BACKTEST, STATUS_REPLACED_BACKTEST
    )
    positions: dict[str, int] = {}
    basis: dict[str, float | None] = {}
    for node in _portfolio_lineage(db, portfolio, user_uuid):
        items = _snapshot_positions(db, node.id)
        next_positions = {code: item["lots"] for code, item in items.items()}
        raw_prices = node.execution_prices if isinstance(node.execution_prices, dict) else {}
        prices: dict[str, float | None] = {}
        for code in set(positions) | set(next_positions):
            if positions.get(code, 0) == next_positions.get(code, 0):
                continue
            try:
                price = float(raw_prices[code])
            except (KeyError, TypeError, ValueError, OverflowError):
                price = None
            prices[code] = price if price is not None and math.isfinite(price) and price > 0 else None
        basis, _ = _apply_position_snapshot(positions, basis, next_positions, prices, 0.0)
        if not is_backtest:
            for code, item in items.items():
                purchase = item["purchase_price"]
                reference = item["reference_price"]
                item_total = item["item_total"]
                if (purchase is None or reference is None or item_total is None or
                    not all(math.isfinite(x) and x > 0 for x in (purchase, reference, item_total))):
                    continue
                shares = item["lots"] * 100
                if (abs(item_total - shares * purchase) <= 0.01 and
                    abs(shares * purchase - shares * reference) > 0.01):
                    basis[code] = purchase
        positions = next_positions
    return basis


def get_portfolio_performance(
    db: Session,
    user_id: str,
    portfolio_id: str,
    end_date: date | None = None,
    backtest: bool = False,
) -> dict:
    """Calculate performance from NAV, including every stored rebalance.

    ``PortofolioItem`` has no transaction ledger.  The implementation uses the
    existing parent chain as dated position snapshots. Lot deltas and recorded
    execution prices drive cash and weighted-average basis; live PATCH edits
    may correct the remaining position's basis without moving cash. Persisted
    cash is a reconciliation check. Raw Close is a daily valuation mark.

    Fees, taxes, dividends, deposits/withdrawals, and explicit corporate-action
    events are not represented by current models. New simulations explicitly
    assume zero transaction costs; other untracked components stay unknown.
    """
    import pandas as pd

    try:
        user_uuid, portfolio_uuid = UUID(user_id), UUID(portfolio_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise PortfolioNotFoundError(f"ID tidak valid: {exc}") from exc

    portfolio = (
        db.query(Portofolio)
        .filter(Portofolio.id == portfolio_uuid, Portofolio.user_id == user_uuid)
        .first()
    )
    if portfolio is None:
        raise PortfolioNotFoundError(
            f"Portofolio {portfolio_id} tidak ditemukan (atau bukan milik Anda)."
        )

    is_backtest = portfolio.status_portofolio in (
        STATUS_ACTIVE_BACKTEST,
        STATUS_REPLACED_BACKTEST,
    )
    if backtest and not is_backtest:
        raise PortfolioNotFoundError("Portofolio live tidak dapat dihitung sebagai backtest.")
    lineage = _portfolio_lineage(db, portfolio, user_uuid)
    snapshots = [
        {
            "date": _as_date(node.execution_date) if node.execution_date is not None else _snapshot_date(node),
            "requested_date": _snapshot_date(node),
            "id": str(node.id),
            "budget": float(node.budget),
            "total_terpakai": float(node.total_terpakai),
            "cash": float(node.sisa_budget) if node.sisa_budget is not None else None,
            "positions": _snapshot_positions(db, node.id),
            "execution_prices": node.execution_prices,
        }
        for node in lineage
    ]
    if not snapshots:
        raise PortfolioNotFoundError("Rantai portofolio kosong.")
    if any(left["date"] > right["date"] for left, right in zip(snapshots, snapshots[1:])):
        raise PortfolioNotFoundError("Tanggal eksekusi rantai portofolio tidak berurutan.")
    if end_date is not None and end_date < snapshots[-1]["date"]:
        raise PortfolioNotFoundError("end_date lebih awal dari snapshot portofolio yang diminta.")

    # Item windows are half-open. A historical snapshot must not be valued
    # after its positions were closed by a later rebalance that the requested
    # portfolio ID does not include.
    closed_dates = [data.get("end_date") for data in snapshots[-1]["positions"].values()
                    if data.get("end_date") is not None]
    valuation_end = end_date
    if closed_dates:
        last_active_day = min(closed_dates) - timedelta(days=1)
        if last_active_day < snapshots[-1]["date"]:
            raise PortfolioNotFoundError("Item snapshot berakhir sebelum tanggal eksekusinya.")
        valuation_end = min(valuation_end, last_active_day) if valuation_end is not None else last_active_day

    requested_start = snapshots[0]["requested_date"]
    price_query_start = snapshots[0]["date"] - timedelta(days=14)
    stock_id_to_ticker = {
        str(position["stock_id"]): ticker
        for snapshot in snapshots
        for ticker, position in snapshot["positions"].items()
    }
    stock_ids = list({position["stock_id"] for snapshot in snapshots for position in snapshot["positions"].values()})

    # Load only the stored universe required by the complete rebalance chain.
    sync_stats = None
    if not is_backtest:
        price_rows = _query_price_rows(db, stock_ids, price_query_start, valuation_end)
        if not price_rows:
            sync_stats = _sync_prices_on_demand(db, reason=f"data performa {portfolio_id}")
            price_rows = _query_price_rows(db, stock_ids, price_query_start, valuation_end)
        if not price_rows:
            fallback = _latest_available_price_date(db, stock_ids, valuation_end)
            if fallback is not None and fallback < price_query_start:
                price_query_start = fallback
                price_rows = _query_price_rows(db, stock_ids, price_query_start, valuation_end)
        if not price_rows:
            raise PortfolioNotFoundError(
                "Belum ada data harga tersimpan untuk saham portofolio. " + _describe_sync(sync_stats)
            )
        records = [
            {"date": row.date, "ticker": stock_id_to_ticker[str(row.stock_id)],
             "price": row.close}
            for row in price_rows
            if row.close is not None
        ]
        df_price = pd.DataFrame(records).pivot_table(
            index="date", columns="ticker", values="price", aggfunc="last"
        ).sort_index()
    else:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent.parent.parent
        price_file = root / "data" / "Master_OHLCV_15Tahun.parquet"
        tickers = sorted(set(stock_id_to_ticker.values()))
        filters = [("Date", ">=", pd.Timestamp(price_query_start)), ("Ticker", "in", [f"{ticker}.JK" for ticker in tickers])]
        if valuation_end is not None:
            filters.append(("Date", "<=", pd.Timestamp(valuation_end)))
        try:
            window = pd.read_parquet(price_file, filters=filters)
        except Exception as exc:
            raise PortfolioNotFoundError(f"Gagal membaca data Parquet: {exc}") from exc
        if window.empty:
            raise PortfolioNotFoundError("Belum ada data harga pada rentang simulasi ini.")
        window["Date"] = pd.to_datetime(window["Date"])
        window["Ticker"] = window["Ticker"].astype(str).str.replace(".JK", "", regex=False)
        window["price"] = window["Close"]
        df_price = window.pivot_table(index="Date", columns="Ticker", values="price", aggfunc="last").sort_index()

    df_price.index = pd.to_datetime(df_price.index)
    if valuation_end is not None:
        df_price = df_price.loc[df_price.index.date <= valuation_end]
    if df_price.empty:
        raise PortfolioNotFoundError("Tidak ada harga sebelum akhir window item snapshot.")
    actual_prices = df_price.copy()
    # A suspended ticker may use its last mark for at most seven calendar
    # days. Older marks are not sufficient evidence for a valid daily NAV.
    df_price = actual_prices.ffill()
    for ticker in df_price.columns:
        last_seen = pd.Series(actual_prices.index, index=actual_prices.index).where(
            actual_prices[ticker].notna()
        ).ffill()
        age = actual_prices.index.to_series() - last_seen
        df_price[ticker] = df_price[ticker].where(age <= pd.Timedelta(days=7))
    df_price = df_price.dropna(how="all")
    if df_price.empty:
        raise PortfolioNotFoundError("Data harga belum cukup untuk menghitung performa.")
    # Legacy snapshots have no verified execution day. Keep their scheduled
    # date; if it is not a trading day, apply the position on the first
    # available session on/after it and report accounting as unverified.
    df_price = df_price.loc[df_price.index.date >= snapshots[0]["date"]]
    if df_price.empty:
        raise PortfolioNotFoundError("Tidak ada harga setelah awal snapshot.")
    if valuation_end is not None and df_price.index.max().date() < snapshots[-1]["date"]:
        raise PortfolioNotFoundError("Tidak ada hari valuasi untuk snapshot yang diminta sebelum end_date.")
    actual_start = df_price.index.min().date()
    if valuation_end is not None and actual_start > valuation_end:
        raise PortfolioNotFoundError("end_date lebih awal daripada harga pertama yang tersedia.")

    missing_price_tickers = sorted(
        ticker for ticker in stock_id_to_ticker.values() if ticker not in df_price.columns
    )
    valid_tickers = set(df_price.columns)

    active_snapshot_index = 0
    positions: dict[str, int] = {}
    cost_basis: dict[str, Decimal | None] = {}
    initial_capital = _money(lineage[0].budget)
    if not initial_capital.is_finite() or initial_capital <= 0:
        raise PortfolioNotFoundError("Modal awal portofolio tidak valid.")
    cash_balance: Decimal | None = initial_capital
    realized_pnl: Decimal | None = Decimal("0")
    rows: list[dict] = []
    events: list[dict] = []
    reasons: list[str] = []
    accounting_status = "verified"
    prior_nav: Decimal | None = None
    tolerance = Decimal("0.05")

    def mark_problem(message: str, legacy: bool = False) -> None:
        nonlocal accounting_status
        if message not in reasons:
            reasons.append(message)
        if accounting_status != "invalid":
            accounting_status = "legacy_unverified" if legacy else "invalid"

    for timestamp in df_price.index:
        day = timestamp.date()
        while active_snapshot_index < len(snapshots) and snapshots[active_snapshot_index]["date"] <= day:
            snapshot = snapshots[active_snapshot_index]
            next_positions = {ticker: int(data["lots"]) for ticker, data in snapshot["positions"].items()}
            invalid_window = False
            for ticker, data in snapshot["positions"].items():
                if (data.get("start_date") is not None and data["start_date"] > snapshot["date"] or
                    data.get("end_date") is not None and data["end_date"] <= snapshot["date"]):
                    mark_problem(f"Window item {ticker} pada snapshot {snapshot['id']} tidak mencakup tanggal eksekusi.")
                    invalid_window = True
            if active_snapshot_index:
                previous = snapshots[active_snapshot_index - 1]
                previous_ends = [data.get("end_date") for data in previous["positions"].values()]
                if any(value is not None for value in previous_ends) and any(
                    value != snapshot["date"] for value in previous_ends
                ):
                    mark_problem(f"end_date item snapshot {previous['id']} tidak selaras dengan rebalance {snapshot['date']}.")
                    invalid_window = True
            if invalid_window:
                cash_balance = None
                realized_pnl = None
            # budget = pre-trade NAV estimate; total_terpakai = post-trade
            # allocation at stored harga_acuan; sisa_budget = post-trade cash.
            # Do not compare either budget or allocation with today's Close
            # NAV: execution slippage and valuation dates can differ.
            reference_prices = [data.get("reference_price") for data in snapshot["positions"].values()]
            if snapshot["execution_prices"] is not None:
                if not all(price is not None and math.isfinite(price) and price > 0 for price in reference_prices):
                    mark_problem(f"Harga acuan snapshot {snapshot['id']} tidak lengkap.")
                else:
                    allocation_value = sum(
                        data["lots"] * 100 * _money(data["reference_price"])
                        for data in snapshot["positions"].values()
                    )
                    if abs(allocation_value - _money(snapshot["total_terpakai"])) > tolerance:
                        mark_problem(f"Total terpakai snapshot {snapshot['id']} tidak cocok dengan jumlah lot × harga acuan tersimpan.")
            delta = {ticker: next_positions.get(ticker, 0) - positions.get(ticker, 0)
                     for ticker in set(positions) | set(next_positions)
                     if next_positions.get(ticker, 0) != positions.get(ticker, 0)}
            cash_before = cash_balance
            before_holdings = Decimal("0")
            before_known = True
            for ticker, lots in positions.items():
                if ticker not in valid_tickers or pd.isna(df_price.at[timestamp, ticker]):
                    before_known = False
                    missing_price_tickers.append(ticker)
                else:
                    before_holdings += lots * 100 * _money(df_price.at[timestamp, ticker])
            nav_before = cash_before + before_holdings if cash_before is not None and before_known else None
            prices = snapshot["execution_prices"]
            event_realized: Decimal | None = Decimal("0")
            if prices is None:
                mark_problem(f"Snapshot {snapshot['id']} tidak menyimpan harga eksekusi; P/L historis tidak dapat dibuktikan.", legacy=True)
                if day != snapshot["date"]:
                    mark_problem(
                        f"Snapshot legacy {snapshot['id']} dijadwalkan {snapshot['date']} dan baru dapat dinilai pada {day}; tanggal transaksi sebenarnya tidak diketahui.",
                        legacy=True,
                    )
                realized_pnl = None
                cost_basis = {ticker: None for ticker in next_positions}
                cash_balance = _money(snapshot["cash"]) if snapshot["cash"] is not None else None
                if cash_balance is None or not cash_balance.is_finite() or cash_balance < -tolerance:
                    cash_balance = None
                    mark_problem(f"Kas snapshot legacy {snapshot['id']} tidak valid.")
                event_realized = None
            else:
                changed_prices: dict[str, Decimal] = {}
                for ticker in delta:
                    raw = prices.get(ticker) if isinstance(prices, dict) else None
                    try:
                        price = _money(raw)
                    except (TypeError, ValueError, ArithmeticError):
                        price = Decimal("0")
                    if raw is None or not price.is_finite() or price <= 0:
                        mark_problem(f"Harga eksekusi {ticker} pada snapshot {snapshot['id']} tidak tersedia atau tidak valid.")
                    else:
                        changed_prices[ticker] = price
                if day != snapshot["date"]:
                    mark_problem(f"Hari eksekusi snapshot {snapshot['id']} ({snapshot['date']}) tidak ada dalam data valuasi.")
                if cash_before is None or len(changed_prices) != len(delta):
                    cash_balance = None
                    realized_pnl = None
                    cost_basis = {ticker: None for ticker in next_positions}
                    event_realized = None
                else:
                    next_cash = cash_before
                    for ticker, lots_delta in delta.items():
                        price = changed_prices[ticker]
                        next_cash -= lots_delta * 100 * price
                        if lots_delta < 0:
                            basis = cost_basis.get(ticker)
                            if basis is None or realized_pnl is None:
                                realized_pnl = None
                                event_realized = None
                            else:
                                gain = -lots_delta * 100 * (price - basis)
                                realized_pnl += gain
                                event_realized += gain
                        else:
                            old_lots = positions.get(ticker, 0)
                            old_basis = cost_basis.get(ticker)
                            cost_basis[ticker] = (
                                (old_lots * old_basis + lots_delta * price) / (old_lots + lots_delta)
                                if old_lots and old_basis is not None else
                                (None if old_lots else price)
                            )
                    for ticker in set(positions) - set(next_positions):
                        cost_basis.pop(ticker, None)
                    stored_cash = _money(snapshot["cash"]) if snapshot["cash"] is not None else None
                    if (next_cash < -tolerance or stored_cash is None or
                        not stored_cash.is_finite() or abs(next_cash - stored_cash) > tolerance):
                        cash_balance = None
                        realized_pnl = None
                        cost_basis = {ticker: None for ticker in next_positions}
                        event_realized = None
                        mark_problem(f"Kas snapshot {snapshot['id']} tidak cocok dengan arus BUY/SELL dari selisih lot dan harga eksekusi.")
                    else:
                        cash_balance = next_cash
            if not is_backtest and prices is not None and realized_pnl is not None:
                # A live PATCH is a cost-basis correction, not a cash flow.
                # Apply it after this snapshot's trades, so a later edit
                # cannot rewrite realized P/L on the same rebalance event.
                for ticker, data in snapshot["positions"].items():
                    purchase, item_total, reference = (
                        data.get("purchase_price"), data.get("item_total"),
                        data.get("reference_price"),
                    )
                    if purchase is None or item_total is None or reference is None:
                        continue
                    if not all(math.isfinite(value) and value > 0 for value in (purchase, item_total, reference)):
                        continue
                    shares = data["lots"] * 100
                    edited_total = shares * _money(purchase)
                    reference_total = shares * _money(reference)
                    if (abs(_money(item_total) - edited_total) <= tolerance and
                        abs(edited_total - reference_total) > tolerance):
                        cost_basis[ticker] = _money(purchase)
            positions = next_positions
            after_nav = None
            if cash_balance is not None and all(
                ticker in valid_tickers and pd.notna(df_price.at[timestamp, ticker]) for ticker in positions
            ):
                after_nav = cash_balance + sum(
                    lots * 100 * _money(df_price.at[timestamp, ticker]) for ticker, lots in positions.items()
                )
            if active_snapshot_index > 0:
                events.append({
                    "portfolio_id": snapshot["id"], "execution_date": day.isoformat(),
                    "delta_lots": dict(sorted(delta.items())),
                    "cash_before": float(cash_before) if cash_before is not None else None,
                    "cash_after": float(cash_balance) if cash_balance is not None else None,
                    "nav_before": float(nav_before) if nav_before is not None else None,
                    "nav_after": float(after_nav) if after_nav is not None else None,
                    "realized_pnl_delta": float(event_realized) if event_realized is not None else None,
                })
            active_snapshot_index += 1

        holdings_value = Decimal("0")
        unrealized_pnl: Decimal | None = Decimal("0")
        holdings_known = True
        stale_tickers = []
        for ticker, lots in positions.items():
            if ticker not in valid_tickers or pd.isna(df_price.at[timestamp, ticker]):
                missing_price_tickers.append(ticker)
                holdings_known = False
                unrealized_pnl = None
                continue
            if ticker not in actual_prices.columns or pd.isna(actual_prices.at[timestamp, ticker]):
                stale_tickers.append(ticker)
            market_value = lots * 100 * _money(df_price.at[timestamp, ticker])
            holdings_value += market_value
            basis = cost_basis.get(ticker)
            if basis is None or unrealized_pnl is None:
                unrealized_pnl = None
            else:
                unrealized_pnl += market_value - lots * 100 * basis
        if not holdings_known:
            mark_problem(f"Harga valuasi tidak tersedia pada {day}; NAV hari itu tidak dapat dihitung.")
        nav = cash_balance + holdings_value if cash_balance is not None and holdings_known else None
        daily_return = nav / prior_nav - 1 if nav is not None and prior_nav not in (None, 0) else None
        cumulative_return = nav / initial_capital - 1 if nav is not None and initial_capital else None
        rows.append({
            "date": timestamp, "portfolio_value": nav, "cash_balance": cash_balance,
            "holdings_market_value": holdings_value if holdings_known else None,
            "daily_return": daily_return, "portfolio_return": cumulative_return,
            "realized_pnl_cumulative": realized_pnl, "unrealized_pnl": unrealized_pnl,
            "stale_price_tickers": sorted(stale_tickers),
        })
        prior_nav = nav

    if active_snapshot_index < len(snapshots):
        raise PortfolioNotFoundError("Snapshot yang diminta belum tercakup oleh rentang harga/end_date.")

    series_df = pd.DataFrame(rows).set_index("date")
    ihsg_return = None

    def benchmark_on_portfolio_days(index_closes):
        index_closes = index_closes.dropna().sort_index()
        if index_closes.empty:
            return None
        dates = series_df.index
        calendar = index_closes.index.union(dates).sort_values()
        marks = index_closes.reindex(calendar).ffill().reindex(dates)
        seen = pd.Series(index_closes.index, index=index_closes.index).reindex(calendar).ffill().reindex(dates)
        age = dates.to_series() - seen
        return (marks / index_closes.iloc[0] - 1.0).where(age <= pd.Timedelta(days=7))

    if not is_backtest:
        query = db.query(IdxComposite).filter(IdxComposite.date >= actual_start)
        if valuation_end is not None:
            query = query.filter(IdxComposite.date <= valuation_end)
        idx_rows = query.order_by(IdxComposite.date.asc()).all()
        if idx_rows:
            idx = pd.Series({pd.Timestamp(row.date): row.close for row in idx_rows if row.close is not None}).sort_index()
            ihsg_return = benchmark_on_portfolio_days(idx)
    else:
        try:
            from pathlib import Path
            context_file = Path(__file__).resolve().parent.parent.parent.parent / "data" / "Master_Market_Context_15Tahun.parquet"
            filters = [("Ticker", "==", "^JKSE"), ("Date", ">=", pd.Timestamp(actual_start))]
            if valuation_end is not None:
                filters.append(("Date", "<=", pd.Timestamp(valuation_end)))
            idx_data = pd.read_parquet(context_file, filters=filters)
            if not idx_data.empty:
                idx = idx_data.set_index(pd.to_datetime(idx_data["Date"]))["Close"].dropna().sort_index()
                ihsg_return = benchmark_on_portfolio_days(idx)
        except Exception as exc:  # benchmark failure must not hide NAV
            logger.warning("Gagal membaca IHSG dari Parquet: %s", exc)

    series = []
    for timestamp, row in series_df.iterrows():
        ihsg = ihsg_return.loc[timestamp] if ihsg_return is not None and timestamp in ihsg_return.index else None
        series.append({
            "date": timestamp.isoformat(),
            "portfolio_value": float(row["portfolio_value"]) if pd.notna(row["portfolio_value"]) else None,
            "cash_balance": float(row["cash_balance"]) if pd.notna(row["cash_balance"]) else None,
            "holdings_market_value": float(row["holdings_market_value"]) if pd.notna(row["holdings_market_value"]) else None,
            "daily_return": float(row["daily_return"]) if pd.notna(row["daily_return"]) else None,
            "portfolio_return": float(row["portfolio_return"]) if pd.notna(row["portfolio_return"]) else None,
            "realized_pnl_cumulative": float(row["realized_pnl_cumulative"]) if pd.notna(row["realized_pnl_cumulative"]) else None,
            "unrealized_pnl": float(row["unrealized_pnl"]) if pd.notna(row["unrealized_pnl"]) else None,
            "stale_price_tickers": row["stale_price_tickers"],
            "ihsg_return": float(ihsg) if ihsg is not None and pd.notna(ihsg) else None,
        })

    final_state = positions
    final_row = series[-1] if series else None
    return {
        "start_date": actual_start.isoformat(),
        "requested_start_date": requested_start.isoformat(),
        "start_date_adjusted": actual_start != requested_start,
        "end_date": df_price.index.max().date().isoformat(),
        "holdings": {ticker: lots for ticker, lots in sorted(final_state.items()) if lots > 0},
        "missing_price_tickers": sorted(set(missing_price_tickers)),
        "root_portfolio_id": str(lineage[0].id),
        "evaluated_portfolio_id": str(portfolio.id),
        "rebalance_events": events,
        "accounting_status": accounting_status,
        "accounting_reasons": reasons,
        "initial_capital": float(initial_capital),
        "final_nav": final_row["portfolio_value"] if final_row else None,
        # Portfolio_return remains a decimal for chart compatibility; this
        # summary field is an actual percentage as its name states.
        "total_return_pct": (
            final_row["portfolio_return"] * 100.0
            if final_row and final_row["portfolio_return"] is not None
            else None
        ),
        "realized_pnl": final_row["realized_pnl_cumulative"] if final_row else None,
        "unrealized_pnl": final_row["unrealized_pnl"] if final_row else None,
        "dividend_income": None,
        "transaction_costs": 0.0 if accounting_status == "verified" else None,
        "accounting_limitations": [
            "Simulasi baru mengasumsikan biaya transaksi nol; pajak, dividen, setoran/penarikan eksternal dan corporate action belum dimodelkan.",
            "Valuasi memakai raw Close; cost basis backtest dihitung weighted-average dari harga eksekusi, sedangkan harga_beli yang diubah pengguna menjadi koreksi basis live tanpa mengubah kas.",
            "Backtest mengambil keputusan dari data sampai sesi sebelumnya lalu mensimulasikan eksekusi pada Close sesi efektif; event dengan kas tidak cukup ditolak.",
            "Harga valuasi terakhir dapat diteruskan maksimum tujuh hari kalender; setelah itu NAV tidak dilaporkan tanpa harga baru.",
        ],
        "series": series,
    }
