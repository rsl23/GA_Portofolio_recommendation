import calendar
import logging
import os
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
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
from src.backend.services.openai_llm_caller import explain_portfolio
from src.gaengine.narrative_style import NARRATIVE_VERSION
from src.backend.services.portfolio_dividends import (
    DividendDataUnavailableError, dividend_entitlements, load_dividend_history,
)
from src.backend.services.portfolio_return_metrics import calculate_return_metrics
from src.gaengine.data_loader import build_market_data
from src.gaengine.engine import GeneticEngine
from src.gaengine.progress import check_cancelled, report_progress, set_progress_context
from src.gaengine.ga_config import GAConfig, REBALANCE_FITNESS_PARAMETERS
from src.gaengine.rebalancer import (
    compute_nav,
    align_old_weights_to_market,
    apply_no_trade_band,
    fit_buys_to_available_cash,
    compute_rebalance_deltas,
)
from src.preprocessing.bear_market_detection import assess_market
from src.preprocessing.capital_allocation_policy import (
    CONFIG as ALLOCATION_CONFIG,
    effective_fraction, recommend_fraction, resolve_bear_choice,
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Status portofolio
# ----------------------------------------------------------------------
# Setiap user bisa memiliki DUA jenis portofolio terpisah:
#   LIVE     : "active"             -> sedang dipakai; digantikan -> "replaced"
#   BACKTEST : "active_backtest"    -> simulasi terbaru; digantikan -> "replaced_backtest"
# Keduanya TIDAK saling menimpa. Generate ulang hanya mengubah STATUS;
# penghapusan eksplisit oleh pemilik tersedia melalui endpoint terpisah.
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


class PortfolioLineageConflictError(Exception):
    """Rantai rebalance tidak aman untuk dihapus sebagai satu milik user."""


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


def _execution_close_prices(db: Session, tickers: set[str], requested: date, backtest: bool,
                            require_volume: bool = False) -> tuple[date, dict[str, float]]:
    """Return raw closes on one market session; require traded volume for fills."""
    import pandas as pd
    if not tickers:
        raise MarketDataUnavailableError("Tidak ada ticker untuk dieksekusi.")
    if backtest:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent.parent.parent
        try:
            frame = pd.read_parquet(
                root / "data" / "Master_OHLCV_15Tahun.parquet",
                columns=["Date", "Ticker", "Close", "Volume"],
                filters=[("Date", ">=", pd.Timestamp(requested - timedelta(days=14))),
                         ("Date", "<=", pd.Timestamp(requested)),
                         ("Ticker", "in", [f"{ticker}.JK" for ticker in tickers])],
            )
        except Exception as exc:
            raise MarketDataUnavailableError(f"Harga Close backtest tidak dapat dibaca: {exc}") from exc
        if frame.empty:
            raise MarketDataUnavailableError(f"Tidak ada harga Close hingga {requested}.")
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
        effective = _backtest_trading_day(requested)
        day_rows = frame[frame["Date"] == effective]
        prices = {str(row.Ticker).removesuffix(".JK"): float(row.Close)
                  for row in day_rows.itertuples()
                  if str(row.Ticker).removesuffix(".JK") in tickers and pd.notna(row.Close)}
        if require_volume:
            traded = {str(row.Ticker).removesuffix(".JK") for row in day_rows.itertuples()
                      if str(row.Ticker).removesuffix(".JK") in tickers
                      and pd.notna(row.Volume) and row.Volume > 0}
            not_traded = sorted(tickers - traded)
            if not_traded:
                raise MarketDataUnavailableError(
                    f"Tidak ada volume perdagangan aktual pada {effective} untuk: "
                    f"{', '.join(not_traded)}. Transaksi backtest dibatalkan."
                )
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


def _backtest_untraded_tickers(tickers: set[str], execution_day: date) -> set[str]:
    """Stocks with no actual trade on the execution session (Close is only a mark)."""
    if not tickers:
        return set()
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent.parent
    try:
        frame = pd.read_parquet(
            root / "data" / "Master_OHLCV_15Tahun.parquet",
            columns=["Date", "Ticker", "Volume"],
            filters=[("Date", "=", pd.Timestamp(execution_day)),
                     ("Ticker", "in", [f"{ticker}.JK" for ticker in tickers])],
        )
    except Exception as exc:
        raise MarketDataUnavailableError(f"Volume eksekusi backtest tidak dapat dibaca: {exc}") from exc
    traded = {
        str(row.Ticker).removesuffix(".JK") for row in frame.itertuples()
        if pd.notna(row.Volume) and row.Volume > 0
    }
    return tickers - traded


def _retain_untraded_lots(old_lots: dict[str, int], desired: dict[str, int],
                           untraded: set[str], priorities: dict[str, float],
                           max_stocks: int, min_stocks: int = 3) -> dict[str, int]:
    """Cancel unfillable orders and keep suspended holdings across snapshots."""
    result = {code: qty for code, qty in desired.items() if qty > 0 and code not in untraded}
    frozen = {code: qty for code, qty in old_lots.items() if qty > 0 and code in untraded}
    result.update(frozen)
    # Prefer carrying a tradable old position to inventing a new BUY when a
    # zero-volume candidate leaves fewer than the required number of stocks.
    for code in sorted(set(old_lots) - set(result) - untraded,
                       key=lambda ticker: (-priorities.get(ticker, 0.0), ticker)):
        if len(result) >= min_stocks:
            break
        result[code] = old_lots[code]
    while len(result) > max(max_stocks, len(frozen)):
        removable = set(result) - set(frozen)
        if not removable:
            break
        code = min(removable, key=lambda ticker: (priorities.get(ticker, 0.0), ticker))
        del result[code]
    return result


def _deferred_untraded_signals(old_lots: dict[str, int], ga_target: dict[str, int],
                                untraded: set[str]) -> list[dict]:
    """Keep GA intentions visible without booking impossible transactions."""
    signals = []
    for code in sorted(untraded):
        held, target = old_lots.get(code, 0), ga_target.get(code, 0)
        if held == target:
            continue
        signals.append({
            "ticker": code,
            "action": "BUY" if target > held else "SELL",
            "held_lots": held,
            "target_lots": target,
            "reason": "Belum ada bukti volume perdagangan positif pada sesi acuan; sinyal GA belum dilaksanakan.",
        })
    return signals


def _marked_holdings_value(lots: dict[str, int], prices: dict[str, float]) -> Decimal:
    """Value every retained position, including holdings outside the GA universe."""
    return sum(
        (_money(qty) * 100 * _money(prices[code]) for code, qty in lots.items()),
        Decimal("0"),
    )


def _live_execution_prices(db: Session, ga_data, old_lots: dict[str, int]) -> tuple[date, dict[str, float]]:
    """Use the GA's raw, completed-session Close for live candidates.

    A pre-existing holding outside the current universe still needs an actual
    Close on that same date; a stale DB mark is never treated as a trade.
    """
    execution_day = getattr(ga_data, "price_date", None)
    if execution_day is None:
        raise MarketDataUnavailableError("Tanggal harga Close live dari GA tidak tersedia.")
    execution_day = _as_date(execution_day)
    today_wib = datetime.now(timezone(timedelta(hours=7))).date()
    if execution_day > today_wib or (today_wib - execution_day).days > 7:
        raise MarketDataUnavailableError(
            f"Harga Close live per {execution_day} terlalu lama atau berada di masa depan."
        )
    if len(ga_data.stock_codes) != len(ga_data.prices_per_lot):
        raise MarketDataUnavailableError("Jumlah harga kandidat GA tidak cocok dengan jumlah ticker.")
    prices = {
        code: float(price) / 100.0
        for code, price in zip(ga_data.stock_codes, ga_data.prices_per_lot)
    }
    if any(not math.isfinite(price) or price <= 0 for price in prices.values()):
        raise MarketDataUnavailableError("Harga Close live kandidat GA tidak valid.")
    outside = set(old_lots) - set(prices)
    if outside:
        # A suspended holding still has value. Its last observed Close is a
        # valuation mark only; _live_untraded_tickers prevents a fictional sale.
        rows = db.query(StockUniverse.ticker, MarketData.date, MarketData.close).join(
            MarketData, MarketData.stock_id == StockUniverse.id_stock,
        ).filter(StockUniverse.ticker.in_(outside), MarketData.date <= execution_day).order_by(
            MarketData.date.desc()).all()
        for code, _, close in rows:
            if code not in prices and close is not None and math.isfinite(close) and close > 0:
                prices[code] = float(close)
        missing = outside - set(prices)
        if missing:
            raise MarketDataUnavailableError(f"Harga penilaian posisi lama belum tersedia: {', '.join(sorted(missing))}.")
    return execution_day, prices


def _live_untraded_tickers(db, ga_data, old_lots, execution_day):
    """Use actual session volume; missing evidence outside the screen means HOLD."""
    untraded = set(getattr(ga_data, "untraded_tickers", None) or [])
    outside = set(old_lots) - set(ga_data.stock_codes)
    if outside:
        rows = db.query(StockUniverse.ticker, MarketData.volume).join(
            MarketData, MarketData.stock_id == StockUniverse.id_stock,
        ).filter(StockUniverse.ticker.in_(outside), MarketData.date == execution_day).all()
        traded = {code for code, volume in rows if volume is not None and volume > 0}
        untraded.update(outside - traded)
    return untraded


def _backtest_trading_day(requested: date) -> date:
    """Resolve an actual market session, skipping synthetic zero-volume holidays."""
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent.parent
    try:
        days = pd.read_parquet(
            root / "data" / "Master_OHLCV_15Tahun.parquet",
            columns=["Date", "Volume"],
            filters=[("Date", ">=", pd.Timestamp(requested - timedelta(days=14))),
                     ("Date", "<=", pd.Timestamp(requested))],
        )
    except Exception as exc:
        raise MarketDataUnavailableError(f"Kalender perdagangan backtest tidak dapat dibaca: {exc}") from exc
    days = days[pd.to_numeric(days["Volume"], errors="coerce") > 0]
    if days.empty:
        raise MarketDataUnavailableError(f"Tidak ada hari bursa hingga {requested}.")
    return pd.to_datetime(days["Date"]).dt.date.max()


def preview_bear_market(risk_profile: str, backtest: bool = False,
                        date_ref: date | None = None, live_as_of: date | None = None) -> dict:
    """Read-only, as-of regime and allocation preview for one execution date."""
    if backtest:
        if date_ref is None:
            raise ValueError("date_ref wajib untuk pratinjau backtest.")
        execution_day = _backtest_trading_day(date_ref)
        decision_day = _backtest_trading_day(execution_day - timedelta(days=1))
    else:
        execution_day = None
        now_wib = datetime.now(timezone(timedelta(hours=7)))
        decision_day = now_wib.date() if now_wib.hour >= 17 else now_wib.date() - timedelta(days=1)
        if live_as_of is not None:
            if live_as_of > decision_day or (decision_day - live_as_of).days > 7:
                raise ValueError("Pratinjau bear market sudah berubah; muat ulang sebelum generate.")
            decision_day = live_as_of
    assessment = assess_market(decision_day, backtest)
    if not backtest and assessment["as_of"] is not None:
        decision_day = _as_date(assessment["as_of"])
        completed = now_wib.date() if now_wib.hour >= 17 else now_wib.date() - timedelta(days=1)
        if (completed - decision_day).days > 7:
            raise MarketDataUnavailableError(
                f"Data IHSG terakhir {decision_day} terlalu lama untuk sesi bersama live; perbarui sumber pasar."
            )
    result = {
        **assessment,
        "execution_date": execution_day.isoformat() if execution_day else None,
        "decision_date": decision_day.isoformat(),
        "recommended_fraction": recommend_fraction(
            assessment["status"], assessment["severity"], risk_profile
        ),
        "risk_profile": risk_profile,
        "allocation_config_version": ALLOCATION_CONFIG.version,
        "allocation_formula": "base - severity * (base - minimum); normal = 1.0; unknown = null",
    }
    fingerprint_payload = {key: result[key] for key in (
        "status", "severity", "as_of", "decision_date", "execution_date",
        "config_version", "components", "effective_weights", "recommended_fraction",
        "risk_profile",
        "allocation_config_version",
    )}
    result["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def preview_bear_series(start_date: date, end_date: date,
                        interval_months: int, risk_profile: str) -> dict:
    # The first date belongs to the already stored base snapshot.
    schedule = build_rebalance_schedule(start_date, end_date, interval_months)[1:]
    if not schedule:
        raise ValueError("Tanggal akhir harus mencakup minimal satu jadwal rebalance.")
    if len(schedule) > MAX_SERIES_STEPS:
        raise ValueError(f"Jumlah langkah melebihi {MAX_SERIES_STEPS}.")
    return {"scheduled_dates": [day.isoformat() for day in schedule],
            "assessments": [
                {"scheduled_date": day.isoformat(),
                 **preview_bear_market(risk_profile, True, day)}
                for day in schedule
            ]}


def _limit_lots_to_exposure(old_lots: dict[str, int], desired: dict[str, int],
                            cash_before: float, prices: dict[str, float],
                            fraction: float, min_stocks: int,
                            priority: dict[str, float], buy_fee_rate: float = 0.0,
                            sell_fee_rate: float = 0.0,
                            frozen: set[str] | None = None) -> dict[str, int]:
    """Cap all holdings against post-fee execution NAV, including old positions.

    Cancelling a BUY saves its fee; reducing a HOLD creates a SELL fee. Reprice
    these deltas after every trim rather than treating the pre-trade NAV as a
    fixed denominator. Prices are per share and all adjustments are whole lots.
    """
    fraction, buy_fee, sell_fee = map(_money, (fraction, buy_fee_rate, sell_fee_rate))
    if (not fraction.is_finite() or not 0 <= fraction <= 1 or
            any(not fee.is_finite() or not 0 <= fee < 1 for fee in (buy_fee, sell_fee))):
        raise ValueError("Porsi modal atau tarif biaya transaksi tidak valid.")
    if not _money(cash_before).is_finite() or cash_before < 0:
        raise ValueError("Kas sebelum transaksi tidak valid.")
    for positions in (old_lots, desired):
        if any(not isinstance(qty, (int, np.integer)) or qty < 0 for qty in positions.values()):
            raise ValueError("Lot harus integer non-negatif.")
    for code in set(old_lots) | set(desired):
        price = prices.get(code)
        if price is None or not _money(price).is_finite() or _money(price) <= 0:
            raise ValueError(f"Harga eksekusi tidak valid untuk {code}.")
    nav = _money(cash_before) + sum(
        _money(qty) * 100 * _money(prices[code]) for code, qty in old_lots.items()
    )
    result = {code: qty for code, qty in desired.items() if qty > 0}
    def holdings_value() -> Decimal:
        return sum((_money(qty) * 100 * _money(prices[code])
                    for code, qty in result.items()), Decimal("0"))
    def post_fee_nav() -> Decimal:
        fees = Decimal("0")
        for code in set(old_lots) | set(result):
            delta = result.get(code, 0) - old_lots.get(code, 0)
            fees += abs(delta) * 100 * _money(prices[code]) * (buy_fee if delta > 0 else sell_fee)
        return nav - fees
    while holdings_value() > fraction * post_fee_nav() + Decimal("0.01"):
        candidates = [
            code for code, qty in result.items()
            if code not in (frozen or set()) and
            (qty > 1 or sum(x > 0 for x in result.values()) > min_stocks)
        ]
        if not candidates:
            frozen_value = sum(
                _money(result.get(code, 0)) * 100 * _money(prices[code])
                for code in (frozen or set())
            )
            if frozen and frozen_value > fraction * post_fee_nav() + Decimal("0.01"):
                return result
            raise ValueError(
                "Batas porsi saham tidak dapat dipenuhi karena posisi tidak diperdagangkan atau batas minimum saham."
            )
        code = min(candidates, key=lambda ticker: (priority.get(ticker, 0.0), ticker))
        excess = holdings_value() - fraction * post_fee_nav()
        lot_value = _money(prices[code]) * 100
        pending_buy = max(0, result[code] - old_lots.get(code, 0))
        # Stop at HOLD before switching to SELL: the marginal fee changes sign.
        reduction_per_lot = lot_value * (1 + fraction * buy_fee if pending_buy
                                         else 1 - fraction * sell_fee)
        required_sale = int((excess / reduction_per_lot).to_integral_value(rounding="ROUND_CEILING"))
        max_sale = result[code] if sum(x > 0 for x in result.values()) > min_stocks else result[code] - 1
        if pending_buy:
            max_sale = min(max_sale, pending_buy)
        result[code] -= min(required_sale, max_sale)
        if result[code] == 0:
            del result[code]
    if post_fee_nav() <= 0 or post_fee_nav() - holdings_value() < Decimal("-0.01"):
        raise ValueError("NAV atau kas setelah pembatasan exposure tidak valid.")
    return result


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
    guarded = (getattr(request, "rebalance_model", "legacy") == "guarded_v1"
               and getattr(request, "rebalance", False))
    hybrid_fitness = getattr(request, "fitness_policy", "legacy") == "legacy_cvar_v1"
    experimental_fitness = hybrid_fitness

    # 0. Validasi user dari payload JWT (sub). Lempar error jika sudah terhapus.
    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise UserNotFoundError(f"User ID pada token tidak valid: {user_id}") from e
    user = db.query(User).filter(User.id == user_uuid).first()
    if user is None:
        raise UserNotFoundError("User pada token tidak ditemukan di database.")
    exposure_limit = None
    if getattr(request, "bear_management", False):
        report_progress("market", "Memeriksa kondisi pasar",
                        "Menyesuaikan analisis dengan kondisi pasar dan pilihan alokasi modal Anda.")
        preview = preview_bear_market(request.risk_profile, backtest, date_ref)
        if (preview["as_of"] is not None and
            str(request.bear_preview_as_of) != preview["as_of"]):
            raise ValueError("Pratinjau bear market sudah berubah; muat ulang sebelum generate.")
        if request.bear_preview_fingerprint != preview["fingerprint"]:
            raise ValueError("Indikator pasar berubah sejak pratinjau; muat ulang sebelum generate.")
        choice = resolve_bear_choice(user.bear_allocation_policy, preview["status"], request.bear_choice)
        exposure_limit = effective_fraction(preview, request.risk_profile, choice)

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
            profile_options = ({"include_profile_features": True}
                               if getattr(request, "profile_policy", "legacy") != "legacy" else {})
            market_data = build_market_data(
                min_price=1.0, backtest=True, date_ref=decision_day,
                **profile_options
            )
        except Exception as e:
            raise MarketDataUnavailableError(
                f"Gagal merakit MarketData backtest untuk {decision_day}: {e}"
            ) from e
    elif market_data is None or market_data.n_stocks == 0:
        logger.info("Membentuk MarketData menggunakan data LIVE...")
        try:
            profile_choice = getattr(request, "profile_policy", "legacy")
            profile_options = (
                {"include_profile_features": True,
                 "profile_benchmark_live": profile_choice == "composition_v2"}
                if profile_choice != "legacy" else {})
            market_data = build_market_data(min_price=1.0, **profile_options)
        except ValueError as exc:
            raise MarketDataUnavailableError(f"Gagal merakit MarketData live: {exc}") from exc

    if market_data is None or market_data.n_stocks == 0:
        raise MarketDataUnavailableError(
            "Gagal membentuk MarketData: histori OHLCV/fundamental tidak tersedia, "
            "atau tidak ada kandidat dengan harga Close valid pada sesi penilaian. "
            "filtered_stock_cache yang terisi saja belum cukup; periksa juga sumber harga."
        )
    # Bear indicators have their own observed date. Older causal indicators
    # must not rewind a newer stock-price/return snapshot to the IHSG date.
    if not backtest and getattr(request, "bear_management", False) and preview["as_of"] is not None:
        if _as_date(preview["as_of"]) > _as_date(market_data.price_date):
            raise MarketDataUnavailableError(
                "Harga saham tertinggal dari indikator pasar; perbarui OHLCV sebelum generate."
            )

    if getattr(request, "profile_policy", "legacy") != "legacy" and market_data.profile_features is None:
        # Startup's live cache is deliberately legacy. Reload a local object;
        # never mutate the shared cache while another request is using it.
        if not backtest:
            expected_session = market_data.price_date
            market_data = build_market_data(min_price=1.0, include_profile_features=True,
                                           live_as_of=expected_session,
                                           profile_benchmark_live=(getattr(request, "profile_policy", "legacy") == "composition_v2"))
            if market_data is not None and market_data.price_date != expected_session:
                raise MarketDataUnavailableError(
                    f"Histori komposisi live untuk sesi {expected_session} belum lengkap; perbarui sumber harga."
                )
        if market_data is None or market_data.profile_features is None:
            raise ValueError("Bukti komposisi profil tidak tersedia.")

    report_progress("analyzing", "Menganalisis saham pilihan",
                    f"Menyiapkan {market_data.n_stocks} kandidat saham beserta risiko dan hubungan antar saham.",
                    candidates=market_data.n_stocks)
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
    requested_contribution = getattr(request, "contribution_amount", 0)
    if not math.isfinite(requested_contribution) or requested_contribution < 0:
        raise ValueError("Setoran modal harus berupa angka tidak negatif dan terbatas.")
    if requested_contribution and (not backtest or not request.rebalance):
        raise ValueError("Setoran langsung hanya berlaku pada rebalance backtest; gunakan endpoint tambah modal untuk live.")
    applied_contribution = Decimal("0")
    name_portofolio = (getattr(request, "name_portofolio", None) or "").strip()
    description = (getattr(request, "description", None) or "").strip() or None
    
    live_formation_day = datetime.now(timezone(timedelta(hours=7))).date() if not backtest else None
    if getattr(request, 'rebalance', False):
        try:
            old_snapshot_row = _resolve_action_portfolio(db, user_uuid,
                getattr(request, 'rebalance_portfolio_id', None), backtest=backtest)
            old_portfolio_id = str(old_snapshot_row.id)
            old_portfolio_res = _to_portfolio_response(db, old_snapshot_row)
            # Snapshot hasil rebalance tetap memakai identitas portofolio lama.
            name_portofolio = name_portofolio or getattr(old_snapshot_row, "name_portofolio", None) or ""
            description = description if description is not None else getattr(old_snapshot_row, "description", None)
            previous_default_name = (
                f"Portofolio {old_snapshot_row.risk_profile} {_snapshot_date(old_snapshot_row).isoformat()}"
            )
            if not name_portofolio or name_portofolio == previous_default_name:
                root_snapshot = _portfolio_lineage(db, old_snapshot_row, user_uuid)[0]
                name_portofolio = (
                    getattr(root_snapshot, "name_portofolio", None) or
                    f"Portofolio {root_snapshot.risk_profile} {_snapshot_date(root_snapshot).isoformat()}"
                )
            applied_contribution = _money(requested_contribution) + _money(old_snapshot_row.pending_deposit or 0)
            if not applied_contribution.is_finite() or applied_contribution < 0:
                raise ValueError("Total setoran untuk rebalance tidak valid.")
            for alloc in old_portfolio_res.allocations:
                if alloc.lots > 0:
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
            leftover_cash = float(_money(old_portfolio_res.sisa_budget) + applied_contribution)
            if backtest:
                execution_day, all_close_prices = _execution_close_prices(
                    db, set(old_lots) | set(market_data.stock_codes), execution_target, True,
                )
            else:
                execution_day, all_close_prices = _live_execution_prices(db, market_data, old_lots)
            if backtest and execution_day != execution_target:
                raise MarketDataUnavailableError("Hari harga eksekusi berbeda dari jadwal bursa efektif.")
            if old_snapshot_row is not None:
                old_cost_basis = _rebalance_cost_basis(
                    db, old_snapshot_row, user_uuid, allow_legacy_reference=True,
                )
            previous_execution_day = (
                old_snapshot_row.execution_date if old_snapshot_row is not None else None
            ) or old_portfolio_res.date_ref or old_portfolio_res.created_at
            previous_execution_day = _as_date(previous_execution_day)
            effective_day = execution_day if backtest else live_formation_day
            if effective_day <= previous_execution_day:
                raise MarketDataUnavailableError(
                    f"Hari pembentukan {effective_day} harus sesudah snapshot sebelumnya ({previous_execution_day})."
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
        if backtest:
            execution_day, all_close_prices = _execution_close_prices(
                db, set(market_data.stock_codes), execution_target, True,
            )
        else:
            execution_day, all_close_prices = _live_execution_prices(db, market_data, {})
        if backtest and execution_day != execution_target:
            raise MarketDataUnavailableError("Hari harga eksekusi berbeda dari jadwal bursa efektif.")
    if not name_portofolio:
        name_portofolio = f"Portofolio {request.risk_profile} {(execution_day if backtest else live_formation_day).isoformat()}"
    live_untraded = (_live_untraded_tickers(db, market_data, old_lots, execution_day)
                     if not backtest else set())
    experimental_profile_v2 = getattr(request, "profile_policy", "legacy") == "composition_v2"
    if experimental_profile_v2 and guarded:
        raise ValueError("composition_v2 belum mendukung guarded_v1; pilih rebalance legacy.")
    if guarded or (experimental_profile_v2 and old_lots):
        from src.gaengine.held_universe import load_held_universe
        from src.gaengine.data_loader import DATA_DIR
        sector_map = None
        if market_data.profile_features is not None:
            sector_map = {ticker: sector for ticker, sector in db.query(
                StockUniverse.ticker, StockUniverse.sektor).all()}
        if backtest:
            market_data = load_held_universe(
                market_data, old_lots, DATA_DIR, sector_map,
                allow_untraded_signal=experimental_profile_v2,
            )
        else:
            from src.gaengine.live_evidence import load_live_held_universe
            market_data = load_live_held_universe(market_data, old_lots, sector_map,
                                                 frozen_tickers=live_untraded)
        if hybrid_fitness and market_data.market_stress is None:
            # Extending the holdings universe invalidates the old stock-row
            # matrix. Rebuild it so hybrid retains legacy market-stress risk.
            from src.gaengine.market_stress import load_market_stress
            market_data.market_stress = load_market_stress(
                market_data.stock_codes, market_data.price_date, DATA_DIR)
        old_weights_aligned = [old_lots.get(c, 0)*p/ga_budget for c,p in
                               zip(market_data.stock_codes,market_data.prices_per_lot)]
        # A failed BUY screen no longer makes an old holding a forced sell.
        eliminated_stocks = []
    if experimental_fitness:
        horizon = getattr(request, "fitness_horizon_days", 63)
        history = np.asarray(market_data.returns)
        if history.shape[1] < 3 * horizon or not np.isfinite(history).all() or np.any(history <= -1):
            raise MarketDataUnavailableError(
                f"{request.fitness_policy} memerlukan minimal {3*horizon} sesi return valid "
                f"({history.shape[1]} tersedia). Gunakan fitness legacy pada kedua mode "
                "atau lengkapi histori; CVaR tidak diganti dengan angka nol."
            )
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
        profile_policy=("legacy" if experimental_profile_v2 else getattr(request, "profile_policy", "legacy")),
        max_stocks=10,
        risk_free_rate=market_data.risk_free_rate,
        # seed=42,
        data=market_data,
        fitness_policy=getattr(request, "fitness_policy", "legacy"),
        forecast_horizon_days=getattr(request, "fitness_horizon_days", 63),
        cvar_tail_fraction=getattr(request, "fitness_cvar_tail_fraction", .1),
        lambda_cvar=getattr(request, "fitness_lambda_cvar", .5),
        is_rebalance=getattr(request, 'rebalance', False),
        rebalance_model="guarded_v1" if guarded else "legacy",
        buy_fee_rate=getattr(request, "rebalance_buy_fee_rate", .0015) if (guarded or experimental_fitness or getattr(request, "charge_transaction_fees", False)) else 0.,
        sell_fee_rate=getattr(request, "rebalance_sell_fee_rate", .0025) if (guarded or experimental_fitness or getattr(request, "charge_transaction_fees", False)) else 0.,
        old_weights=old_weights_aligned,
        old_lots=old_lots,
        old_prices_per_lot=old_signal_prices_per_lot,
        old_cost_basis=old_cost_basis,
        old_cash=leftover_cash if old_portfolio_id else None,
        allow_cash=exposure_limit is not None and exposure_limit < 1.0,
        min_exposure=exposure_limit if exposure_limit is not None and exposure_limit < 1.0 else 0.4,
        max_exposure=exposure_limit if exposure_limit is not None and exposure_limit < 1.0 else 1.0,
        **(REBALANCE_FITNESS_PARAMETERS if getattr(request, 'rebalance', False) else {}),
    )

    if config.is_rebalance:
        logger.info(
            "Bobot fitness rebalance: turnover=%.2f, momentum=%.2f, utility=%.2f, return-NAV=%.2f (cash aktif=%s)",
            config.turnover_penalty_beta, config.momentum_bonus, config.rebalance_utility_weight,
            config.cash_return_weight, config.allow_cash,
        )

    # 3. Evolusi genetika
    report_progress("optimizing", "Merangkai kombinasi saham",
                    "Menyiapkan kombinasi awal untuk dibandingkan oleh algoritma genetika.",
                    total_generations=config.generations)
    logger.info("=== MEMULAI EVOLUSI GENETIKA ===")
    rebalance_diagnostics = None
    if experimental_profile_v2:
        from types import SimpleNamespace
        from src.gaengine.composition_v2_experiment import run as run_profile_v2

        solution, history = run_profile_v2(config)
        engine = SimpleNamespace(history=history)
    elif guarded:
        from src.gaengine.guarded_rebalance import GuardedRebalance
        engine = GuardedRebalance(config)
        solution, rebalance_diagnostics = engine.run()
    else:
        engine = GeneticEngine(config)
        # Generation information already goes through the non-blocking logger
        # and progress stream; avoid synchronous print() to the VS Code PTY.
        solution = engine.run(verbose=False)
    if not solution.diversification_ok or not solution.budget_ok:
        raise ValueError("GA tidak menghasilkan portofolio yang memenuhi batas saham dan budget.")
    
    report_progress("allocating", "Memeriksa komposisi terpilih",
                    "Menghitung jumlah lot, sisa kas, dan biaya sesuai alokasi Anda.",
                    generation=len(engine.history), total_generations=config.generations,
                    fitness=float(solution.fitness))
    # --- APPLY NO TRADE BAND ---
    if getattr(request, 'rebalance', False) and not guarded and not experimental_profile_v2 and not (
        exposure_limit is not None and exposure_limit < 1.0
    ):
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
    ga_signal_lots = dict(desired_lots)
    untraded = (_backtest_untraded_tickers(
        set(old_lots) | set(market_data.stock_codes), execution_day)
        if backtest else live_untraded)
    frozen = set(old_lots) & untraded
    if untraded:
        desired_lots = _retain_untraded_lots(
            old_lots, desired_lots, untraded,
            dict(zip(market_data.stock_codes, solution.genes)),
            config.max_stocks, config.min_stocks,
        )
        logger.info("Rebalance %s: posisi dibekukan %s; kandidat tanpa transaksi %s",
                    execution_day, sorted(frozen), sorted(untraded - set(old_lots)))
    # The live candidate fill follows MarketData/GA prices, while a full sell
    # outside that universe follows its observed Close. Use exactly those same
    # prices for affordability, exposure, and the persisted cash delta.
    execution_price_map = {**all_close_prices,
                           **{code: float(price) for code, price in ga_prices.items()}}
    affordable_lots, _ = fit_buys_to_available_cash(
        old_lots, desired_lots,
        leftover_cash if old_portfolio_id else float(request.budget),
        execution_price_map,
        dict(zip(market_data.stock_codes, solution.genes)), config.min_stocks,
        config.buy_fee_rate, config.sell_fee_rate,
    )
    if exposure_limit is not None and exposure_limit < 1.0:
        affordable_lots = _limit_lots_to_exposure(
            old_lots, affordable_lots,
            leftover_cash if old_portfolio_id else float(request.budget),
            execution_price_map, exposure_limit,
            len(frozen) if frozen else config.min_stocks,
            dict(zip(market_data.stock_codes, solution.genes)),
            config.buy_fee_rate, config.sell_fee_rate,
            frozen=frozen,
        )
    if experimental_profile_v2:
        from src.gaengine.composition_v2_experiment import trim_execution_lots

        execution_lot_prices = np.asarray(
            [execution_price_map[c] * 100 for c in market_data.stock_codes], dtype=float)
        trimmed = trim_execution_lots(
            config,
            [affordable_lots.get(c, 0) for c in market_data.stock_codes],
            execution_lot_prices,
            min_lots=[old_lots.get(c, 0) if c in frozen else 0
                      for c in market_data.stock_codes],
        )
        reductions = {
            c: affordable_lots.get(c, 0) - int(q)
            for c, q in zip(market_data.stock_codes, trimmed)
            if affordable_lots.get(c, 0) > int(q)
        }
        if reductions:
            rebalance_diagnostics = rebalance_diagnostics or {}
            rebalance_diagnostics["composition_v2_execution_reductions"] = reductions
        affordable_lots = {c: int(q) for c, q in zip(market_data.stock_codes, trimmed) if q > 0}
    if any(affordable_lots.get(code, 0) != old_lots[code] for code in frozen):
        raise ValueError("Lot saham tanpa perdagangan berubah saat penyesuaian eksekusi.")
    if affordable_lots != desired_lots:
        logger.info("Menyesuaikan lot ke kas dan batas exposure pada harga eksekusi %s", execution_day)
    from src.backend.services.portfolio_signal_snapshot import score_execution_lots
    solution = score_execution_lots(config, solution.genes, solution.exposure, affordable_lots)
    if experimental_profile_v2:
        from src.gaengine.composition_v2_experiment import ExperimentalChromosome

        rescored = ExperimentalChromosome(config, solution.genes)
        rescored.exposure = solution.exposure
        rescored.evaluate_fitness(lots_override=solution.lots)
        solution = rescored
    if guarded:
        # HOLD/rotation was decided before the fill, just as in a live plan.
        # Do not cancel executed trades after observing the execution Close.
        rebalance_diagnostics.update(decision_locked=True, selected_utility=float(solution.fitness),
                                     execution_adjusted=affordable_lots != ga_signal_lots)
    if (not solution.budget_ok or
            (not solution.diversification_ok and
             not (frozen and len(affordable_lots) >= config.min_stocks))):
        raise ValueError("Lot eksekusi melanggar batas saham atau budget.")
    deferred_signals = _deferred_untraded_signals(old_lots, ga_signal_lots, untraded)
    if deferred_signals:
        rebalance_diagnostics = rebalance_diagnostics or {}
        rebalance_diagnostics['deferred_signals'] = deferred_signals
    if experimental_fitness:
        rebalance_diagnostics = rebalance_diagnostics or {}
        rebalance_diagnostics.update(fitness_policy=config.fitness_policy,
            fitness_diagnostics=solution.fitness_diagnostics,
            score_basis='decision_prices_with_execution_fitted_lots')
    if frozen:
        rebalance_diagnostics = rebalance_diagnostics or {}
        rebalance_diagnostics['frozen_lots'] = {code: old_lots[code] for code in sorted(frozen)}
        rebalance_diagnostics['fitness_excludes_frozen_outside_universe'] = bool(
            frozen - set(market_data.stock_codes))
    if experimental_profile_v2:
        from src.gaengine.composition_v2_experiment import validate_execution
    else:
        from src.gaengine.profile_policy import validate_execution
    if frozen - set(market_data.stock_codes) and config.profile_policy != "legacy":
        profile_composition = {
            "version": "composition_v2" if experimental_profile_v2 else config.profile_policy,
            "compliant": None,
            "reason": "Posisi tanpa perdagangan di luar kandidat tidak tercakup dalam metrik komposisi.",
        }
    else:
        profile_composition = validate_execution(
            config, solution.lots,
            np.asarray([execution_price_map[c] * 100 for c in market_data.stock_codes]),
        )

    # 4. Susun laporan hasil
    if getattr(market_data, "market_stress", None) is not None:
        evidence = market_data.market_stress
        logger.info(
            "GA risk model=%s | events=%d archive_end=%s | stress_risk=%s coverage=%s uncertainty_penalty=%.6f",
            config.risk_model, len(evidence.events), evidence.history_end,
            solution.market_stress_risk, solution.market_stress_coverage,
            solution.market_stress_missing_penalty,
        )
    codes = market_data.stock_codes
    prices = (np.asarray([float(ga_prices[code] * 100) for code in codes], dtype=float)
              if backtest else market_data.prices_per_lot)
    lots = solution.lots
    alloc = lots * prices
    total = float(_marked_holdings_value(affordable_lots, execution_price_map))

    rows = sorted(
        ((code, int(qty), float(execution_price_map[code] * 100))
         for code, qty in affordable_lots.items()),
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
        new_lots_dict = dict(affordable_lots)
        new_weights_dict = {code: (qty * execution_price_map[code] * 100 / total if total else 0)
                            for code, qty in affordable_lots.items()}
        old_weights_dict = {code: w for code, w in zip(market_data.stock_codes, old_weights_aligned)}
        
        rebalance_deltas = compute_rebalance_deltas(
            old_lots=old_lots,
            new_lots=new_lots_dict,
            old_weights=old_weights_dict,
            new_weights=new_weights_dict,
            eliminated_stocks=eliminated_stocks,
            old_cost_basis=old_cost_basis,
            execution_prices=execution_price_map,
        )
        for delta in rebalance_deltas:
            if delta["kode"] in frozen:
                delta["reason"] = "Tidak diperdagangkan pada hari eksekusi; lot lama tetap dimiliki"
                delta["action_detail"] = "HOLD_NO_TRADE"

    # Estimasi return tahunan dari alokasi terhadap modal termasuk sisa kas.
    expected_return = None
    if total > 0 and not (frozen - set(codes)) and market_data.returns.shape[0] >= solution.n_active:
        aw = alloc / float(request.budget)
        p_daily = (market_data.returns * aw[:, None]).sum(axis=0)
        p_daily = p_daily[~np.isnan(p_daily) & ~np.isinf(p_daily)]
        if p_daily.size:
            expected_return = float(p_daily.mean() * 252.0)

    # 5a. date_ref acuan portofolio (kolom Portofolio.date_ref):
    #     - BACKTEST : tanggal simulasi dari parameter.
    #     - LIVE     : sekarang (selaras created_at).
    from datetime import datetime as _dt
    if backtest:
        date_ref_dt = _dt.combine(date_ref, _dt.min.time())
    else:
        # Quote provenance and position inception are different dates. Never
        # grant live performance before the recommendation was created.
        rebalance_diagnostics = rebalance_diagnostics or {}
        rebalance_diagnostics.update(bear_assessment_date=(preview['as_of'] if getattr(request, 'bear_management', False) else None),
                                     valuation_mark_date=execution_day.isoformat(),
                                     live_date_policy='formation-wib-v1')
        execution_day = live_formation_day
        date_ref_dt = _dt.combine(live_formation_day, _dt.min.time())

    new_lots_for_accounting = {code: int(lot) for code, lot, _ in rows if lot > 0}
    changed_tickers = {code for code in set(old_lots) | set(new_lots_for_accounting)
                       if old_lots.get(code, 0) != new_lots_for_accounting.get(code, 0)}
    average_buy_prices, _ = _apply_position_snapshot(
        old_lots, old_cost_basis, new_lots_for_accounting, execution_price_map, 0.0,
    )
    if backtest and changed_tickers:
        traded_day, _ = _execution_close_prices(
            db, changed_tickers, execution_day, True, require_volume=True,
        )
        if traded_day != execution_day:
            raise MarketDataUnavailableError("Tanggal perdagangan saham berbeda dari hari eksekusi.")
    execution_prices = {
        code: str(ga_prices.get(code, _money(all_close_prices[code])))
        for code in sorted(changed_tickers)
    }
    cash_before = _money(old_portfolio_res.sisa_budget if old_portfolio_id else request.budget) + applied_contribution
    cash_after = cash_before + sum(
        (_money(old_lots.get(code, 0)) - _money(new_lots_for_accounting.get(code, 0)))
        * 100 * _money(execution_prices[code]) for code in changed_tickers
    )
    from src.gaengine.guarded_rebalance import transaction_account
    _, transaction_costs = transaction_account(
        old_lots, new_lots_for_accounting,
        {c: float(execution_price_map[c]) for c in changed_tickers}, float(cash_before),
        config.buy_fee_rate, config.sell_fee_rate,
    )
    cash_after -= _money(transaction_costs)
    fallback_sources = {code: source for code, source in (getattr(market_data, 'price_sources', None) or {}).items()
                        if source in {'filtered_stock_cache', 'market_data_db'} and code in new_lots_for_accounting}
    if fallback_sources:
        rebalance_diagnostics = rebalance_diagnostics or {}
        rebalance_diagnostics.update(execution_price_sources=fallback_sources,
                                     fallback_price_date=market_data.price_date.isoformat())
    rebalance_diagnostics = rebalance_diagnostics or {}
    if experimental_profile_v2:
        rebalance_diagnostics["profile_policy"] = "composition_v2"
        rebalance_diagnostics["profile_composition"] = profile_composition
    if rebalance_diagnostics is not None:
        rebalance_diagnostics.update(execution_transaction_costs=transaction_costs,
            buy_fee_rate=config.buy_fee_rate, sell_fee_rate=config.sell_fee_rate,
            execution_delta_lots={c:new_lots_for_accounting.get(c,0)-old_lots.get(c,0)
                                  for c in sorted(changed_tickers)})
    holdings_after = _marked_holdings_value(new_lots_for_accounting, execution_price_map)
    if exposure_limit is not None and exposure_limit < 1.0:
        execution_nav = cash_after + holdings_after
        frozen_value = sum(
            _money(new_lots_for_accounting.get(code, 0)) * 100 * _money(execution_price_map[code])
            for code in frozen
        )
        unavoidable = frozen_value > execution_nav * _money(exposure_limit) + Decimal("0.01")
        if execution_nav <= 0 or (holdings_after > execution_nav * _money(exposure_limit) + Decimal("0.01")
                                  and not unavoidable):
            raise ValueError(
                "Exposure portofolio melebihi batas modal terpilih pada harga eksekusi "
                f"(saham Rp{holdings_after:.2f}, kas Rp{cash_after:.2f}, "
                f"NAV Rp{execution_nav:.2f}, batas {exposure_limit:.4%}, "
                f"biaya Rp{transaction_costs:.2f})."
            )
        if unavoidable:
            rebalance_diagnostics = rebalance_diagnostics or {}
            rebalance_diagnostics['exposure_cap_blocked_by_frozen_lots'] = True
            logger.warning("Batas exposure tidak dapat dicapai: posisi tanpa perdagangan bernilai Rp%s", frozen_value)
    if cash_after < Decimal("-0.01"):
        raise ValueError("Rebalance menghasilkan kas negatif; snapshot dibatalkan.")
    # budget is the pre-trade NAV at the valuation mark. If an actual/simulated
    # execution differs from that mark, the difference is trading slippage,
    # not an external cash flow or an invalid snapshot.

    # 4a. Narasi TEMPLATE (selalu dibuat): dipakai apa adanya bila narasi LLM
    #     tidak tersedia / gagal, sehingga endpoint generate tidak pernah gagal
    #     hanya karena layanan LLM bermasalah.
    narasi = (
        f"Portofolio dengan profil risiko {request.risk_profile} memiliki {len(affordable_lots)} "
        f"saham (GA menilai {market_data.n_stocks} kandidat). Total dana terpakai "
        f"Rp{total:,.0f} dari budget Rp{request.budget:,.0f}. Skor fitness yang dicapai "
        f"{solution.fitness:.4f} dengan Sharpe ratio {solution.sharpe_ratio:.3f}. "
        f"Penurunan maksimum (max drawdown) tercatat {solution.max_drawdown:.2%} "
        f"dan korelasi rata-rata antar saham {solution.avg_correlation:.3f}. "
        f"Sisa kas setelah biaya Rp{cash_after:,.0f}, dengan biaya transaksi Rp{transaction_costs:,.0f}. "
        f"Rekomendasi ini dihasilkan Algoritma Genetika."
    )

    # 4b. Narasi LLM (OpenAI): menjelaskan MENGAPA & BAGAIMANA portofolio ini
    #     terpilih oleh GA -- metrik fitness function (Sharpe, MDD, korelasi,
    #     bonus fundamental), batasan budget/diversifikasi, serta peran tiap
    #     emiten pemenang. Bahan/prompt dirakit src/gaengine/prompt_builder.py,
    #     pemanggilan model ada di src/backend/services/openai_llm_caller.py.
    #     Bila gagal (tanpa API key, timeout, dsb) -> None, narasi template dipakai.
    report_progress("narrative", "Merangkai penjelasan untuk Anda",
                    "Membandingkan hasil, merangkum risiko, dan mencari konteks berita sesuai tanggal keputusan.")
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
        reporting={
            "cash_after": float(cash_after),
            "transaction_costs": float(transaction_costs),
            "investment_total": float(holdings_after),
            "execution_prices": {code: float(price) for code, price in execution_price_map.items()},
            "exposure_limit": exposure_limit,
            "uncovered_positions": sorted(frozen - set(market_data.stock_codes)),
        },
    )
    if narasi_llm:
        narasi = narasi_llm
    if not narasi_llm:
        from src.gaengine.narrative_evidence import risk_notes
        notes = risk_notes(solution, config)
        if notes:
            narasi += " " + " ".join(notes)
    if guarded and not narasi_llm:
        narasi += (
            f" Mode guarded_v1: keputusan {rebalance_diagnostics['reason']}. "
            f"Skor keputusan {solution.fitness:.6f}; HOLD persis dibandingkan "
            "dengan perubahan posisi setelah biaya. Skenario historis "
            "belum merupakan prediksi return tervalidasi."
        )
    if frozen:
        narasi += (f" Saham tanpa perdagangan pada {execution_day}: "
                   f"{', '.join(sorted(frozen))}; lot lama tetap dimiliki dan tidak diperdagangkan.")
        if frozen - set(market_data.stock_codes):
            narasi += " Skor GA tidak mencakup posisi tertahan di luar kandidat filter."
    logger.info(
        "Narasi portofolio memakai %s.",
        ("penjelasan fakta terverifikasi (fallback OpenAI)"
         if getattr(narasi_llm, "source", None) == "verified_template"
         else "penjelasan LLM (OpenAI)" if narasi_llm else "template GA (fallback)"),
    )
    rebalance_diagnostics["narrative_source"] = getattr(narasi_llm, "source", "openai" if narasi_llm else "ga_template")
    rebalance_diagnostics["narrative_version"] = NARRATIVE_VERSION
    if getattr(narasi_llm, "news_context", None) is not None:
        rebalance_diagnostics["news_context"] = narasi_llm.news_context

    if backtest and not narasi_llm:
        narasi += f" Simulasi memakai data historis per {date_ref}, bukan kondisi pasar saat ini."

    report_progress("saving", "Menyimpan hasil analisis",
                    "Menyiapkan rincian portofolio agar siap Anda tinjau.")
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

    replace_query = db.query(Portofolio).filter(Portofolio.user_id == user_uuid)
    if old_portfolio_id:
        # A selected series must never replace an unrelated active portfolio.
        replace_query = replace_query.filter(Portofolio.id == UUID(old_portfolio_id))
    else:
        replace_query = replace_query.filter(Portofolio.status_portofolio == status_lama)
    replaced = replace_query.update(
        {"status_portofolio": status_digantikan}, synchronize_session=False,
    )
    if replaced:
        logger.info(
            "Supersede (%s): %d portofolio '%s' -> '%s'.",
            "backtest" if backtest else "live", replaced, status_lama, status_digantikan,
        )
    if old_portfolio_id:
        old_snapshot_row.pending_deposit = 0.0

    # 6. Simpan hasil ke database: portofolios + portofolio_items
    portofolio = Portofolio(
        user_id=user_uuid,
        is_list_active=not getattr(request, 'rebalance', False),
        name_portofolio=name_portofolio,
        description=description,
        budget=request.budget,
        total_terpakai=float(holdings_after),
        sisa_budget=float(cash_after),
        capital_contribution=float(applied_contribution),
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
        parent_portofolio_id=UUID(str(old_portfolio_id)) if old_portfolio_id and getattr(request, 'rebalance', False) else None,
        turnover_rate=float(solution.turnover_rate) if getattr(request, 'rebalance', False) else None,
        turnover_penalty_beta=float(config.turnover_penalty_beta) if getattr(request, 'rebalance', False) else None,
        # acuan tanggal portofolio: tanggal simulasi (backtest) / sekarang (live)
        date_ref=date_ref_dt,
        execution_date=execution_day,
        execution_prices=execution_prices,
        transaction_costs=transaction_costs,
        rebalance_metadata=rebalance_diagnostics,
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
    if getattr(request, 'rebalance', False) and old_portfolio_id:
        old_items_with_ticker = _query_items_with_ticker(db, UUID(str(old_portfolio_id)))
        _close_previous_items(db, old_items_with_ticker, execution_day)

    items = []
    for ticker, lot, price in rows:
        if lot > 0:
            action = "buy"
            start_dt = _dt.combine(execution_day, _dt.min.time())
            harga_beli = average_buy_prices[ticker]
            
            if getattr(request, 'rebalance', False) and ticker in delta_map:
                action = delta_map[ticker]["action"].lower()
            
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
        check_cancelled()
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
        execution_prices=portofolio.execution_prices,
        price_date=execution_day if backtest else market_data.price_date,
        is_list_active=getattr(portofolio, "is_list_active", None),
        fitness_policy=config.fitness_policy,
        fitness_diagnostics=getattr(solution, "fitness_diagnostics", None),
        profile_composition=profile_composition,
        transaction_costs=transaction_costs,
        rebalance_diagnostics=rebalance_diagnostics,
        deferred_signals=deferred_signals,
        parent_portofolio_id=str(old_portfolio_id) if old_portfolio_id else None,
        execution_date=execution_day,
        name_portofolio=name_portofolio,
        description=description,
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
        capital_contribution=float(applied_contribution),
    )


# ----------------------------------------------------------------------
# Rebalance series: lanjutkan snapshot tersimpan tiap N bulan
# ----------------------------------------------------------------------
# Batas jumlah tanggal (langkah) rebalance. Tiap langkah menjalankan GA penuh
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
    Controller REBALANCE SERIES (khusus BACKTEST): melanjutkan snapshot yang
    sudah tersimpan, lalu rebalance berantai pada tanggal berikutnya.

    Alur (eksekusi SEKUENSIAL, bukan paralel -- tiap langkah bergantung pada
    hasil langkah sebelumnya: NAV, old_lots, dan status active_backtest):
      1. start_date adalah tanggal snapshot tersimpan; GA tidak dijalankan lagi.
      2. tiap +interval bulan: rebalance dari ID snapshot terakhir yang sukses.

    Semua langkah memakai mode backtest (date_ref = tanggal jadwal, data
    historis parquet) sehingga portofolio LIVE tidak tersentuh. Portofolio
    terakhir berstatus "active_backtest", langkah sebelumnya menjadi
    "replaced_backtest" (tetap tersimpan di DB).

    Kegagalan per langkah:
      - langkah gagal MarketDataUnavailableError (mis. tanggal di
        luar cakupan parquet) -> dicatat pada `skipped`, series LANJUT dari
        portofolio terakhir yang berhasil
      - error lain (ticker tak terdaftar, SQL, dll.) -> series berhenti dan
        error diteruskan; portofolio yang sudah terlanjur commit TIDAK dihapus
      - composition_v2 yang tidak menemukan komposisi patuh -> tanggal dicatat
        sebagai skipped; komposisi lama tetap aktif dan tidak memakai legacy

    Returns: RebalanceSeriesResponse berisi snapshot lama dalam rangkaian
    beserta hasil baru, urut kronologis dari portofolio awal,
    plus daftar tanggal yang dilewati beserta alasannya.

    Raises: ValueError (rentang salah / jumlah langkah > MAX_SERIES_STEPS),
            UserNotFoundError,
            StockNotFoundError.
    """
    user_uuid = _parse_user_uuid(user_id, db)
    base = _resolve_action_portfolio(
        db, user_uuid, request.base_portfolio_id, backtest=True,
    )
    if request.start_date != _snapshot_date(base):
        raise ValueError("Tanggal mulai harus sama dengan tanggal snapshot backtest terbaru yang dipilih.")
    if request.risk_profile != base.risk_profile:
        raise ValueError("Profil risiko tidak sesuai dengan portofolio yang dipilih.")
    if request.profile_policy == "composition_v2" and request.rebalance_model != "legacy":
        raise ValueError("composition_v2 belum mendukung guarded_v1; pilih rebalance legacy.")
    schedule = build_rebalance_schedule(
        request.start_date, request.end_date, request.interval_months
    )[1:]
    if not schedule:
        raise ValueError("Tanggal akhir harus mencakup minimal satu jadwal rebalance.")
    if len(schedule) > MAX_SERIES_STEPS:
        raise ValueError(
            f"Jumlah langkah series ({len(schedule)}) melebihi batas "
            f"{MAX_SERIES_STEPS}. Perbesar interval_months atau perpendek rentang."
        )
    policy = "legacy"
    resolved_bear_choices: dict[date, str] = {}
    if request.bear_management:
        user = db.query(User).filter(User.id == UUID(user_id)).first()
        if user is None:
            raise UserNotFoundError("User tidak ditemukan.")
        policy = user.bear_allocation_policy
        if policy == "manual":
            if request.bear_choices is None or set(request.bear_choices) != set(schedule):
                raise ValueError("Mode manual memerlukan pilihan modal untuk setiap tanggal series.")
        for day in schedule:
            preview = preview_bear_market(request.risk_profile, True, day)
            if (preview["as_of"] is not None and
                str((request.bear_preview_dates or {}).get(day)) != preview["as_of"]):
                raise ValueError(f"Pratinjau tanggal {day} sudah berubah; muat ulang.")
            if (request.bear_preview_fingerprints or {}).get(day) != preview["fingerprint"]:
                raise ValueError(f"Indikator tanggal {day} berubah sejak pratinjau; muat ulang.")
            choice = resolve_bear_choice(policy, preview["status"], (request.bear_choices or {}).get(day))
            effective_fraction(preview, request.risk_profile, choice)
            resolved_bear_choices[day] = choice
    if not math.isfinite(request.contribution_amount) or request.contribution_amount < 0:
        raise ValueError("Nominal setoran berkala tidak valid.")
    if request.contribution_amount > 0 and request.contribution_interval_months is None:
        raise ValueError("Isi contribution_interval_months saat setoran berkala diaktifkan.")
    contribution_dates = (
        build_rebalance_schedule(request.start_date, request.end_date,
                                 request.contribution_interval_months)[1:]
        if request.contribution_amount > 0 else []
    )

    logger.info(
        "=== REBALANCE SERIES (backtest): %d tanggal, %s s/d %s, interval %d bulan ===",
        len(schedule), request.start_date, request.end_date, request.interval_months,
    )

    portfolios: list[PortfolioResponse] = [
        _to_portfolio_response(db, node)
        for node in _portfolio_lineage(db, base, user_uuid)
    ]
    skipped: list[SkippedRebalanceStep] = []
    last_applied_date = request.start_date
    last_portfolio_id = base.id

    from src.gaengine.profile_policy import ProfileConstraintError

    for index, tanggal in enumerate(schedule):
        set_progress_context(series_index=index + 1, series_total=len(schedule),
                             series_date=tanggal.isoformat())
        report_progress("market", "Menyiapkan rebalance berikutnya",
                        f"Membaca data historis untuk jadwal {tanggal.isoformat()}.")
        due_count = sum(last_applied_date < due <= tanggal for due in contribution_dates)
        # Request BARU tiap langkah: generate_new_portfolio memutasi budget
        # (override NAV) pada objek request, jadi tidak boleh dipakai ulang.
        step_request = PortfolioGenerateRequest(
            budget=request.budget,
            fitness_policy=request.fitness_policy,
            fitness_horizon_days=request.fitness_horizon_days,
            fitness_cvar_tail_fraction=request.fitness_cvar_tail_fraction,
            fitness_lambda_cvar=request.fitness_lambda_cvar,
            charge_transaction_fees=request.charge_transaction_fees,
            rebalance_model=getattr(request, "rebalance_model", "legacy"),
            rebalance_buy_fee_rate=getattr(request, "rebalance_buy_fee_rate", .0015),
            rebalance_sell_fee_rate=getattr(request, "rebalance_sell_fee_rate", .0025),
            profile_policy=getattr(request, "profile_policy", "legacy"),
            risk_profile=request.risk_profile,
            name_portofolio=request.name_portofolio,
            description=request.description,
            backtest=True,
            date_ref=tanggal,
            rebalance=True,
            rebalance_portfolio_id=last_portfolio_id,
            contribution_amount=due_count * request.contribution_amount,
            bear_management=request.bear_management,
            bear_choice=resolved_bear_choices.get(tanggal),
            bear_preview_as_of=(request.bear_preview_dates or {}).get(tanggal),
            bear_preview_fingerprint=(request.bear_preview_fingerprints or {}).get(tanggal),
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
            db.rollback()
            logger.warning(
                "Rebalance series: langkah %d/%d (%s) dilewati -> %s",
                index + 1, len(schedule), tanggal, e,
            )
            skipped.append(SkippedRebalanceStep(date=tanggal, reason=str(e)))
            report_progress("skipped", "Jadwal ini dilewati",
                            "Data historis belum memadai. Melanjutkan ke jadwal berikutnya.")
            continue
        except ProfileConstraintError as e:
            if request.profile_policy != "composition_v2":
                raise
            db.rollback()
            logger.warning(
                "Rebalance series v2: langkah %d/%d (%s) dilewati karena batas profil: %s",
                index + 1, len(schedule), tanggal, e,
            )
            skipped.append(SkippedRebalanceStep(date=tanggal, reason=str(e)))
            report_progress("skipped", "Jadwal ini dilewati",
                            "Komposisi yang memenuhi profil belum ditemukan. Melanjutkan jadwal berikutnya.")
            continue
        except ValueError as e:
            logger.error("Rebalance series gagal pada %s (fitness=%s, model=%s): %s",
                         tanggal, request.fitness_policy, request.rebalance_model, e)
            raise ValueError(f"Rebalance series gagal pada {tanggal}: {e}") from e

        portfolios[-1].status_portofolio = STATUS_REPLACED_BACKTEST
        portfolios.append(hasil)
        last_applied_date = tanggal
        last_portfolio_id = UUID(hasil.id)
        logger.info(
            "Rebalance series: langkah %d/%d selesai (%s, rebalance=%s, id=%s).",
            index + 1, len(schedule), tanggal, True, hasil.id,
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
        evaluation_mode=("legacy" if not request.bear_management else
                         "automatic_strategy" if policy in ("always_follow", "always_full") else "what_if"),
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


def _to_portfolio_response(
    db: Session, portofolio: Portofolio,
    replay_cache: dict | None = None,
    known_portfolios: dict | None = None,
) -> PortfolioResponse:
    """Bangun PortfolioResponse dari satu baris Portofolio + item-itemnya.

    Dipakai BERSAMA oleh get_active_portfolio & get_all_my_portfolios agar bentuk
    respons selalu identik. Kas dan alokasi snapshot harus berasal dari kolom
    Portofolio yang disimpan saat eksekusi; perubahan harga_beli milik user
    tidak merupakan BUY/SELL atau perubahan kas. Average beli snapshot
    rebalance dihitung ulang dari rantai transaksi agar data lama juga akurat.
    """
    item_pairs = _query_items_with_ticker(db, portofolio.id)
    items = [pair[0] for pair in item_pairs]
    average_buy_prices = (
        _rebalance_cost_basis(db, portofolio, portofolio.user_id,
                              replay_cache=replay_cache,
                              known_portfolios=known_portfolios,
                              allow_legacy_reference=True)
        if getattr(portofolio, "is_rebalance", False) else None
    )
    allocations = [_to_item_response(item, ticker) for item, ticker in item_pairs]
    if average_buy_prices is not None:
        for allocation in allocations:
            allocation.harga_beli = average_buy_prices.get(allocation.ticker)

    metadata = getattr(portofolio, "rebalance_metadata", None) or {}
    return PortfolioResponse(
        id=str(portofolio.id),
        is_list_active=getattr(portofolio, "is_list_active", None),
        holdings_edited=bool(metadata.get("holdings_edited")),
        price_date=metadata.get('valuation_mark_date') or portofolio.execution_date,
        execution_prices=portofolio.execution_prices,
        fitness_policy=metadata.get("fitness_policy", "legacy"),
        fitness_diagnostics=metadata.get("fitness_diagnostics"),
        parent_portofolio_id=(str(portofolio.parent_portofolio_id)
                             if getattr(portofolio, "parent_portofolio_id", None) else None),
        execution_date=getattr(portofolio, "execution_date", None),
        name_portofolio=getattr(portofolio, "name_portofolio", None),
        description=getattr(portofolio, "description", None),
        user_id=str(portofolio.user_id),
        fitness_score=portofolio.fitness_score,
        sharpe_ratio=portofolio.sharpe_ratio,
        expected_return=None,  # tidak disimpan di DB (hanya dihitung saat generate)
        max_drawdown=portofolio.max_drawdown,
        avg_correlation=portofolio.avg_correlation,
        skor_fundamental=portofolio.skor_fundamental,
        total_terpakai=portofolio.total_terpakai,
        sisa_budget=portofolio.sisa_budget,
        n_active=sum(1 for item in items if item.jumlah_lot > 0),
        allocated_budget_ok=portofolio.sisa_budget >= -0.01,
        risk_profile=portofolio.risk_profile,
        status_portofolio=portofolio.status_portofolio,
        created_at=portofolio.created_at,
        date_ref=portofolio.date_ref,
        budget=portofolio.budget,
        allocations=allocations,
        narasi_llm=portofolio.narasi_llm,
        pending_deposit=getattr(portofolio, "pending_deposit", 0) or 0.0,
        capital_contribution=getattr(portofolio, "capital_contribution", 0) or 0.0,
        transaction_costs=getattr(portofolio, "transaction_costs", 0) or 0.0,
        rebalance_diagnostics=getattr(portofolio, "rebalance_metadata", None),
        deferred_signals=metadata.get("deferred_signals", []),
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

    known_portfolios = {portofolio.id: portofolio for portofolio in rows}
    replay_cache: dict = {}
    return [
        _to_portfolio_response(db, portofolio, replay_cache, known_portfolios)
        for portofolio in rows
    ]


def set_portfolio_list_activity(
    db: Session, user_id: str, portfolio_id: str, is_active: bool,
) -> dict:
    """Set user-managed root activity without changing snapshot execution status."""
    user_uuid = _parse_user_uuid(user_id, db)
    try:
        portfolio_uuid = UUID(str(portfolio_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise PortfolioNotFoundError("Portofolio tidak ditemukan.") from exc
    portfolio = db.query(Portofolio).filter(
        Portofolio.id == portfolio_uuid, Portofolio.user_id == user_uuid,
    ).with_for_update().first()
    if portfolio is None:
        raise PortfolioNotFoundError("Portofolio tidak ditemukan.")
    if portfolio.parent_portofolio_id is not None:
        raise ValueError("Status aktif hanya dapat diatur pada portofolio awal.")
    portfolio.is_list_active = is_active
    db.commit()
    return {"portfolio_id": str(portfolio.id), "is_list_active": is_active}


def delete_portfolio_lineage(db: Session, user_id: str, portfolio_id: str) -> dict:
    """Hapus snapshot terpilih dan seluruh descendant rebalance secara atomik.

    Ancestor tetap ada. Bila snapshot aktif termasuk yang dihapus, ancestor
    terdekat atau snapshot tersisa terbaru pada mode yang sama menjadi aktif.
    Item ancestor yang tidak lagi memiliki child dibuka kembali masa berlakunya.
    """
    user_uuid = _parse_user_uuid(user_id, db)
    try:
        target_id = UUID(str(portfolio_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise PortfolioNotFoundError("Portofolio tidak ditemukan.") from exc

    target = (
        db.query(Portofolio)
        .filter(Portofolio.id == target_id, Portofolio.user_id == user_uuid)
        .with_for_update()
        .first()
    )
    if target is None:
        raise PortfolioNotFoundError("Portofolio tidak ditemukan.")

    backtest = target.status_portofolio.endswith("_backtest")
    active_status = STATUS_ACTIVE_BACKTEST if backtest else STATUS_ACTIVE
    replaced_status = STATUS_REPLACED_BACKTEST if backtest else STATUS_REPLACED
    if target.parent_portofolio_id is not None:
        parent = (
            db.query(Portofolio)
            .filter(Portofolio.id == target.parent_portofolio_id)
            .with_for_update()
            .first()
        )
        if parent is not None and (
            parent.user_id != user_uuid or
            parent.status_portofolio.endswith("_backtest") != backtest
        ):
            raise PortfolioLineageConflictError(
                "Rantai rebalance tidak konsisten; portofolio tidak dihapus."
            )
    levels: list[list[Portofolio]] = [[target]]
    seen = {target.id}
    while levels[-1]:
        children = (
            db.query(Portofolio)
            .filter(Portofolio.parent_portofolio_id.in_([row.id for row in levels[-1]]))
            .with_for_update()
            .all()
        )
        if not children:
            break
        for child in children:
            if child.id in seen or child.user_id != user_uuid or (
                child.status_portofolio.endswith("_backtest") != backtest
            ):
                raise PortfolioLineageConflictError(
                    "Rantai rebalance tidak konsisten; portofolio tidak dihapus."
                )
            seen.add(child.id)
        levels.append(children)

    snapshots = [row for level in levels for row in level]
    deleted_ids = [row.id for row in snapshots]
    deleted_set = set(deleted_ids)
    deleted_active = any(row.status_portofolio == active_status for row in snapshots)
    retained_parents = {
        row.parent_portofolio_id for row in snapshots
        if row.parent_portofolio_id is not None and row.parent_portofolio_id not in deleted_set
    }

    item_count = (
        db.query(PortofolioItem)
        .filter(PortofolioItem.portofolio_id.in_(deleted_ids))
        .delete(synchronize_session=False)
    )
    # Hapus dari daun ke parent agar tetap aman pada database yang FK parent-nya
    # belum memakai ON DELETE SET NULL.
    for level in reversed(levels):
        for row in level:
            db.query(Portofolio).filter(Portofolio.id == row.id).delete(
                synchronize_session=False
            )

    for parent_id in retained_parents:
        retained_parent = db.query(Portofolio.id).filter(
            Portofolio.id == parent_id, Portofolio.user_id == user_uuid,
        ).first()
        has_child = db.query(Portofolio.id).filter(
            Portofolio.parent_portofolio_id == parent_id
        ).first() is not None
        if retained_parent is not None and not has_child:
            db.query(PortofolioItem).filter(
                PortofolioItem.portofolio_id == parent_id
            ).update({PortofolioItem.end_date: None}, synchronize_session=False)

    remaining = (
        db.query(Portofolio)
        .filter(
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio.in_((active_status, replaced_status)),
        )
        .order_by(Portofolio.created_at.desc(), Portofolio.id.desc())
        .all()
    )
    active = next((row for row in remaining if row.status_portofolio == active_status), None)
    if deleted_active and active is None and remaining:
        parent_ids = {row.parent_portofolio_id for row in remaining}
        leaves = [row for row in remaining if row.id not in parent_ids]
        active = next(
            (row for row in leaves if row.id == target.parent_portofolio_id),
            leaves[0] if leaves else None,
        )
        if active is not None:
            active.status_portofolio = active_status
            db.query(PortofolioItem).filter(
                PortofolioItem.portofolio_id == active.id
            ).update({PortofolioItem.end_date: None}, synchronize_session=False)

    db.commit()
    return {
        "deleted_portfolio_ids": [str(row_id) for row_id in deleted_ids],
        "deleted_portfolios": len(deleted_ids),
        "deleted_items": item_count,
        "active_portfolio_id": str(active.id) if active is not None else None,
    }


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
            name_portofolio=getattr(p, "name_portofolio", None),
            description=getattr(p, "description", None),
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


def update_portfolio_identity(
    db: Session, user_id: str, portfolio_id: str, name: str, description: str | None,
) -> PortfolioResponse:
    """Set one user-facing name and description across a live rebalance chain."""
    user_uuid = _parse_user_uuid(user_id, db)
    try:
        portfolio_uuid = UUID(portfolio_id)
    except (ValueError, TypeError) as exc:
        raise PortfolioNotFoundError("ID portofolio tidak valid.") from exc
    portfolio = db.query(Portofolio).filter(
        Portofolio.id == portfolio_uuid, Portofolio.user_id == user_uuid,
    ).first()
    if portfolio is None:
        raise PortfolioNotFoundError("Portofolio tidak ditemukan.")
    if portfolio.status_portofolio not in (STATUS_ACTIVE, STATUS_REPLACED):
        raise ValueError("Nama portofolio live hanya dapat diubah pada rangkaian live.")
    if db.query(Portofolio.id).filter(
        Portofolio.parent_portofolio_id == portfolio_uuid,
        Portofolio.user_id == user_uuid,
    ).first():
        raise ValueError("Pilih snapshot terbaru untuk mengubah identitas rangkaian.")
    lineage = _portfolio_lineage(db, portfolio, user_uuid)
    for snapshot in lineage:
        snapshot.name_portofolio = name
        snapshot.description = description
    db.commit()
    return _to_portfolio_response(db, portfolio)

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
    return _to_portfolio_response(db, portfolio)


class ItemNotFoundError(Exception):
    """Exception domain: item portofolio tidak ditemukan / bukan milik user."""


def _resolve_action_portfolio(db: Session, user_uuid: UUID, portfolio_id=None, backtest=False):
    """Lock the selected series leaf; a historical snapshot is never a trade target."""
    active = STATUS_ACTIVE_BACKTEST if backtest else STATUS_ACTIVE
    replaced = STATUS_REPLACED_BACKTEST if backtest else STATUS_REPLACED
    query = db.query(Portofolio).filter(Portofolio.user_id == user_uuid)
    if portfolio_id is None:
        query = query.filter(Portofolio.status_portofolio == active).order_by(
            Portofolio.created_at.desc(), Portofolio.id.desc())
    else:
        query = query.filter(Portofolio.id == UUID(str(portfolio_id)))
    portfolio = query.with_for_update().first()
    if portfolio is None:
        raise PortfolioNotFoundError('Portofolio tujuan tidak ditemukan pada akun Anda.')
    db.refresh(portfolio)
    allowed = (active, replaced) if portfolio_id is not None else (active,)
    if portfolio.status_portofolio not in allowed:
        raise ValueError('Mode portofolio tujuan tidak sesuai. Muat ulang halaman.')
    if db.query(Portofolio.id).filter(Portofolio.parent_portofolio_id == portfolio.id).first():
        raise ValueError('Portofolio sudah memiliki rebalance berikutnya. Muat ulang posisi terbaru.')
    return portfolio


def add_live_capital(db: Session, user_id: str, amount: float, portfolio_id=None) -> PortfolioResponse:
    """Catat setoran sebagai kas tertunda, tanpa mengubah snapshot historis/NAV.

    Dana baru masuk ke kas eksekusi dan NAV pada rebalance live berikutnya.
    """
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Nominal setoran harus lebih dari nol dan berupa angka terbatas.")
    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise UserNotFoundError("User ID pada token tidak valid.") from exc
    portfolio = _resolve_action_portfolio(db, user_uuid, portfolio_id)
    pending = _money(portfolio.pending_deposit or 0) + _money(amount)
    if not pending.is_finite():
        raise ValueError("Total setoran tidak valid.")
    portfolio.pending_deposit = float(pending)
    db.commit()
    db.refresh(portfolio)
    return _to_portfolio_response(db, portfolio)


def update_harga_beli(db: Session, user_id: str, item_id: str, harga_beli: float) -> PortofolioItem:
    """
    Ubah harga_beli (per lembar) pada satu item portofolio milik user.
    Validasi kepemilikan: item harus berada di portofolio milik user (dari JWT)
    dan berstatus active - portofolio lama (replaced) tidak boleh diedit.
    total_investasi ikut dihitung ulang: jumlah_lot * 100 * harga_beli.
    Raises: ItemNotFoundError.
    """
    from uuid import UUID

    if not math.isfinite(harga_beli) or harga_beli <= 0:
        raise ValueError("Harga beli per lembar harus berupa angka positif yang terbatas.")

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

    # Keep the old price-only API compatible with the inline position editor,
    # including explicit basis overrides equal to the snapshot reference price.
    from src.backend.services.portfolio_holdings import update_live_holding
    from src.backend.models.schemas.portfolio_schema import UpdateHoldingRequest
    update_live_holding(db, user_id, str(item.portofolio_id), item.stock.ticker,
        UpdateHoldingRequest(jumlah_lot=item.jumlah_lot, expected_lots=item.jumlah_lot,
                             harga_beli=harga_beli))
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


def _sync_prices_on_demand(db: Session, reason: str, stock_ids: list | None = None) -> dict | None:
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
        stats = (sync_market_data(db, stock_ids=stock_ids) if stock_ids is not None
                 else sync_market_data(db))
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
def _portfolio_lineage(
    db: Session, portfolio: Portofolio, user_uuid: UUID,
    known_portfolios: dict | None = None,
) -> list[Portofolio]:
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
        parent = (known_portfolios or {}).get(current.parent_portofolio_id)
        current = parent if parent is not None else (
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
    replay_cache: dict | None = None,
    known_portfolios: dict | None = None,
    allow_legacy_reference: bool = False,
) -> dict[str, float | None]:
    """Replay recorded lot deltas to recover the held positions' average cost.

    Display/GA callers may reconstruct legacy buys from the saved acquisition
    reference price. Strict accounting callers still require execution prices.
    Live PATCH corrections
    are applied to the remaining lots, matching the performance endpoint.
    Used for GA input and read-time average buy prices; it does not rewrite
    cash or historical snapshots.
    """
    is_backtest = portfolio.status_portofolio in (
        STATUS_ACTIVE_BACKTEST, STATUS_REPLACED_BACKTEST
    )
    positions: dict[str, int] = {}
    basis: dict[str, float | None] = {}
    for node in _portfolio_lineage(db, portfolio, user_uuid, known_portfolios):
        cache_key = (node.id, allow_legacy_reference)
        if replay_cache is not None and cache_key in replay_cache:
            cached_positions, cached_basis = replay_cache[cache_key]
            positions, basis = dict(cached_positions), dict(cached_basis)
            continue
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
            # Older simulations stored their purchase mark on the item,
            # before execution_prices existed. Use that mark only for added
            # lots; never reset the remaining position to today's mark.
            if (price is None and allow_legacy_reference and node.execution_prices is None
                    and next_positions.get(code, 0) > positions.get(code, 0)):
                try:
                    price = float(items[code].get("reference_price"))
                except (TypeError, ValueError, OverflowError):
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
                if ((getattr(node, 'rebalance_metadata', None) or {}).get('manual_cost_basis', {}).get(code) == purchase or
                    (abs(item_total - shares * purchase) <= 0.01 and
                     abs(shares * purchase - shares * reference) > 0.01)):
                    basis[code] = purchase
        positions = next_positions
        if replay_cache is not None:
            replay_cache[cache_key] = (dict(positions), dict(basis))
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

    Setoran tercatat sebagai arus dana eksternal pada snapshot rebalance.
    Dividen bruto diakui sebagai pendapatan/piutang sejak ex-date berdasarkan
    posisi sebelum ex-date. File sumber tidak memiliki tanggal pembayaran;
    piutang tidak menjadi kas transaksi dan tidak direinvestasikan otomatis.
    Biaya, pajak, penarikan dan corporate action lain belum dimodelkan.
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
            "capital_contribution": float(getattr(node, "capital_contribution", 0) or 0),
            "positions": _snapshot_positions(db, node.id),
            "execution_prices": node.execution_prices,
            "transaction_costs": float(getattr(node, "transaction_costs", 0) or 0),
            "manual_cost_basis": (getattr(node, 'rebalance_metadata', None) or {}).get('manual_cost_basis', {}),
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
    if not stock_ids:
        # A manually corrected root can be entirely cash. Keep its original
        # quote universe only to obtain observed valuation sessions, not positions.
        zero_items = _query_items_with_ticker(db, portfolio.id)
        stock_ids = [item.stock_id for item, _ in zero_items]
        stock_id_to_ticker = {str(item.stock_id): code for item, code in zero_items}

    # Load only the stored universe required by the complete rebalance chain.
    sync_stats = None
    splits_by_date = {}
    if not is_backtest:
        price_rows = _query_price_rows(db, stock_ids, price_query_start, valuation_end)
        available_ids = {str(row.stock_id) for row in price_rows
                         if row.close is not None and math.isfinite(row.close) and row.close > 0}
        missing_ids = [stock_id for stock_id in stock_ids if str(stock_id) not in available_ids]
        if missing_ids:
            sync_stats = _sync_prices_on_demand(
                db, reason=f"histori saham belum lengkap untuk performa {portfolio_id}",
                stock_ids=missing_ids,
            )
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
            if row.close is not None and math.isfinite(row.close) and row.close > 0
        ]
        if not records:
            raise PortfolioNotFoundError(
                "Belum ada harga Close valid untuk rentang performa. " + _describe_sync(sync_stats)
            )
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
        if "Stock Splits" in window:
            split_values = pd.to_numeric(window["Stock Splits"], errors="coerce")
            split_rows = window.loc[split_values.notna() & ~split_values.isin([0, 1])]
            splits_by_date = {day: set(group["Ticker"]) for day, group in split_rows.groupby("Date")}
        window["price"] = window["Close"]
        df_price = window.pivot_table(index="Date", columns="Ticker", values="price", aggfunc="last").sort_index()

    df_price.index = pd.to_datetime(df_price.index)
    if valuation_end is not None:
        df_price = df_price.loc[df_price.index.date <= valuation_end]
    if df_price.empty:
        raise PortfolioNotFoundError("Tidak ada harga sebelum akhir window item snapshot.")
    # Live positions become effective on their formation day, which can be
    # later than the completed Close session used by GA. Keep a one-day mark
    # from the persisted snapshot so later price syncs do not move that trade.
    formation_marks: dict[date, dict[str, float]] = {}
    if not is_backtest:
        observed_days = set(df_price.index.date)
        for snapshot in snapshots:
            mark_day = snapshot["date"]
            if (mark_day in observed_days or snapshot["execution_prices"] is None or
                    valuation_end is not None and mark_day > valuation_end):
                continue
            marks = {
                ticker: data.get("reference_price")
                for ticker, data in snapshot["positions"].items()
            }
            for ticker, raw in snapshot["execution_prices"].items():
                if ticker not in marks:
                    try:
                        marks[ticker] = float(raw)
                    except (TypeError, ValueError):
                        pass
            if marks and all(
                price is not None and math.isfinite(price) and price > 0
                for price in marks.values()
            ):
                formation_marks[mark_day] = marks
        if formation_marks:
            df_price = df_price.reindex(
                df_price.index.union(pd.to_datetime(list(formation_marks))).sort_values()
            )
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
    for mark_day, marks in formation_marks.items():
        df_price.loc[pd.Timestamp(mark_day), list(marks)] = list(marks.values())
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
    price_accounting_status = "verified"
    dividend_events: list[dict] = []
    dividend_data = {"source": "data/dividend_events_20_tahun.xlsx", "status": "available",
                     "coverage": "recorded_events_only", "reasons": []}
    try:
        dividend_history = load_dividend_history()
        dividend_schedule = dividend_entitlements(dividend_history, snapshots, df_price.index.max().date())
        dividend_income: Decimal | None = Decimal("0")
        dividend_data.update({
            "first_event_date": str(dividend_history.first_event_date),
            "last_event_date": str(dividend_history.last_event_date),
        })
        if dividend_history.last_event_date is not None and df_price.index.max().date() > dividend_history.last_event_date:
            dividend_data["status"] = "partial"
            dividend_data["reasons"].append(
                f"Akhir valuasi melewati event terakhir dalam arsip dividen ({dividend_history.last_event_date}); dividen setelahnya belum dapat dipastikan lengkap."
            )
    except DividendDataUnavailableError as exc:
        dividend_schedule = []
        dividend_income = None
        dividend_data["status"] = "unavailable"
        dividend_data["reasons"].append(str(exc))
        logger.warning("Dividen performa %s tidak tersedia: %s", portfolio_id, exc)
    dividend_index = 0
    prior_nav: Decimal | None = None
    total_contributions = Decimal("0")
    cumulative_growth: Decimal | None = Decimal("1")
    cumulative_transaction_costs = Decimal("0")
    tolerance = Decimal("0.05")

    def mark_problem(message: str, legacy: bool = False, *, price_ledger: bool = True) -> None:
        nonlocal accounting_status, price_accounting_status
        if message not in reasons:
            reasons.append(message)
        if accounting_status != "invalid":
            accounting_status = "legacy_unverified" if legacy else "invalid"
        if price_ledger and price_accounting_status != "invalid":
            price_accounting_status = "legacy_unverified" if legacy else "invalid"

    for reason in dividend_data["reasons"]:
        mark_problem(reason, legacy=True, price_ledger=False)
    for mark_day in formation_marks:
        mark_problem(
            f"Valuasi {mark_day} memakai harga acuan snapshot dari sesi selesai terakhir; harga pasar pada hari pembentukan belum tersimpan.",
            legacy=True,
        )

    split_affected = set()
    for timestamp in df_price.index:
        day = timestamp.date()
        # Current ledger does not reconcile split lot quantities. Keep the
        # existing NAV API, but do not certify its subsequent risk metrics.
        split_affected.update(ticker for ticker in splits_by_date.get(timestamp, set())
                              if positions.get(ticker, 0) > 0)
        day_contributions = Decimal("0")
        day_dividends = Decimal("0") if dividend_income is not None else None
        while dividend_index < len(dividend_schedule) and dividend_schedule[dividend_index]["ex_date"] <= day:
            entitlement = dividend_schedule[dividend_index]
            dividend_income += entitlement["amount"]
            day_dividends += entitlement["amount"]
            dividend_events.append({
                **{key: value for key, value in entitlement.items()
                   if key not in ("ex_date", "dividend_per_share", "amount")},
                "ex_date": entitlement["ex_date"].isoformat(),
                "recognition_date": day.isoformat(),
                "dividend_per_share": float(entitlement["dividend_per_share"]),
                "amount": float(entitlement["amount"]),
            })
            dividend_index += 1
        while active_snapshot_index < len(snapshots) and snapshots[active_snapshot_index]["date"] <= day:
            snapshot = snapshots[active_snapshot_index]
            fee = _money(snapshot['transaction_costs'])
            if not fee.is_finite() or fee < 0:
                mark_problem(f"Biaya snapshot {snapshot['id']} tidak valid.")
                cash_balance = None
                fee = Decimal('0')
            cumulative_transaction_costs += fee
            contribution = _money(snapshot["capital_contribution"])
            if not contribution.is_finite() or contribution < 0 or (active_snapshot_index == 0 and contribution):
                mark_problem(f"Setoran snapshot {snapshot['id']} tidak valid.")
                contribution = Decimal("0")
            day_contributions += contribution
            total_contributions += contribution
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
            nav_before = (cash_before + before_holdings + dividend_income
                          if cash_before is not None and before_known and dividend_income is not None else None)
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
                    next_cash = cash_before + contribution - fee
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
                    if (snapshot['manual_cost_basis'].get(ticker) == purchase or
                        (abs(_money(item_total) - edited_total) <= tolerance and
                         abs(edited_total - reference_total) > tolerance)):
                        cost_basis[ticker] = _money(purchase)
            positions = next_positions
            after_nav = None
            if cash_balance is not None and dividend_income is not None and all(
                ticker in valid_tickers and pd.notna(df_price.at[timestamp, ticker]) for ticker in positions
            ):
                after_nav = cash_balance + dividend_income + sum(
                    lots * 100 * _money(df_price.at[timestamp, ticker]) for ticker, lots in positions.items()
                )
            if active_snapshot_index > 0:
                events.append({
                    "portfolio_id": snapshot["id"], "execution_date": day.isoformat(),
                    "delta_lots": dict(sorted(delta.items())),
                    "cash_before": float(cash_before) if cash_before is not None else None,
                    "capital_contribution": float(contribution),
                    "transaction_costs": float(fee),
                    "cash_after": float(cash_balance) if cash_balance is not None else None,
                    "nav_before": float(nav_before) if nav_before is not None else None,
                    "nav_after": float(after_nav) if after_nav is not None else None,
                    "realized_pnl_delta": float(event_realized) if event_realized is not None else None,
                    "dividend_receivable": float(dividend_income) if dividend_income is not None else None,
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
            if (day in formation_marks or ticker not in actual_prices.columns or
                    pd.isna(actual_prices.at[timestamp, ticker])):
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
        nav_excluding_dividends = cash_balance + holdings_value if cash_balance is not None and holdings_known else None
        nav = (nav_excluding_dividends + dividend_income
               if nav_excluding_dividends is not None and dividend_income is not None else None)
        daily_return = ((nav - day_contributions) / prior_nav - 1
                        if nav is not None and prior_nav not in (None, 0) else
                        (nav / initial_capital - 1 if nav is not None and prior_nav is None and not rows else None))
        if cumulative_growth is not None and daily_return is not None:
            cumulative_growth *= 1 + daily_return
        elif nav is None:
            cumulative_growth = None
        cumulative_return = cumulative_growth - 1 if nav is not None and cumulative_growth is not None else None
        rows.append({
            "date": timestamp, "portfolio_value": nav, "cash_balance": cash_balance,
            "holdings_market_value": holdings_value if holdings_known else None,
            "portfolio_value_excluding_dividends": nav_excluding_dividends,
            "dividend_income": day_dividends, "dividend_income_cumulative": dividend_income,
            "dividend_receivable": dividend_income,
            "daily_return": daily_return, "portfolio_return": cumulative_return,
            "capital_contribution": day_contributions,
            "total_contributions": total_contributions,
            "realized_pnl_cumulative": realized_pnl, "unrealized_pnl": unrealized_pnl,
            "stale_price_tickers": sorted(stale_tickers),
            "unreconciled_split_tickers": sorted(split_affected),
        })
        prior_nav = nav

    if active_snapshot_index < len(snapshots):
        raise PortfolioNotFoundError("Snapshot yang diminta belum tercakup oleh rentang harga/end_date.")

    series_df = pd.DataFrame(rows).set_index("date")
    ihsg_return = None

    benchmark_observed_dates = set()
    benchmark_previous_dates = {}
    benchmark_prices = {}
    market_session_dates = None
    market_previous_dates = {}

    def benchmark_on_portfolio_days(index_closes):
        index_closes = index_closes.dropna().sort_index()
        benchmark_observed_dates.update(index_closes.index)
        benchmark_prices.update(index_closes.to_dict())
        benchmark_previous_dates.update({day: previous.date().isoformat()
                                        for previous, day in zip(index_closes.index, index_closes.index[1:])})
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
                from src.preprocessing.ihsg_sessions import validated_ihsg_rows
                idx_data = validated_ihsg_rows(idx_data)
                idx = idx_data.set_index(pd.to_datetime(idx_data["Date"]))["Close"].dropna().sort_index()
                ihsg_return = benchmark_on_portfolio_days(idx)
        except Exception as exc:  # benchmark failure must not hide NAV
            logger.warning("Gagal membaca IHSG dari Parquet: %s", exc)

    if is_backtest:
        from src.preprocessing.ihsg_sessions import portfolio_session_calendar
        market_session_dates = portfolio_session_calendar(benchmark_observed_dates, series_df.index)
        ordered_sessions = sorted(market_session_dates or [])
        market_previous_dates = {day: previous.date().isoformat()
                                 for previous, day in zip(ordered_sessions, ordered_sessions[1:])}

    series = []
    for timestamp, row in series_df.iterrows():
        ihsg = ihsg_return.loc[timestamp] if ihsg_return is not None and timestamp in ihsg_return.index else None
        series.append({
            "date": timestamp.isoformat(),
            "portfolio_value": float(row["portfolio_value"]) if pd.notna(row["portfolio_value"]) else None,
            "cash_balance": float(row["cash_balance"]) if pd.notna(row["cash_balance"]) else None,
            "holdings_market_value": float(row["holdings_market_value"]) if pd.notna(row["holdings_market_value"]) else None,
            "portfolio_value_excluding_dividends": float(row["portfolio_value_excluding_dividends"]) if pd.notna(row["portfolio_value_excluding_dividends"]) else None,
            "dividend_income": float(row["dividend_income"]) if pd.notna(row["dividend_income"]) else None,
            "dividend_income_cumulative": float(row["dividend_income_cumulative"]) if pd.notna(row["dividend_income_cumulative"]) else None,
            "dividend_receivable": float(row["dividend_receivable"]) if pd.notna(row["dividend_receivable"]) else None,
            "daily_return": float(row["daily_return"]) if pd.notna(row["daily_return"]) else None,
            "portfolio_return": float(row["portfolio_return"]) if pd.notna(row["portfolio_return"]) else None,
            "capital_contribution": float(row["capital_contribution"]),
            "total_contributions": float(row["total_contributions"]),
            "realized_pnl_cumulative": float(row["realized_pnl_cumulative"]) if pd.notna(row["realized_pnl_cumulative"]) else None,
            "unrealized_pnl": float(row["unrealized_pnl"]) if pd.notna(row["unrealized_pnl"]) else None,
            "stale_price_tickers": row["stale_price_tickers"],
            "unreconciled_split_tickers": row["unreconciled_split_tickers"],
            "ihsg_return": float(ihsg) if ihsg is not None and pd.notna(ihsg) else None,
            "ihsg_observed": timestamp in benchmark_observed_dates,
            "ihsg_previous_observed_date": benchmark_previous_dates.get(timestamp),
            "is_market_session": (timestamp in market_session_dates if market_session_dates is not None else None),
            "previous_market_session_date": market_previous_dates.get(timestamp),
        })

    final_state = positions
    final_row = series[-1] if series else None
    return_metrics = calculate_return_metrics(
        initial_capital=float(initial_capital),
        final_nav=final_row["portfolio_value"] if final_row else None,
        start_date=actual_start.isoformat(),
        end_date=df_price.index.max().date().isoformat(),
        twr_return_pct=(final_row["portfolio_return"] * 100
                        if final_row and final_row["portfolio_return"] is not None else None),
        contributions=[(point["date"], point["capital_contribution"]) for point in series],
    )
    return {
        **return_metrics,
        "start_date": actual_start.isoformat(),
        "requested_start_date": requested_start.isoformat(),
        "start_date_adjusted": actual_start != requested_start,
        "end_date": df_price.index.max().date().isoformat(),
        "holdings": {ticker: lots for ticker, lots in sorted(final_state.items()) if lots > 0},
        "missing_price_tickers": sorted(set(missing_price_tickers)),
        "root_portfolio_id": str(lineage[0].id),
        "evaluated_portfolio_id": str(portfolio.id),
        "benchmark_series": [{"date": day.isoformat(),
                              "close": float(benchmark_prices[day]) if day in benchmark_prices else None}
                             for day in sorted(market_session_dates or benchmark_observed_dates)],
        "rebalance_events": events,
        "accounting_status": accounting_status,
        "price_accounting_status": price_accounting_status,
        "split_check_status": "local_archive" if is_backtest else "unavailable_live_ledger",
        "accounting_reasons": reasons,
        "initial_capital": float(initial_capital),
        "total_contributions": float(total_contributions),
        "final_nav": final_row["portfolio_value"] if final_row else None,
        "final_nav_excluding_dividends": final_row["portfolio_value_excluding_dividends"] if final_row else None,
        "net_investment_gain": (final_row["portfolio_value"] - float(initial_capital + total_contributions)
                                if final_row and final_row["portfolio_value"] is not None else None),
        # Portfolio_return remains a decimal for chart compatibility; this
        # summary field is an actual percentage as its name states.
        "total_return_pct": (
            final_row["portfolio_return"] * 100.0
            if final_row and final_row["portfolio_return"] is not None
            else None
        ),
        "realized_pnl": final_row["realized_pnl_cumulative"] if final_row else None,
        "unrealized_pnl": final_row["unrealized_pnl"] if final_row else None,
        "dividend_income": final_row["dividend_income_cumulative"] if final_row else None,
        "dividend_receivable": final_row["dividend_receivable"] if final_row else None,
        "dividend_events": dividend_events,
        "dividend_data": dividend_data,
        "dividend_policy": {"recognition": "ex_date_accrual", "payment_dates_available": False,
                            "cash_payments_modeled": False, "reinvested": False, "tax_modeled": False},
        "transaction_costs": float(cumulative_transaction_costs) if accounting_status == "verified" else None,
        "accounting_limitations": [
            "Setoran pada snapshot rebalance tercatat sebagai arus dana eksternal dan dikeluarkan dari return; setoran live yang masih menunggu rebalance belum masuk NAV. Penarikan belum dimodelkan.",
            "Biaya transaksi memakai nilai tersimpan pada setiap snapshot (default nol untuk snapshot lama); pajak dan corporate action selain dividen tunai belum dimodelkan.",
            "Dividen bruto memakai event dalam arsip lokal dan kepemilikan sebelum ex-date. Tanggal pembayaran tidak tersedia: pendapatan ditahan sebagai piutang dalam NAV, tidak menambah kas transaksi dan tidak direinvestasikan. Kelengkapan arsip dan penyesuaian nominal akibat stock split perlu diperiksa terhadap sumber.",
            "Valuasi memakai raw Close; cost basis backtest dihitung weighted-average dari harga eksekusi, sedangkan harga_beli yang diubah pengguna menjadi koreksi basis live tanpa mengubah kas.",
            "Backtest mengambil keputusan dari data sampai sesi sebelumnya lalu mensimulasikan eksekusi pada Close sesi efektif; event dengan kas tidak cukup ditolak.",
            "Harga valuasi terakhir dapat diteruskan maksimum tujuh hari kalender; setelah itu NAV tidak dilaporkan tanpa harga baru.",
        ],
        "series": series,
    }
