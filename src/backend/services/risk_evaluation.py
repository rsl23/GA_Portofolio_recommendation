"""Descriptive, flow-adjusted out-of-sample risk evaluation (no optimizer calls)."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from src.gaengine.decision_utility import empirical_cvar

VERSION = "risk-evaluation-v5"
# Research hypotheses, not industry suitability standards. Applied to total NAV.
PROFILE_CAPS = {"Conservative": .8, "Moderate": 1., "Aggressive": 1.3}
PROFILE_ALIASES = {"konservatif": "Conservative", "moderat": "Moderate", "agresif": "Aggressive",
                   **{name.lower(): name for name in PROFILE_CAPS}}
WINDOW = 63
MIN_CAPTURE = 20


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _capture(portfolio, benchmark, rising):
    selected = (benchmark > 0 if rising else benchmark < 0) & portfolio.notna()
    p, b = portfolio[selected], benchmark[selected]
    count = int(selected.sum())
    if count < MIN_CAPTURE:
        return {"pct": None, "sessions": count, "status": "insufficient_directional_sessions"}
    if (p <= -1).any() or (b <= -1).any():
        return {"pct": None, "sessions": count, "status": "invalid_returns"}
    pg = float(np.expm1(np.log1p(p).mean()))
    bg = float(np.expm1(np.log1p(b).mean()))
    return {"pct": pg / bg * 100 if abs(bg) > 1e-10 else None, "sessions": count,
            "status": "available" if abs(bg) > 1e-10 else "benchmark_constant"}


def _drawdown(returns, dates):
    # A missing observation breaks the wealth path; never bridge it silently.
    if len(returns) < 1 or returns.isna().any() or (returns < -1).any():
        return {"mdd_pct": None, "max_underwater_sessions": None,
                "current_underwater_sessions": None, "worst_recovery_sessions": None,
                "worst_trough_date": None, "recovered": None,
                "status": "insufficient_history" if len(returns) < 1 else "incomplete_nav"}
    wealth = np.r_[1., np.cumprod(1 + returns.to_numpy())]
    peaks = np.maximum.accumulate(wealth)
    underwater = wealth < peaks - 1e-12
    duration = maximum = 0
    for below in underwater:
        duration = duration + 1 if below else 0
        maximum = max(maximum, duration)
    dd = 1 - wealth / peaks
    trough = int(np.argmax(dd))
    recovery = np.flatnonzero(wealth[trough + 1:] >= peaks[trough] - 1e-12)
    has_loss = dd[trough] > 1e-12
    return {"status": "available", "mdd_pct": float(dd[trough] * 100), "max_underwater_sessions": maximum,
            "current_underwater_sessions": duration,
            "worst_recovery_sessions": (int(recovery[0] + 1) if len(recovery) else None) if has_loss else 0,
            "worst_trough_date": dates[trough] if has_loss else None,
            "recovered": bool(len(recovery)) if has_loss else True}


def _absolute_statistics(returns):
    daily = returns.dropna()
    n = len(daily)
    return {
        "volatility_pct": float(daily.std(ddof=1) * np.sqrt(252) * 100) if n >= 20 else None,
        "volatility_status": "available" if n >= 20 else "incomplete_nav" if len(returns) >= 20 else "insufficient_history",
        "downside_deviation_pct": float(np.sqrt(np.mean(np.minimum(daily, 0) ** 2)) * np.sqrt(252) * 100) if n >= 20 else None,
        "daily_cvar_95_pct": float(empirical_cvar(-daily.to_numpy(), .05) * 100) if n >= 100 else None,
        "cvar_status": "available" if n >= 100 else "incomplete_nav" if len(returns) >= 100 else "insufficient_tail_history",
        "cvar_min_sessions": 100,
    }


def snapshot_profile_cap(metadata, profile):
    """Use the saved v2 rule; never relabel an old snapshot with a new limit."""
    metadata = metadata or {}
    if metadata.get("profile_policy") != "composition_v2":
        return None
    composition = metadata.get("profile_composition") or {}
    canonical = PROFILE_ALIASES.get(str(profile).strip().lower())
    if canonical == composition.get("profile"):
        value = _number((composition.get("limits") or {}).get("max_vol_ratio"))
        return float(value) if math.isfinite(value) and value > 0 else None
    from src.gaengine.composition_v2_experiment import POLICY_VERSION, LIMITS
    if composition.get("policy_version") == POLICY_VERSION and canonical in LIMITS:
        return LIMITS[canonical].max_vol_ratio
    return None


def evaluate_risk(performance: dict, profile: str, start_date=None, end_date=None,
                  *, profile_cap=None) -> dict:
    """Rebase to the selected window. Fees remain in NAV; deposits are excluded.

    Absolute metrics describe the complete recorded simulation NAV (cash,
    holdings and accrued dividends). Corporate-action flags do not change that
    recorded path; they remain an audit limitation, not a validation certificate.
    Comparisons separately exclude dividends to match the IHSG price index.
    """
    profile = PROFILE_ALIASES.get(str(profile).strip().lower())
    if profile is None:
        raise ValueError("Profil risiko tidak dikenal.")
    cap = PROFILE_CAPS[profile] if profile_cap is None else float(profile_cap)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("Batas risiko profil tidak valid.")
    selected_rows = [row for row in performance.get("series", [])
            if (not start_date or row["date"][:10] >= str(start_date))
            and (not end_date or row["date"][:10] <= str(end_date))]
    rows = [row for row in selected_rows if row.get("is_market_session") is not False]
    affected = sorted({ticker for row in rows for ticker in row.get("unreconciled_split_tickers", [])})
    dates = [row["date"][:10] for row in rows]
    verified = performance.get("price_accounting_status", performance.get("accounting_status")) == "verified"
    event_dates = {str(event["execution_date"])[:10] for event in performance.get("rebalance_events", [])}
    # Do not silently discard an external cash flow or trade on an unverified session.
    verified = verified and not any(row.get("is_market_session") is False
        and (_number(row.get("capital_contribution", 0)) != 0 or row["date"][:10] in event_dates)
        for row in selected_rows)
    nav = pd.Series([_number(row.get("portfolio_value_excluding_dividends")) for row in rows], dtype=float)
    total_nav = pd.Series([_number(row.get("portfolio_value")) for row in rows], dtype=float)
    flows = pd.Series([_number(row.get("capital_contribution")) for row in rows], dtype=float)
    benchmark = pd.Series([_number(row.get("ihsg_return")) for row in rows], dtype=float) + 1
    estimated_sessions = sum(bool(row.get("stale_price_tickers", [])) for row in rows)
    observed = pd.Series([row.get("ihsg_observed") is True for row in rows], dtype=bool)
    price_return = (nav - flows) / nav.shift() - 1
    price_valid = nav.shift().gt(0) & nav.ge(0) & verified
    nav_adjacent = pd.Series([i > 0 and (row.get("is_market_session") is not True
        or row.get("previous_market_session_date") == dates[i - 1])
        for i, row in enumerate(rows)], dtype=bool)
    price_valid &= nav_adjacent
    adjacent = pd.Series([i > 0 and row.get("ihsg_previous_observed_date") == dates[i - 1]
                          for i, row in enumerate(rows)], dtype=bool)
    price_return = price_return.where(price_valid)
    total_return = ((total_nav - flows) / total_nav.shift() - 1).where(
        total_nav.shift().gt(0) & total_nav.ge(0) & nav_adjacent & verified)
    absolute = _absolute_statistics(total_return.iloc[1:])
    b_return = benchmark.pct_change(fill_method=None).where(observed & observed.shift(fill_value=False))
    paired = price_return.notna() & b_return.notna() & adjacent & b_return.gt(-1) & price_return.ge(-1)
    p, b = price_return.where(paired), b_return.where(paired)
    n = int(paired.sum())
    pstd, bstd = p.std(ddof=1), b.std(ddof=1)
    valid_std = n >= 20 and pd.notna(bstd) and bstd > 1e-10
    rolling_b = b.rolling(WINDOW, min_periods=WINDOW).std(ddof=1)
    rolling = (p.rolling(WINDOW, min_periods=WINDOW).std(ddof=1) / rolling_b).where(rolling_b > 1e-10)
    valid_windows = rolling.dropna()
    compliance = float(valid_windows.le(cap + 1e-12).mean() * 100) if len(valid_windows) else None
    latest = float(rolling.iloc[-1]) if len(rolling) and pd.notna(rolling.iloc[-1]) else None
    # Reconstruct prior stock basket P&L, excluding fees, dividends and new cash.
    fees = {}
    for event in performance.get("rebalance_events", []):
        day = str(event["execution_date"])[:10]
        fees[day] = fees.get(day, 0.) + _number(event.get("transaction_costs"))
    holdings = pd.Series([_number(row.get("holdings_market_value")) for row in rows], dtype=float)
    basket = ((nav - nav.shift() - flows + pd.Series([fees.get(d, 0.) for d in dates], dtype=float))
              / holdings.shift()).where(price_valid & holdings.shift().gt(0) & paired)
    basket_std = basket.std(ddof=1)
    basket_ratio = (float(basket_std / b[basket.notna()].std(ddof=1))
                    if basket.count() >= 20 and b[basket.notna()].std(ddof=1) > 1e-10 else None)
    warnings = [f"Ambang {cap:g} kali IHSG dan jendela 63 sesi adalah aturan evaluasi eksperimen, bukan standar kesesuaian investor.",
                "Seluruh rentang dibandingkan dengan profil snapshot yang dipilih; perubahan profil historis belum dinilai per snapshot.",
                "IHSG memakai price return; perbandingan mengeluarkan piutang dividen. Kas tanpa bunga dan biaya transaksi tetap masuk NAV.",
                "Risiko dihitung dari NAV simulasi tersimpan, termasuk valuasi estimasi yang sudah ada. NAV yang hilang tidak disambung."]
    if not verified:
        warnings.append("Pembukuan belum berstatus verified; metrik risiko NAV tidak dihitung.")
    if affected:
        warnings.append("Stock split belum direkonsiliasi untuk " + ", ".join(affected)
                        + "; angka mendeskripsikan NAV simulasi yang tersimpan, bukan audit kebenaran lot/cash historis.")
    if performance.get("split_check_status") == "unavailable_live_ledger":
        warnings.append("Ledger live belum menyimpan riwayat stock split; angka perlu diverifikasi bila holdings melewati corporate action.")
    portfolio_status = ("pending_review" if not verified else
                        "simulation_estimate" if affected or estimated_sessions else "available")
    eligibility = "unverified_nav" if not verified else "eligible"
    relative_status = ("available" if valid_std else "benchmark_unavailable" if not observed.any() else
                         "insufficient_pairs" if n < 20 else "benchmark_constant")
    def window_status(available):
        return "available" if available else ("insufficient_history" if len(rows) <= WINDOW else "incomplete_window")
    drawdown = _drawdown(total_return.iloc[1:], dates)
    # Profile assessment uses TOTAL portfolio returns on matched market sessions.
    # Keep this separate from the legacy price-only comparison fields.
    profile_pairs = total_return.notna() & b_return.notna() & adjacent & b_return.gt(-1)
    profile_n = int(profile_pairs.sum())
    profile_bvol = b_return[profile_pairs].std(ddof=1)
    profile_pvol = total_return[profile_pairs].std(ddof=1)
    profile_valid = profile_n >= WINDOW and pd.notna(profile_bvol) and profile_bvol > 1e-10
    profile_actual = float(profile_pvol * np.sqrt(252) * 100) if profile_valid else None
    profile_limit = float(profile_bvol * np.sqrt(252) * 100 * cap) if profile_valid else None
    profile_assessment = dict(
        basis='matched_total_portfolio_volatility', sessions=profile_n, minimum_sessions=WINDOW,
        actual_volatility_pct=profile_actual, limit_volatility_pct=profile_limit,
        status=('within_limit' if profile_actual <= profile_limit else 'above_limit') if profile_valid else 'insufficient_data',
        notice='Batas eksperimen profil adalah ' + str(cap) + ' kali volatilitas IHSG pada sesi yang sama; bukan batas toleransi pribadi atau standar keuangan baku.')
    if eligibility != "eligible":
        drawdown["status"] = eligibility
    return {
        "version": VERSION, "start_date": dates[0] if dates else None, "end_date": dates[-1] if dates else None,
        "profile": profile, "basis": "recorded_total_simulation_nav",
        "profile_assessment": profile_assessment,
        "comparison_basis": "price_only_nav_vs_ihsg_price_return",
        "simulation_notice": "Risiko dihitung dari NAV hasil backtest yang tersimpan. Angka ini mendeskripsikan simulasi; validasi transaksi historis belum diaudit menyeluruh.",
        "portfolio_status": portfolio_status,
        "metric_eligibility": {**{metric: eligibility for metric in
            ("volatility", "compliance", "downside_capture", "upside_capture", "drawdown", "recovery")}},
        "availability": {"volatility": absolute["volatility_status"] if verified else "unverified_nav",
                         "relative_volatility": relative_status,
                         "compliance": window_status(bool(len(valid_windows))),
                         "latest_window": window_status(latest is not None)},
        "excluded_non_session_points": len(selected_rows) - len(rows),
        "nav_return_sessions": int(total_return.iloc[1:].notna().sum()),
        "estimated_valuation_sessions": estimated_sessions,
        "profile_cap": cap, "window_sessions": WINDOW, "minimum_capture_sessions": MIN_CAPTURE,
        "observations": max(0, len(rows) - 1), "paired_sessions": n,
        "coverage_pct": n / (len(rows) - 1) * 100 if len(rows) > 1 else None,
        "annualized_volatility_pct": absolute["volatility_pct"],
        "downside_deviation_pct": absolute["downside_deviation_pct"],
        "daily_cvar_95_pct": absolute["daily_cvar_95_pct"],
        "cvar_status": absolute["cvar_status"], "cvar_min_sessions": absolute["cvar_min_sessions"],
        "comparison_portfolio_volatility_pct": float(pstd * np.sqrt(252) * 100) if valid_std else None,
        "ihsg_annualized_volatility_pct": float(bstd * np.sqrt(252) * 100) if valid_std else None,
        "relative_volatility": float(pstd / bstd) if valid_std else None,
        "stock_basket_relative_volatility": basket_ratio,
        "latest_rolling_relative_volatility": latest, "valid_windows": len(valid_windows),
        "eligible_windows": max(0, len(rows) - WINDOW), "compliance_pct": compliance,
        "latest_status": ("within_limit" if latest <= cap else "above_limit") if latest is not None else "insufficient_data",
        "downside_capture": _capture(p, b, False), "upside_capture": _capture(p, b, True),
        "drawdown": drawdown,
        "notes": ["Batas profil dan jendela 63 sesi adalah aturan eksperimen penelitian.",
                  "Seluruh rentang dibandingkan dengan profil yang dipilih; risiko profil merupakan batas atas.",
                  "Perbandingan IHSG memakai sesi yang sama; data yang tidak tersedia tidak diisi dengan nol."],
        "warnings": warnings,
    }
