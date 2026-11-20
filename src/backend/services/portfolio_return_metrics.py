"""Return summaries from the already valued NAV and dated external deposits.

Actual/365 day count. Annual rates are published only after 365 elapsed days.
MWR solves XIRR, then de-annualizes it for the selected holding period. The
current ledger supports deposits only, giving a unique root when one exists.
Dividends, trades and fees belong in NAV, never in external cash flows.
"""

from datetime import date
import math


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _growth_pct(log_growth: float) -> float | None:
    try:
        result = math.expm1(log_growth) * 100
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _money_weighted_log_rate(initial_capital, final_nav, start, end, deposits):
    if final_nav <= 0 or start >= end:
        return None
    flows = [(start, initial_capital), *deposits]
    # Compare log future values to avoid overflow for long histories/extreme rates.
    def residual(rate):
        terms = [math.log(amount) + rate * (end - day).days / 365
                 for day, amount in flows if amount > 0]
        largest = max(terms)
        return largest + math.log(sum(math.exp(term - largest) for term in terms)) - math.log(final_nav)

    low, high = -700.0, 700.0
    if residual(low) > 0 or residual(high) < 0:
        return None
    for _ in range(160):
        middle = (low + high) / 2
        if residual(middle) > 0:
            high = middle
        else:
            low = middle
    return (low + high) / 2


def calculate_return_metrics(*, initial_capital, final_nav, start_date: str,
                             end_date: str, twr_return_pct,
                             contributions: list[tuple[str, float]]) -> dict:
    """Additive API fields; all rates are percentages, missing values stay null."""
    start, end = date.fromisoformat(start_date[:10]), date.fromisoformat(end_date[:10])
    days = max(0, (end - start).days)
    result = {
        "performance_days": days,
        "twr_return_pct": twr_return_pct if _finite(twr_return_pct) else None,
        "cash_return": None,
        "cash_return_pct": None,
        "money_weighted_return_pct": None,
        "money_weighted_return_annualized_pct": None,
        "annualized_twr_return_pct": None,
        "annualization_status": "insufficient_history" if days < 365 else "unavailable",
    }
    if days >= 365 and _finite(twr_return_pct) and twr_return_pct >= -100:
        result["annualized_twr_return_pct"] = (
            -100.0 if twr_return_pct == -100 else
            _growth_pct(math.log1p(twr_return_pct / 100) * 365 / days)
        )
        if result["annualized_twr_return_pct"] is not None:
            result["annualization_status"] = "available"
    if not _finite(initial_capital) or initial_capital <= 0 or not _finite(final_nav) or final_nav < 0:
        return result
    deposits = []
    for raw_day, amount in contributions:
        day = date.fromisoformat(raw_day[:10])
        if not _finite(amount) or amount < 0 or not start <= day <= end:
            return result
        if amount:
            deposits.append((day, amount))
    paid_capital = initial_capital + sum(amount for _, amount in deposits)
    gain = final_nav - paid_capital
    result["cash_return"] = gain
    result["cash_return_pct"] = gain / paid_capital * 100
    rate = _money_weighted_log_rate(initial_capital, final_nav, start, end, deposits)
    if rate is not None:
        result["money_weighted_return_pct"] = _growth_pct(rate * days / 365)
        if days >= 365:
            result["money_weighted_return_annualized_pct"] = _growth_pct(rate)
    return result
