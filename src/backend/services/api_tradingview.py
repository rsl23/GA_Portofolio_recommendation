"""ZAPI TradingView fundamentals for Indonesian equities (live only)."""
import logging
import math

import requests

from src.backend.core.config import settings

logger = logging.getLogger(__name__)


def fetch_api_tradingview_fundamentals(code: str):
    """Fetch one full quote; core scope omits essential balance-sheet data."""
    if not settings.X_API_KEY:
        logger.warning("ZAPI API key unavailable; cannot fetch TradingView fundamentals for %s", code)
        return None
    endpoint = f"{settings.BASE_URL.rstrip('/')}/finance:tradingview/quote"
    try:
        response = requests.get(
            endpoint,
            headers={"x-api-key": settings.X_API_KEY, "Accept": "application/json"},
            params={"symbol": f"IDX:{code}", "market": "indonesia", "scope": "full"},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning("TradingView fundamentals unavailable for %s (%s)", code, type(exc).__name__)
        return None


def _number(value):
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _divide(numerator, denominator):
    return numerator / denominator if math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0 else float("nan")


def parse_tradingview_fundamentals(payload, close, *, code=None):
    """Return cache units: IDR EPS, valuation multiples, ROE/yield fractions.

    Close is the completed session from the existing OHLCV pipeline. Ignore
    upstream PE/PB fields, which can refer to another price. DER uses total debt
    / total equity to match the backtest definition, rather than the provider's
    debt/common-equity ratio or total liabilities/equity ratio.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return None
    data = payload["data"]
    if code and data.get("symbol") != f"IDX:{code}":
        return None
    # Never divide an IDR quote by unmapped USD earnings or book value.
    currencies = [data.get(key) for key in ("fundamentalCurrencyCode", "currencyFund") if data.get(key)]
    idr_basis = data.get("currency") == "IDR" and bool(currencies) and all(c == "IDR" for c in currencies)
    price = _number(close)
    if not math.isfinite(price) or price <= 0:
        price = float("nan")
    eps = _number(data.get("earningsPerShareTtm")) if idr_basis else float("nan")
    book = _number(data.get("bookValuePerShareFq")) if idr_basis else float("nan")
    debt = _number(data.get("totalDebtFq"))
    equity = _number(data.get("totalEquityFq"))
    roe = _number(data.get("returnOnCommonEquityTtm"))
    return {
        "eps": eps,
        "per": _divide(price, eps),
        "pbv": _divide(price, book),
        "roe": roe / 100.0,
        "der": _divide(debt, equity) if debt >= 0 else float("nan"),
        "dividend_yield": _number(data.get("dividendYieldPercent")) / 100.0,
    }
