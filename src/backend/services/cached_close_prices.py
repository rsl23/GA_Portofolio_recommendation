"""Validate a cached IDX Close as an execution/valuation quote for one session."""
from datetime import date, datetime
import math


def cached_close_for_session(record, session: date, volume) -> float | None:
    value = getattr(record, 'price_date', None)
    if not isinstance(value, (date, datetime, str)):
        return None
    try:
        day = value.date() if isinstance(value, datetime) else (
            date.fromisoformat(value[:10]) if isinstance(value, str) else value)
        price = float(record.close)
        traded_volume = float(volume)
    except (ValueError, TypeError, AttributeError):
        return None
    if (day != session or not math.isfinite(price) or price <= 0 or
            not math.isfinite(traded_volume) or traded_volume <= 0):
        return None
    return price
