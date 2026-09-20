"""
SHIM KOMPATIBILITAS (mode LIVE)
==============================

Logika loader kini TERPADU di :mod:`src.gaengine.data_loader` (live + backtest
dalam satu kode, hanya sumber datanya yang berbeda).

File ini dipertahankan hanya agar import lama tetap berjalan, mis.:

    from src.gaengine.data_loader_live import build_market_data

Disarankan memakai import baru:
    from src.gaengine.data_loader import build_market_data
"""

from src.gaengine.data_loader import (  # noqa: F401
    build_market_data,
    take_ohlcv_data,
    take_fundamental_data,
    take_bi_rate,
    process_single_fundamental,
    _normalize_metrics,
    _parse_float_string,
    MarketData,
)

__all__ = [
    "build_market_data",
    "take_ohlcv_data",
    "take_fundamental_data",
    "take_bi_rate",
    "process_single_fundamental",
    "_normalize_metrics",
    "_parse_float_string",
    "MarketData",
]