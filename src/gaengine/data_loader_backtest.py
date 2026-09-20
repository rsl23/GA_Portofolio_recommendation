"""
SHIM KOMPATIBILITAS (mode BACKTEST)
===================================

Logika loader kini TERPADU di :mod:`src.gaengine.data_loader` (live + backtest
dalam satu kode, hanya sumber datanya yang berbeda).

File ini dipertahankan hanya agar import lama tetap berjalan, mis.:

    from src.gaengine.data_loader_backtest import build_market_data

PENTING: loader backtest lama memakai file yang tidak ada lagi di folder
``data/`` (Daftar Saham - 20260401.xlsx & Kuartalan_lengkap (1).xlsx). Versi
terpadu mengambil universe + fundamental langsung dari:

    - run_live_preprocessing(backtest=True, date=date_ref)   (stock_filtering)
    - data/Master_OHLCV_15Tahun.parquet
    - data/fundamental_quarterly.parquet
    - data/dividend_events_20_tahun.xlsx
    - data/BI-7Day-RR.xlsx

Sehingga mode backtest kini dijalankan dengan:

    from src.gaengine.data_loader import build_market_data
    build_market_data(backtest=True, date_ref=date(2024, 6, 28))

Disarankan memakai import baru:
    from src.gaengine.data_loader import build_market_data
"""

from src.gaengine.data_loader import (  # noqa: F401
    build_market_data,
    _normalize_metrics,
    _parse_number,
    _load_dividend_yields,
    _load_bi_rate_history,
    _provide_backtest_data,
    MarketData,
)

__all__ = [
    "build_market_data",
    "_normalize_metrics",
    "_parse_number",
    "_load_dividend_yields",
    "_load_bi_rate_history",
    "_provide_backtest_data",
    "MarketData",
]