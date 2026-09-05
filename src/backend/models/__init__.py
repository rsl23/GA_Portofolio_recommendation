# Import seluruh model agar terdaftar di satu registry SQLAlchemy (Base.metadata).
# Penting agar relationship antar model (User <-> Portofolio <-> PortofolioItem)
# dapat di-resolve dan create_all() membuat semua tabel.
from src.backend.models.users import User
from src.backend.models.portofolios import Portofolio
from src.backend.models.portofolio_items import PortofolioItem
from src.backend.models.stock_universe import StockUniverse
from src.backend.models.market_status import DailyMarketStatus
from src.backend.models.filtered_stocks_cache import FilteredStockCache
from src.backend.models.market_data import MarketData

__all__ = [
    "User",
    "Portofolio",
    "PortofolioItem",
    "StockUniverse",
    "DailyMarketStatus",
    "FilteredStockCache",
    "MarketData",
]
