from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class MarketData:
    """Everything the GA needs to evaluate a portfolio."""

    stock_codes: List[str]
    prices_per_lot: np.ndarray           # IDR price of one lot (100 shares)
    returns: np.ndarray                  # (n_stocks, T) aligned daily returns
    correlation: np.ndarray              # (n_stocks, n_stocks)
    fundamental_scores: np.ndarray       # (n_stocks,) composite 0-1
    fundamental_metrics: np.ndarray      # (n_stocks, 5) PER/PBV/ROE/DER/DivYld
    risk_free_rate: float                # annual fraction, e.g. 0.0575

    def __post_init__(self) -> None:
        self.prices_per_lot = np.asarray(self.prices_per_lot, dtype=float)
        self.returns        = np.asarray(self.returns, dtype=float)
        self.correlation    = np.asarray(self.correlation, dtype=float)
        self.fundamental_scores  = np.asarray(self.fundamental_scores, dtype=float)
        self.fundamental_metrics = np.asarray(self.fundamental_metrics, dtype=float)

    @property
    def n_stocks(self) -> int:
        return len(self.stock_codes)

