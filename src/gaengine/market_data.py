from dataclasses import dataclass
from typing import List, Optional
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
    rebalance_signal: Optional[np.ndarray] = None  # recent trend/downside change, [-1, 1]

    def __post_init__(self) -> None:
        self.prices_per_lot = np.asarray(self.prices_per_lot, dtype=float)
        self.returns        = np.asarray(self.returns, dtype=float)
        self.correlation    = np.asarray(self.correlation, dtype=float)
        self.fundamental_scores  = np.asarray(self.fundamental_scores, dtype=float)
        self.fundamental_metrics = np.asarray(self.fundamental_metrics, dtype=float)
        if self.rebalance_signal is None:
            self.rebalance_signal = self._recent_risk_reward_change()
        else:
            self.rebalance_signal = np.asarray(self.rebalance_signal, dtype=float)
        if self.rebalance_signal.shape != (self.n_stocks,):
            raise ValueError("rebalance_signal harus memiliki satu nilai per saham.")

    def _recent_risk_reward_change(self) -> np.ndarray:
        """Change in 21-session risk/reward versus the preceding 63 sessions.

        The feature is computed once from the already-cut-off return window,
        not from future prices. It is a *change* signal so the GA's existing
        full-window Sharpe and drawdown are not simply repeated.
        """
        if self.returns.ndim != 2 or self.returns.shape[1] < 84:
            return np.zeros(self.n_stocks)
        recent = np.nan_to_num(self.returns[:, -21:], nan=0.0)
        earlier = np.nan_to_num(self.returns[:, -84:-21], nan=0.0)
        standard_error = np.sqrt(recent.var(axis=1) / 21 + earlier.var(axis=1) / 63)
        trend_change = (recent.mean(axis=1) - earlier.mean(axis=1)) / np.maximum(standard_error, 0.001)
        recent_downside = np.sqrt(np.mean(np.minimum(recent, 0.0) ** 2, axis=1))
        earlier_downside = np.sqrt(np.mean(np.minimum(earlier, 0.0) ** 2, axis=1))
        normalizer = np.maximum(np.std(self.returns[:, -84:], axis=1), 0.005)
        downside_change = (recent_downside - earlier_downside) / normalizer
        return np.tanh(0.45 * trend_change - 0.25 * downside_change)

    @property
    def n_stocks(self) -> int:
        return len(self.stock_codes)

