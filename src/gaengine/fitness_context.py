"""Prepare immutable scoring inputs once per evolution, never across runs."""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

PREPARE_INPUTS = True
_current = ContextVar("fitness_evaluation_inputs", default=None)


class FitnessInputs:
    def __init__(self, config):
        self.config = config
        data = config.data
        prices = data.prices_per_lot
        self.lambda_mdd = config.lambda_mdd
        self.dividend_bonus_weight = config.dividend_bonus_weight
        self.cvar_weight = config.lambda_cvar * config.profile.lambda_mdd
        self.execution_prices = {c: p / 100. for c, p in (config.old_prices_per_lot or {}).items()}
        self.execution_prices.update(dict(zip(data.stock_codes, (prices / 100.).tolist())))
        self.candidate_prices = dict(zip(data.stock_codes, prices))
        self.hybrid_old_prices = {**(config.old_prices_per_lot or {}), **self.candidate_prices}
        old = config.old_lots or {}
        self.old_alloc = np.asarray([old.get(code, 0) * prices[i]
                                    for i, code in enumerate(data.stock_codes)], dtype=float)
        # Preserve the original summation order, including held-only tickers.
        self.turnover_codes = set(dict.fromkeys(data.stock_codes)) | set(old)
        yields = np.asarray(data.dividend_yields if data.dividend_yields is not None
                            else data.fundamental_metrics[:, 4], dtype=float)
        self.dividend_signals = np.where(np.isfinite(yields),
            np.clip(yields, 0, config.dividend_yield_cap), 0) / config.dividend_yield_cap
        self.hybrid_account = None


@contextmanager
def fitness_inputs_scope(config):
    prepared = FitnessInputs(config) if PREPARE_INPUTS else None
    token = _current.set(prepared)
    try:
        yield prepared
    finally:
        _current.reset(token)


def current_fitness_inputs(config):
    prepared = _current.get()
    return prepared if prepared is not None and prepared.config is config else None
