"""Immutable inputs for scoring actual fills with the original decision data."""
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
import math
import numpy as np


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def freeze_signal(config, solution):
    return {"config": json_value({field.name: getattr(config, field.name)
                                  for field in fields(config) if field.name != "data"}),
            "data": json_value(config.data), "genes": json_value(solution.genes),
            "exposure": float(solution.exposure)}


def score_actual_lots(snapshot, target):
    from src.gaengine.market_data import MarketData
    from src.gaengine.market_stress import MarketStressData
    from src.gaengine.ga_config import GAConfig
    from src.gaengine.profile_policy import ProfileFeatures
    values = dict(snapshot["data"])
    values["price_date"] = date.fromisoformat(values["price_date"])
    if values.get("market_stress"):
        stress = dict(values["market_stress"])
        stress["as_of"] = date.fromisoformat(stress["as_of"])
        if stress.get("history_end"):
            stress["history_end"] = date.fromisoformat(stress["history_end"])
        values["market_stress"] = MarketStressData(**stress)
    if values.get("profile_features"):
        values["profile_features"] = ProfileFeatures(**values["profile_features"])
    data = MarketData(**values)
    config = GAConfig(**snapshot["config"], data=data)
    solution = score_execution_lots(config, snapshot["genes"], snapshot["exposure"], target)
    if not solution.budget_ok or not solution.diversification_ok:
        raise ValueError("Lot aktual tidak memenuhi batas kelayakan pada data sinyal; periksa transaksi atau buat rekomendasi baru.")
    return config, solution


def score_execution_lots(config, genes, exposure, target):
    """Shared live/backtest reporting score; never reselect HOLD after fills."""
    from src.gaengine.profile_policy import prepare_policy
    from src.gaengine.chromosome import Chromosome
    prepare_policy(config)
    solution = Chromosome(config, genes)
    solution.exposure = exposure
    solution.evaluate_fitness(lots_override=np.asarray([target.get(code, 0) for code in config.data.stock_codes]))
    if config.rebalance_model == "guarded_v1":
        from src.gaengine.guarded_rebalance import GuardedRebalance
        candidate = GuardedRebalance(config).evaluate(solution.lots)
        if candidate is None:
            raise ValueError("Lot eksekusi tidak memenuhi batas guarded_v1 pada data sinyal.")
        solution.fitness = candidate.utility
    return solution
