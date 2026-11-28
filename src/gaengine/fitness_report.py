"""Small, JSON-safe snapshot of already computed GA metrics (no new queries)."""
import math


def fitness_report(solution, config):
    # Execution fitting may evaluate a copied config with different lot prices.
    config = getattr(solution, "config", config)
    def number(value):
        return float(value) if value is not None and math.isfinite(float(value)) else None

    def metric(name):
        return number(getattr(solution, name, None))

    report = {name: metric(name) for name in (
        "score_sharpe", "mdd_risk_score", "risk_component", "market_stress_risk",
        "market_stress_coverage", "market_stress_missing_penalty", "momentum_score",
        "dividend_score", "dividend_bonus", "rebalance_utility", "turnover_rate", "roi_annual",
    )}
    report.update(
        fitness_mode=config.fitness_mode, is_rebalance=config.is_rebalance,
        lookthrough_stock_risk=config.lookthrough_stock_risk and config.allow_cash
            and config.fitness_mode != "baseline",
        mdd_residual_weight=number(config.mdd_residual_weight),
        coefficients={name: number(value) for name, value in {
            "risk": config.lambda_mdd, "correlation": config.correlation_penalty,
            "fundamental": config.fundamental_bonus, "momentum": config.momentum_bonus,
            "dividend": config.dividend_bonus_weight, "turnover": config.turnover_penalty_beta,
            "rebalance_utility": config.rebalance_utility_weight,
            "return_nav": config.cash_return_weight if config.allow_cash and config.fitness_mode != "baseline" else 0,
        }.items()},
    )
    # Signed contributions are meaningful for the additive fitness only.
    if config.fitness_mode != "decision_aware" and not (
        config.rebalance_model == "guarded_v1" and config.fitness_policy == "legacy"
    ):
        terms = {"sharpe": report["score_sharpe"]}
        for key, raw, coefficient in (
            ("risk", report["risk_component"], -config.lambda_mdd),
            ("correlation", metric("avg_correlation"), -config.correlation_penalty),
            ("fundamental", metric("skor_fundamental"), config.fundamental_bonus),
            ("momentum", report["momentum_score"], config.momentum_bonus),
            ("turnover", report["turnover_rate"], -config.turnover_penalty_beta),
            ("rebalance_utility", report["rebalance_utility"], config.rebalance_utility_weight),
            ("return_nav", report["roi_annual"], report["coefficients"]["return_nav"]),
        ):
            terms[key] = number(raw * coefficient) if raw is not None else None
        terms["dividend"] = report["dividend_bonus"]
        missing = report["market_stress_missing_penalty"]
        terms["stress_missing"] = -missing if missing is not None else None
        diagnostics = getattr(solution, "fitness_diagnostics", None) or {}
        terms["cvar"] = number(-diagnostics.get("cvar_penalty", 0))
        report["coefficients"]["cvar"] = (
            number(config.lambda_cvar * config.profile.lambda_mdd)
            if config.fitness_policy == "legacy_cvar_v1" else 0.0
        )
        report["contributions"] = terms

    profile = getattr(solution, "profile_metrics", None)
    if profile and hasattr(solution, "upside_signal"):
        from .composition_v2_experiment import ADAPTIVE_VOLATILITY_PENALTY, uses_adaptive_conservative

        report["upside_signal"] = metric("upside_signal")
        report["volatility_ratio"] = number(profile["volatility_ratio"])
        report["stock_volatility"] = number(profile["volatility"])
        report["benchmark_volatility"] = number(profile["benchmark_volatility"])
        upside_weight = profile["limits"]["upside_coefficient"]
        volatility_weight = ADAPTIVE_VOLATILITY_PENALTY if uses_adaptive_conservative(config) else 0.0
        report["coefficients"].update(upside=number(upside_weight), volatility=volatility_weight)
        if "contributions" in report:
            report["contributions"].update(
                upside=number(solution.upside_signal * upside_weight),
                volatility=number(-volatility_weight * profile["volatility_ratio"]),
            )

    data = config.data
    rows = []
    values = solution.lots * data.prices_per_lot
    total = float(values.sum())
    for i, code in enumerate(data.stock_codes):
        if solution.lots[i] <= 0:
            continue
        per, pbv, roe, der, dividend_yield = data.fundamental_metrics[i]
        rows.append(dict(ticker=code, weight=number(values[i] / total) if total > 0 else None,
                         score=number(data.fundamental_scores[i]), per=number(per), pbv=number(pbv),
                         roe=number(roe), der=number(der), dividend_yield=number(dividend_yield)))
    report["fundamentals"] = rows
    return report
