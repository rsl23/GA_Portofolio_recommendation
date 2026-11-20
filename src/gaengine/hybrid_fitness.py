"""Legacy score plus a historical, after-cost CVaR loss penalty."""
import numpy as np
from .decision_utility import candidate_utility, prepare_candidate_account
from .fitness_context import current_fitness_inputs


def hybrid_cvar(config, lots):
    data=config.data; horizon=config.forecast_horizon_days
    prepared = current_fitness_inputs(config)
    cache=getattr(data,'_hybrid_blocks',None)
    if cache is None or cache[0]!=horizon:
        history=np.asarray(data.returns,dtype=float)
        n=history.shape[1]//horizon
        if n<3 or not np.isfinite(history).all() or np.any(history<=-1):
            raise ValueError('Hybrid CVaR memerlukan minimal tiga horizon return valid.')
        blocks=history[:,-n*horizon:].reshape(len(history),n,horizon)
        cache=(horizon,np.expm1(np.log1p(blocks).sum(axis=2)).T)
        data._hybrid_blocks=cache
    diagnostics=dict(policy='legacy_cvar_v1',horizon_days=horizon,
        scenario_method='completed_nonoverlapping_blocks',as_of=str(data.price_date),
        forecast_status='historical_scenarios_not_validated_forecast')
    old_prices = (prepared.hybrid_old_prices if prepared is not None else
                  {**(config.old_prices_per_lot or {}), **dict(zip(data.stock_codes,data.prices_per_lot))})
    old_cash = float(config.old_cash if config.old_cash is not None else config.budget)
    if prepared is not None and prepared.hybrid_account is None:
        prepared.hybrid_account = prepare_candidate_account(data.stock_codes, data.prices_per_lot,
            config.old_lots or {}, old_prices, old_cash)
    _,cash,fees=candidate_utility(lots,data.stock_codes,data.prices_per_lot,
        config.old_lots or {},old_prices,old_cash,
        cache[1],0.,config.buy_fee_rate,config.sell_fee_rate,
        prepared.cvar_weight if prepared is not None else config.lambda_cvar*config.profile.lambda_mdd,
        0.,0.,config.cvar_tail_fraction,
        len(cache[1]),diagnostics=diagnostics,
        prepared_account=prepared.hybrid_account if prepared is not None else None)
    return diagnostics['cvar_penalty'],cash,fees,diagnostics
