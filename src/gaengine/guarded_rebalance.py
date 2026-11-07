"""Opt-in integer-lot rebalance with exact HOLD and net scenario utility.

This is a research policy, not a validated return forecaster. Scenarios use
completed, non-overlapping historical horizons and shrink their mean to zero.
Acquisition prices never enter trading decisions. No future prices enter search.
"""
from dataclasses import dataclass
import math
import numpy as np

from .chromosome import Chromosome
from .decision_utility import candidate_utility
from .benchmark_fitness import benchmark_utility, fitness_scenarios
from .profile_policy import composition_metrics, prepare_policy
from .rebalancer import fit_buys_to_available_cash


def block_scenarios(returns, horizon, shrinkage):
    history = np.asarray(returns, dtype=float)
    if (history.ndim != 2 or horizon < 1 or not 0 <= shrinkage <= 1
            or not np.isfinite(history).all() or np.any(history <= -1)):
        raise ValueError('Return/horizon/shrinkage tidak valid.')
    n = history.shape[1] // horizon
    if n < 3:
        raise ValueError('guarded_v1 memerlukan minimal tiga horizon historis lengkap.')
    # Align blocks backwards from the decision session; omit incomplete prefix.
    blocks = history[:, -n*horizon:].reshape(history.shape[0], n, horizon)
    log_samples = np.log1p(blocks).sum(axis=2).T
    # Shrink in log space: subtracting an arithmetic mean can create returns
    # below -100% after a large historical winner. Gross prices stay positive.
    samples = np.expm1(log_samples-shrinkage*log_samples.mean(axis=0,keepdims=True))
    if np.any(samples <= -1):
        raise ValueError('Skenario setelah shrinkage tidak valid.')
    return samples


def transaction_account(old_lots, new_lots, prices_per_share, cash, buy_fee, sell_fee):
    """Single source of truth for cash and costs, including full sells."""
    if (not math.isfinite(cash) or cash < 0 or
            not all(math.isfinite(x) and 0 <= x <= 1 for x in (buy_fee,sell_fee))):
        raise ValueError('Kas atau tarif biaya tidak valid.')
    fees = 0.
    cash_after = float(cash)
    for c in set(old_lots) | set(new_lots):
        before, after = old_lots.get(c,0),new_lots.get(c,0)
        if any(not isinstance(q,(int,np.integer)) or q<0 for q in (before,after)):
            raise ValueError('Lot transaksi harus integer non-negatif.')
        delta = after-before
        if not delta:
            continue
        p = prices_per_share.get(c)
        if p is None or not math.isfinite(p) or p<=0:
            raise ValueError(f'Harga transaksi {c} tidak tersedia.')
        cost = abs(delta)*100*p*(buy_fee if delta>0 else sell_fee)
        fees += cost
        cash_after -= delta*100*p+cost
    if cash_after < -.01:
        raise ValueError('Kas setelah biaya transaksi negatif.')
    return max(0.,cash_after),fees


@dataclass
class Candidate:
    lots: np.ndarray
    utility: float
    cash: float
    fees: float


class GuardedRebalance:
    """GA on integer position changes with a protected, unrepaired HOLD state."""
    def __init__(self,config):
        self.config=config
        self.data=config.data
        self.codes=self.data.stock_codes
        self.prices=self.data.prices_per_lot
        self.buy=np.asarray(self.data.buy_eligible if self.data.buy_eligible is not None
                            else np.ones(len(self.codes)),dtype=bool)
        if any(not isinstance(q,(int,np.integer)) or q<0 for q in config.old_lots.values()):
            raise ValueError('Lot snapshot lama harus integer non-negatif.')
        if self.buy.shape!=(len(self.codes),):raise ValueError('Mask BUY tidak sesuai universe.')
        self.old=np.array([config.old_lots.get(c,0) for c in self.codes],dtype=int)
        if set(config.old_lots)-set(self.codes):
            raise ValueError('Universe guarded_v1 harus mencakup seluruh posisi lama.')
        if config.old_cash is None or not config.is_rebalance:
            raise ValueError('guarded_v1 memerlukan posisi dan kas rebalance.')
        nav=float(config.old_cash)+float(self.old@self.prices)
        if abs(nav-config.budget)>.05:
            raise ValueError('NAV sinyal tidak cocok dengan lot, harga dan kas lama.')
        if config.fitness_policy == 'net_cvar_ihsg_v1':
            self.scenarios, self.benchmark_scenarios = fitness_scenarios(self.data, config.forecast_horizon_days)
        else:
            self.scenarios=block_scenarios(self.data.returns,config.forecast_horizon_days,config.scenario_shrinkage)
        self.rng=np.random.default_rng(config.seed)
        self.cache={}
        self.history=[]
        prepare_policy(config)
        self.hold=self.evaluate(self.old)

    def evaluate(self,q):
        q=np.asarray(q,dtype=int)
        key=tuple(q)
        if key in self.cache:return self.cache[key]
        cfg=self.config
        if (np.any(q<0) or np.any(q[~self.buy]>self.old[~self.buy])
                or not cfg.min_stocks<=np.count_nonzero(q)<=cfg.max_stocks):
            return None
        cost=float(q@self.prices)
        if cfg.allow_cash and cost>cfg.budget*cfg.max_exposure+.01:
            return None
        metrics=composition_metrics(cfg,q)
        if metrics is not None and not metrics['compliant']:return None
        try:
            if cfg.fitness_policy == 'net_cvar_ihsg_v1':
                utility,cash,fees,_=benchmark_utility(cfg,q,(self.scenarios,self.benchmark_scenarios))
            else:
                utility,cash,fees=candidate_utility(q,self.codes,self.prices,cfg.old_lots,
                    dict(zip(self.codes,self.prices)),cfg.old_cash,self.scenarios,0.,
                    cfg.buy_fee_rate,cfg.sell_fee_rate,
                    cfg.lambda_cvar*cfg.profile.lambda_mdd,cfg.lambda_concentration,
                    cfg.lambda_uncertainty,cfg.cvar_tail_fraction,len(self.scenarios))
        except ValueError:return None
        result=Candidate(q.copy(),utility,cash,fees)
        self.cache[key]=result
        return result

    def repair(self,q):
        cfg=self.config
        q=np.maximum(np.asarray(q,dtype=int),0)
        q[~self.buy]=np.minimum(q[~self.buy],self.old[~self.buy])
        if np.count_nonzero(q)>cfg.max_stocks:
            # Retain largest positions; random mutations provide exploration.
            order=np.argsort(-(q*self.prices),kind='stable')
            q[order[cfg.max_stocks:]]=0
        # Exposure repair precedes cash fitting; fees are checked again below.
        if cfg.allow_cash and q@self.prices>cfg.budget*cfg.max_exposure:
            q=np.floor(q*(cfg.budget*cfg.max_exposure)/max(q@self.prices,1)).astype(int)
        try:
            desired={c:int(v) for c,v in zip(self.codes,q) if v>0}
            fitted,_=fit_buys_to_available_cash(cfg.old_lots,desired,cfg.old_cash,
                dict(zip(self.codes,self.prices/100)),dict(zip(self.codes,q*self.prices)),
                cfg.min_stocks,cfg.buy_fee_rate,cfg.sell_fee_rate)
            return np.array([fitted.get(c,0) for c in self.codes],dtype=int)
        except ValueError:return q

    def mutate(self,base,global_search=False):
        cfg=self.config;rng=self.rng
        if global_search:
            eligible=np.flatnonzero(self.buy | (self.old>0))
            k=min(len(eligible),int(rng.integers(cfg.min_stocks,cfg.max_stocks+1)))
            chosen=rng.choice(eligible,k,replace=False)
            w=rng.dirichlet(np.ones(k))
            q=np.zeros(len(self.codes),dtype=int)
            q[chosen]=np.floor(cfg.budget*w/self.prices[chosen]).astype(int)
        else:
            q=base.copy()
            # ADD/REDUCE/small swaps around existing lots; no compulsory reset.
            count=int(rng.integers(1,min(4,len(q))+1))
            for i in rng.choice(len(q),count,replace=False):
                step=max(1,int(rng.uniform(.01,.08)*cfg.budget/self.prices[i]))
                q[i]=max(0,q[i]+int(rng.choice([-1,1]))*step)
            if rng.random()<.15 and np.count_nonzero(q)>cfg.min_stocks:
                q[int(rng.choice(np.flatnonzero(q)))]=0
        return self.repair(q)

    def run(self):
        cfg=self.config
        if cfg.population_size<2:raise ValueError('Populasi guarded_v1 minimal dua.')
        population=[self.hold] if self.hold is not None else []
        attempts=0
        while len(population)<cfg.population_size and attempts<cfg.population_size*40:
            q=self.mutate(self.old,global_search=attempts%5==0)
            item=self.evaluate(q)
            if item is not None:population.append(item)
            attempts+=1
        if not population:
            raise ValueError('GA belum menemukan posisi feasible; HOLD melanggar batas. Tidak ada batas yang dilonggarkan.')
        for _ in range(cfg.generations):
            population.sort(key=lambda c:c.utility,reverse=True)
            self.history.append(population[0].utility)
            next_population=population[:max(1,min(cfg.elitism_count,len(population)))]
            for j in range(cfg.population_size-len(next_population)):
                picks=self.rng.integers(0,len(population),size=3)
                parent=max((population[i] for i in picks),key=lambda c:c.utility)
                base=parent.lots
                if self.rng.random()<cfg.crossover_rate:
                    other=population[int(self.rng.integers(len(population)))].lots
                    base=np.where(self.rng.random(len(base))<.5,base,other)
                q=self.mutate(base,global_search=self.rng.random()<.2)
                next_population.append(self.evaluate(q) or parent)
            # HOLD survives even when ordinary elites are replaced.
            if self.hold is not None:next_population.append(self.hold)
            population=next_population
        best=max(population,key=lambda c:c.utility)
        selected_hold=self.hold is not None and best.utility<=self.hold.utility+cfg.min_utility_gain
        chosen=self.hold if selected_hold else best
        genes=chosen.lots*self.prices
        solution=Chromosome(cfg,(genes/genes.sum()).tolist())
        solution.evaluate_fitness(lots_override=chosen.lots)
        solution.fitness=chosen.utility
        diagnostic=dict(model='guarded_v1',reason='hold' if selected_hold else 'risk_repair' if self.hold is None else 'net_utility_gain',
            hold_utility=self.hold.utility if self.hold else None,best_utility=best.utility,
            selected_utility=chosen.utility,min_utility_gain=cfg.min_utility_gain,
            horizon=cfg.forecast_horizon_days,independent_blocks=len(self.scenarios),
            shrinkage_to_zero=cfg.scenario_shrinkage,forecast_status='historical_scenarios_not_validated_forecast',
            scenario_transform='log_mean_v1',
            signal_transaction_costs=chosen.fees,signal_cash_after=chosen.cash,
            hold_only_tickers=[c for c,b in zip(self.codes,self.buy) if not b],
            evaluated_candidates=len(self.cache),seed=cfg.seed)
        diagnostic['held_history'] = self.data.held_history_diagnostics or {}
        if cfg.fitness_policy == 'net_cvar_ihsg_v1':
            diagnostic.update(shrinkage_to_zero=0., scenario_transform='completed_nonoverlapping_blocks',
                fitness_policy=cfg.fitness_policy, fitness_diagnostics=solution.fitness_diagnostics)
        return solution,diagnostic
