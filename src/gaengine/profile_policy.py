"""Experimental, versioned equity-composition constraints (not calibrated advice).

All weights refer to the invested equity basket; idle cash cannot hide basket
risk. Features are frozen at the signal session. Never use a current blue-chip
list for past decisions. The 'core' label below is a disclosed proxy, not QMJ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np


class ProfileConstraintError(ValueError):
    """Missing evidence or no compliant solution found; never relax silently."""


@dataclass(frozen=True)
class ProfilePolicy:
    min_core_weight: float
    max_stock_weight: float
    max_sector_weight: float
    max_volatility_ratio: float


POLICIES = {
    'Conservative': ProfilePolicy(.60, .20, .35, .80),
    'Moderate': ProfilePolicy(.30, .25, .45, 1.00),
    'Aggressive': ProfilePolicy(.00, .35, .60, 1.30),
}


@dataclass
class ProfileFeatures:
    stock_codes: list[str]
    sectors: list[str]
    core_mask: np.ndarray
    benchmark_returns: np.ndarray
    as_of: str
    core_definition: str = 'core-proxy-v1'

    def prepare(self, data):
        if self.stock_codes != data.stock_codes:
            raise ProfileConstraintError('Urutan bukti profil tidak sesuai kandidat.')
        self.core_mask = np.asarray(self.core_mask, dtype=bool)
        self.benchmark_returns = np.asarray(self.benchmark_returns, dtype=float)
        if (self.core_mask.shape != (data.n_stocks,) or len(self.sectors) != data.n_stocks
                or any(not str(s).strip() or str(s).lower() in ('none', 'nan') for s in self.sectors)):
            raise ProfileConstraintError('Bukti core/sektor profil tidak lengkap.')
        if (self.benchmark_returns.shape != (data.returns.shape[1],)
                or not np.isfinite(self.benchmark_returns).all()
                or not np.isfinite(data.returns).all() or data.returns.shape[1] < 60):
            raise ProfileConstraintError('Return saham dan IHSG harus lengkap, selaras, minimal 60 sesi.')
        if str(data.price_date) != self.as_of:
            raise ProfileConstraintError('Tanggal bukti profil berbeda dari sesi harga keputusan.')
        self.benchmark_vol = float(self.benchmark_returns.std() * np.sqrt(252))
        if self.benchmark_vol <= 1e-12:
            raise ProfileConstraintError('Volatilitas IHSG tidak valid untuk batas relatif.')
        self.covariance = np.atleast_2d(np.cov(data.returns, ddof=0)) * 252
        self.stock_vols = np.sqrt(np.maximum(np.diag(self.covariance), 0))
        self.groups = [np.array([i for i,s in enumerate(self.sectors) if s == sector])
                       for sector in sorted(set(self.sectors))]


def policy_for(config):
    if config.profile_policy == 'legacy':
        return None
    return POLICIES[config.profile.value]


def prepare_policy(config):
    policy = policy_for(config)
    if policy is None:
        return
    f = config.data.profile_features
    if f is None:
        raise ProfileConstraintError('composition_v1 memerlukan bukti saham inti, sektor, dan IHSG.')
    f.prepare(config.data)
    if config.max_stocks * policy.max_stock_weight < 1 - 1e-9:
        raise ProfileConstraintError('max_stocks terlalu kecil untuk batas bobot per saham.')
    if int(f.core_mask.sum()) * policy.max_stock_weight < policy.min_core_weight - 1e-9:
        raise ProfileConstraintError('Kandidat saham inti tidak cukup untuk minimum bobot profil.')
    if len(f.groups) * policy.max_sector_weight < 1 - 1e-9:
        raise ProfileConstraintError('Sektor kandidat tidak cukup untuk batas konsentrasi profil.')


def composition_metrics(config, lots, prices_per_lot=None):
    p = policy_for(config)
    if p is None:
        return None
    f = config.data.profile_features
    prices = config.data.prices_per_lot if prices_per_lot is None else np.asarray(prices_per_lot)
    allocation = np.asarray(lots) * prices
    if not np.isfinite(allocation).all() or np.any(allocation < 0) or allocation.sum() <= 0:
        raise ProfileConstraintError('Alokasi saham tidak valid untuk validasi profil.')
    w = allocation / allocation.sum()
    vol = float(np.sqrt(max(0, w @ f.covariance @ w)))
    result = dict(version=config.profile_policy, profile=config.profile.value, as_of=f.as_of,
                  core_definition=f.core_definition, limits=asdict(p),
                  core_weight=float(w[f.core_mask].sum()), max_stock_weight=float(w.max()),
                  max_sector_weight=max(float(w[g].sum()) for g in f.groups),
                  volatility=vol, benchmark_volatility=f.benchmark_vol,
                  volatility_ratio=vol/f.benchmark_vol,
                  core_tickers=[c for c,core in zip(f.stock_codes,f.core_mask) if core])
    violations = []
    for key, threshold, lower in [('core_weight',p.min_core_weight,True),
                                  ('max_stock_weight',p.max_stock_weight,False),
                                  ('max_sector_weight',p.max_sector_weight,False),
                                  ('volatility_ratio',p.max_volatility_ratio,False)]:
        if (result[key] < threshold-1e-9 if lower else result[key] > threshold+1e-9):
            violations.append(key)
    result['violations'] = violations
    result['compliant'] = not violations
    return result


def validate_execution(config, lots, prices_per_lot):
    result = composition_metrics(config, lots, prices_per_lot)
    if result is not None and not result['compliant']:
        raise ProfileConstraintError('Komposisi pada harga eksekusi melanggar profil: '
                                     + ', '.join(result['violations']))
    return result


def _project(weights, core, groups, p):
    """Bounded heuristic projection. Final discrete validation is authoritative."""
    w = np.maximum(weights, 1e-8).copy()
    w /= w.sum()
    cap = p.max_stock_weight - .005
    sector_cap = p.max_sector_weight - .005
    minimum = min(1., p.min_core_weight + (.01 if p.min_core_weight else 0))
    for _ in range(16):
        w = np.minimum(w, cap)
        for g in groups:
            if w[g].sum() > sector_cap:
                w[g] *= sector_cap/w[g].sum()
        deficit = max(0., minimum-w[core].sum())
        if deficit > 0 and core.any():
            room = np.where(core, np.maximum(cap-w,0),0)
            if room.sum() > 0:
                add = min(deficit,room.sum(),w[~core].sum())
                w[~core] *= max(0.,1-add/max(w[~core].sum(),1e-12))
                w += add*room/room.sum()
        gap = 1-w.sum()
        if gap > 1e-10:
            room = np.maximum(cap-w,0)
            for g in groups:
                if room[g].sum() > 0:
                    room[g] *= min(1.,max(0.,sector_cap-w[g].sum())/room[g].sum())
            if room.sum() > 0:
                w += min(gap,room.sum()) * room/room.sum()
        if w.sum() > 1+1e-10:
            w /= w.sum()
        if (abs(w.sum()-1)<1e-9 and w.max()<=cap+1e-9
                and w[core].sum()>=minimum-1e-9
                and all(w[g].sum()<=sector_cap+1e-9 for g in groups)):
            break
    return w / w.sum()


def repair_weights(config, genes):
    """Retain strongest genes, reserve core slots, then reduce estimated risk.

    This is heuristic constraint handling, not a global feasibility solver.
    Failure to find a solution is never reported as proof of infeasibility.
    """
    p = policy_for(config)
    if p is None:
        return np.asarray(genes,dtype=float)
    f = config.data.profile_features
    raw = np.maximum(np.asarray(genes,dtype=float),0)
    # Up to ten holdings avoid tiny token positions and improve lot rounding.
    k = min(config.max_stocks, config.data.n_stocks)
    core_slots = (min(k,math.ceil(p.min_core_weight/(p.max_stock_weight-.005)))
                  if p.min_core_weight else 0)
    core_indices = np.flatnonzero(f.core_mask)
    selected = list(sorted(core_indices,key=lambda i:raw[i],reverse=True)[:core_slots])
    for i in np.argsort(-raw, kind='stable'):
        if int(i) not in selected:
            selected.append(int(i))
        if len(selected) >= k:
            break
    idx = np.array(selected,dtype=int)
    core = f.core_mask[idx]
    groups = [np.flatnonzero(np.array(f.sectors)[idx] == s) for s in sorted(set(np.array(f.sectors)[idx]))]
    w = _project(raw[idx],core,groups,p)
    covariance = f.covariance[np.ix_(idx,idx)]
    bound = (p.max_volatility_ratio*f.benchmark_vol*.99)**2
    # Projected descent only when over the risk ceiling; stop at a safe margin.
    step = .5/max(float(np.linalg.norm(covariance,ord=np.inf)),1e-9)
    for _ in range(16):
        if w @ covariance @ w <= bound:
            break
        w = _project(np.maximum(w-step*(covariance@w),1e-8),core,groups,p)
    result = np.zeros(config.data.n_stocks)
    result[idx] = w
    return result
