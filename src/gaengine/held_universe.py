"""Retain screened-out holdings using as-of prices, without granting BUY rights."""
from dataclasses import replace
import numpy as np
import pandas as pd

from .market_data import MarketData
from .profile_policy import ProfileFeatures


def extend_with_holdings(data,old_lots,history,sectors=None):
    """history: raw Close indexed by sessions, columns = ticker, <= price_date.

    Missing/old marks reject the experiment instead of inventing a zero-risk
    history. Current DB sector labels are a documented historical-data caveat.
    """
    outside=sorted(set(old_lots)-set(data.stock_codes))
    if not outside:
        return replace(data,buy_eligible=np.ones(data.n_stocks,dtype=bool))
    if data.return_dates is None:
        raise ValueError('Tanggal return diperlukan untuk mempertahankan saham di luar filter.')
    days=pd.DatetimeIndex(data.return_dates)
    if days.max().date()!=data.price_date:
        raise ValueError('Tanggal riwayat holdings tidak sesuai data sinyal.')
    history=history.loc[history.index<=pd.Timestamp(data.price_date)].sort_index()
    returns=[];prices=[]
    for c in outside:
        if c not in history:
            raise ValueError(f'Riwayat holdings {c} tidak tersedia; penjualan tidak dipaksakan.')
        raw=history[c].reindex(history.index.union(days)).sort_index()
        raw=raw.where(pd.to_numeric(raw,errors='coerce')>0)
        seen=pd.Series(raw.index,index=raw.index).where(raw.notna()).ffill()
        filled=raw.ffill().where(raw.index.to_series()-seen<=pd.Timedelta(days=7)).reindex(days)
        if filled.isna().any() or not np.isfinite(filled).all():
            raise ValueError(f'Riwayat holdings {c} tidak lengkap/stale; tidak dianggap aman.')
        # Do not forward-fill a decision-session execution mark.
        if pd.Timestamp(data.price_date) not in history.index or pd.isna(history.at[pd.Timestamp(data.price_date),c]):
            raise ValueError(f'Harga sinyal holdings {c} tidak tersedia.')
        returns.append(filled.pct_change(fill_method=None).fillna(0).to_numpy())
        prices.append(float(filled.iloc[-1])*100)
    codes=data.stock_codes+outside
    R=np.vstack([data.returns,*returns])
    features=None
    if data.profile_features is not None:
        old=data.profile_features
        features=ProfileFeatures(codes,old.sectors+[(sectors or {}).get(c) for c in outside],
            np.r_[old.core_mask,np.zeros(len(outside),dtype=bool)],old.benchmark_returns,old.as_of,old.core_definition)
    # Stress archive is not the guarded utility model. Do not reuse a stress
    # matrix whose stock order/universe no longer matches.
    with np.errstate(divide='ignore',invalid='ignore'):
        correlation=np.nan_to_num(np.corrcoef(R))
    return MarketData(codes,np.r_[data.prices_per_lot,prices],R,correlation,
        np.r_[data.fundamental_scores,np.zeros(len(outside))],
        np.vstack([data.fundamental_metrics,np.full((len(outside),5),np.nan)]),data.risk_free_rate,
        price_date=data.price_date,return_dates=list(days),profile_features=features,
        benchmark_returns=data.benchmark_returns,
        held_history_diagnostics=data.held_history_diagnostics,
        buy_eligible=np.r_[np.ones(data.n_stocks,dtype=bool),np.zeros(len(outside),dtype=bool)],
        dividend_yields=(np.r_[data.dividend_yields,np.zeros(len(outside))] if data.dividend_yields is not None else None))


def load_held_universe(data,old_lots,data_dir,sector_map=None):
    outside=sorted(set(old_lots)-set(data.stock_codes))
    if not outside:return extend_with_holdings(data,old_lots,pd.DataFrame())
    if data.return_dates is None:raise ValueError('Cache MarketData tidak memuat tanggal return; muat ulang.')
    bars=pd.read_parquet(data_dir/'Master_OHLCV_15Tahun.parquet',
        filters=[('Ticker','in',[c+'.JK' for c in outside]),
                 ('Date','>=',pd.Timestamp(data.return_dates[0])-pd.Timedelta(days=14)),
                 ('Date','<=',pd.Timestamp(data.price_date))],columns=['Date','Ticker','Close','Volume'])
    bars['Date']=pd.to_datetime(bars.Date)
    bars['Ticker']=bars.Ticker.str.removesuffix('.JK')
    # Historical Close is a valuation mark, not evidence of an executable
    # trade. Match the BUY universe's return-history convention: retain a
    # provider-reported positive Close on zero-volume sessions. Removing it
    # creates artificial holes only for stocks outside today's BUY screen.
    # The decision-session mark must still be backed by an actual trade.
    notes = dict(data.held_history_diagnostics or {})
    for c in outside:
        rows = bars.loc[bars.Ticker == c].sort_values('Date')
        signal = rows.loc[rows.Date == pd.Timestamp(data.price_date)]
        if (signal.empty or not (pd.to_numeric(signal.Volume,errors='coerce') > 0).any()):
            raise ValueError(f'Harga sinyal holdings {c} tidak tersedia dari transaksi pada {data.price_date}.')
        window = rows.loc[rows.Date.isin(pd.DatetimeIndex(data.return_dates))]
        notes[c] = dict(valuation_source='reported_close',
            zero_volume_sessions=int((pd.to_numeric(window.Volume,errors='coerce') == 0).sum()),
            missing_volume_sessions=int(pd.to_numeric(window.Volume,errors='coerce').isna().sum()),
            execution_liquidity_assumed=False)
    history=bars.pivot_table(index='Date',columns='Ticker',values='Close',aggfunc='last')
    extended = extend_with_holdings(data,old_lots,history,sector_map)
    return replace(extended, held_history_diagnostics=notes)
