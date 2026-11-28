"""Retain screened-out holdings using as-of prices, without granting BUY rights."""
from dataclasses import replace
import numpy as np
import pandas as pd

from .market_data import MarketData
from .profile_policy import ProfileFeatures


class HeldHistoryUnavailableError(ValueError):
    def __init__(self, ticker, message):
        super().__init__(message)
        self.ticker = ticker


def extend_with_holdings(data,old_lots,history,sectors=None,*,allow_untraded_signal=False):
    """history: raw Close indexed by sessions, columns = ticker, <= price_date.

    No price before the start of the return window means insufficient history.
    In delayed-execution mode, known trade marks can carry through suspensions.
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
            raise HeldHistoryUnavailableError(c, f'Riwayat holdings {c} tidak tersedia; penjualan tidak dipaksakan.')
        raw=history[c].reindex(history.index.union(days)).sort_index()
        raw=raw.where(pd.to_numeric(raw,errors='coerce')>0)
        seen=pd.Series(raw.index,index=raw.index).where(raw.notna()).ffill()
        filled = raw.ffill()
        if not allow_untraded_signal:
            filled = filled.where(raw.index.to_series()-seen<=pd.Timedelta(days=7))
        filled = filled.reindex(days)
        if filled.isna().any() or not np.isfinite(filled).all():
            raise HeldHistoryUnavailableError(c, f'Riwayat holdings {c} tidak lengkap/stale; tidak dianggap aman.')
        # Do not forward-fill a decision-session execution mark.
        if not allow_untraded_signal and (pd.Timestamp(data.price_date) not in history.index or pd.isna(history.at[pd.Timestamp(data.price_date),c])):
            raise HeldHistoryUnavailableError(c, f'Harga sinyal holdings {c} tidak tersedia.')
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
        eps_by_ticker=data.eps_by_ticker,
        buy_eligible=np.r_[np.ones(data.n_stocks,dtype=bool),np.zeros(len(outside),dtype=bool)],
        dividend_yields=(np.r_[data.dividend_yields,np.zeros(len(outside))] if data.dividend_yields is not None else None))


def _held_signal_evidence(rows, signal_date, ticker, allow_untraded_signal):
    """Distinguish a historical valuation mark from a tradable signal session."""
    signal_date = pd.Timestamp(signal_date)
    signal = rows.loc[rows.Date == signal_date]
    closes = pd.to_numeric(signal.Close, errors='coerce')
    volumes = pd.to_numeric(signal.Volume, errors='coerce')
    valid = closes.gt(0) & np.isfinite(closes)
    if not valid.any() and not allow_untraded_signal:
        raise HeldHistoryUnavailableError(ticker, f'Harga sinyal holdings {ticker} tidak tersedia pada {signal_date.date()}.')
    if (volumes[valid] > 0).any():
        return {'signal_traded': True, 'signal_last_trade_date': signal_date.date().isoformat()}
    if not allow_untraded_signal:
        raise HeldHistoryUnavailableError(ticker, f'Harga sinyal holdings {ticker} tidak tersedia dari transaksi pada {signal_date.date()}.')
    traded = rows.loc[(rows.Date <= signal_date) &
                      (pd.to_numeric(rows.Volume, errors='coerce') > 0) &
                      (pd.to_numeric(rows.Close, errors='coerce') > 0)].sort_values('Date')
    if traded.empty:
        raise HeldHistoryUnavailableError(
            ticker, f'Harga sinyal holdings {ticker} stale pada {signal_date.date()}; '
                    'tidak ada transaksi historis untuk penilaian.')
    return {'signal_traded': False,
            'signal_last_trade_date': traded.iloc[-1].Date.date().isoformat(),
            'valuation_age_days': (signal_date - traded.iloc[-1].Date).days}


def load_held_universe(data,old_lots,data_dir,sector_map=None,*,allow_untraded_signal=False):
    outside=sorted(set(old_lots)-set(data.stock_codes))
    if not outside:return extend_with_holdings(data,old_lots,pd.DataFrame())
    if data.return_dates is None:raise ValueError('Cache MarketData tidak memuat tanggal return; muat ulang.')
    bars=pd.read_parquet(data_dir/'Master_OHLCV_15Tahun.parquet',
        filters=[('Ticker','in',[c+'.JK' for c in outside]),
                 ('Date','<=',pd.Timestamp(data.price_date))],columns=['Date','Ticker','Close','Volume'])
    bars['Date']=pd.to_datetime(bars.Date)
    bars = bars.loc[bars.Date.le(pd.Timestamp(data.price_date))].copy()
    bars['Ticker']=bars.Ticker.str.removesuffix('.JK')
    # In delayed-execution mode carry past trades for valuation, independently
    # of today's volume. Future prices must never enter the signal history.
    notes = dict(data.held_history_diagnostics or {})
    for c in outside:
        rows = bars.loc[bars.Ticker == c].sort_values('Date')
        signal_evidence = _held_signal_evidence(
            rows, data.price_date, c, allow_untraded_signal)
        window = rows.loc[rows.Date.isin(pd.DatetimeIndex(data.return_dates))]
        notes[c] = dict(valuation_source='last_traded_close' if allow_untraded_signal else 'reported_close',
            zero_volume_sessions=int((pd.to_numeric(window.Volume,errors='coerce') == 0).sum()),
            missing_volume_sessions=int(pd.to_numeric(window.Volume,errors='coerce').isna().sum()),
            execution_liquidity_assumed=False, **signal_evidence)
    if allow_untraded_signal:
        bars = bars.loc[pd.to_numeric(bars.Volume, errors='coerce').gt(0)
                        & pd.to_numeric(bars.Close, errors='coerce').gt(0)
                        & np.isfinite(pd.to_numeric(bars.Close, errors='coerce'))]
    history=bars.pivot_table(index='Date',columns='Ticker',values='Close',aggfunc='last')
    extended = extend_with_holdings(data,old_lots,history,sector_map,
                                    allow_untraded_signal=allow_untraded_signal)
    return replace(extended, held_history_diagnostics=notes)
