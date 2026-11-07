"""As-of inputs for composition_v1; historical sector labels remain a caveat."""
import numpy as np
import pandas as pd

from .profile_policy import ProfileConstraintError, ProfileFeatures


def build_profile_features(data, closes, liquidity, data_dir, sectors):
    codes = data.stock_codes
    days = pd.DatetimeIndex(closes.sort_index().index)
    asof = data.price_date
    context = pd.read_parquet(data_dir/'Master_Market_Context_15Tahun.parquet',
        filters=[('Ticker','==','^JKSE'),('Date','>=',pd.Timestamp(days[0])-pd.Timedelta(days=14)),
                 ('Date','<=',pd.Timestamp(asof))],columns=['Date','Close'])
    marks = context.assign(Date=pd.to_datetime(context.Date)).groupby('Date').Close.last().sort_index()
    if marks.empty:
        raise ProfileConstraintError('Riwayat IHSG untuk profil kosong.')
    calendar = marks.index.union(days).sort_values()
    aligned = marks.reindex(calendar).ffill().reindex(days)
    seen = pd.Series(marks.index,index=marks.index).reindex(calendar).ffill().reindex(days)
    if aligned.isna().any() or ((days.to_series()-seen)>pd.Timedelta(days=7)).any():
        raise ProfileConstraintError('IHSG tidak lengkap/stale pada window saham.')
    benchmark = aligned.pct_change(fill_method=None).fillna(0).to_numpy()

    events = pd.read_excel(data_dir/'dividend_events_20_tahun.xlsx')
    events['Ex_Dividend_Date'] = pd.to_datetime(events.Ex_Dividend_Date,errors='coerce')
    amounts = pd.to_numeric(events.Dividend_Per_Saham,errors='coerce')
    paid_each_window = np.ones(len(codes),dtype=bool)
    end = pd.Timestamp(asof)
    # Three completed trailing 12-month windows; no future payment is visible.
    for years in range(3):
        high, low = end-pd.DateOffset(years=years), end-pd.DateOffset(years=years+1)
        paid = set(events.loc[(events.Ex_Dividend_Date>low)&(events.Ex_Dividend_Date<=high)&(amounts>0),'Ticker'])
        paid_each_window &= np.array([c in paid for c in codes])
    roe = data.fundamental_metrics[:,2]
    vols = data.returns.std(axis=1)*np.sqrt(252)
    liq = liquidity.reindex(codes)
    # Existing preprocessing already screens age, size, liquidity, EPS and ROE.
    # Additional proxy: positive current ROE, three payout windows, upper half
    # of candidate liquidity, and volatility in the lower half of candidates.
    # Individual-stock volatility is not comparable to an already diversified
    # index ceiling; the index-relative constraint is applied to the basket.
    core = (paid_each_window & np.isfinite(roe) & (roe>0)
            & np.asarray(liq.ge(liq.median())) & (vols<=np.median(vols)))
    return ProfileFeatures(codes,[sectors.get(c) for c in codes],core,benchmark,str(asof))
