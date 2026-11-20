"""Read-only adapter for a paired bear-allocation policy experiment."""
from datetime import date
from pathlib import Path
import math

import pandas as pd

from src.backend.services.bear_allocation_replay import AllocationReplayUnavailable, run_allocation_comparison
from src.backend.services.risk_evaluation import PROFILE_ALIASES
from src.preprocessing.capital_allocation_policy import recommend_fraction

DATA_DIR = Path(__file__).resolve().parents[3] / 'data'


def _value(value, label):
    try:
        result = float(value)
    except (TypeError,ValueError):
        raise AllocationReplayUnavailable(f'{label} belum tersedia.') from None
    if not math.isfinite(result):
        raise AllocationReplayUnavailable(f'{label} tidak valid.')
    return result


def historical_assessments(db, decision_dates):
    """Same detector/severity as the app, with local breadth loaded once."""
    from src.preprocessing.bear_market_detection import CONFIG, _historical_context, _series, _clean, detect_market_regime
    from src.preprocessing.market_breadth import breadth_universe, calculate_breadth
    from src.backend.models.stock_universe import StockUniverse
    last = max(decision_dates)
    context = _historical_context()
    ihsg = _clean(_series(context,'^JKSE'),last)
    vix = _clean(_series(context,'^VIX'),last)
    fx = _clean(_series(context,'USDIDR=X'),last)
    listing_rows = db.query(StockUniverse.ticker,StockUniverse.listing_date,
        StockUniverse.delisting_date,StockUniverse.relisting_date).all()
    listings = pd.DataFrame(listing_rows,columns=['ticker','listing_date','delisting_date','relisting_date'])
    earliest = ihsg.loc[:str(min(decision_dates))].index[-CONFIG.slow_ma:]
    if len(earliest) < CONFIG.slow_ma:
        raise AllocationReplayUnavailable('Histori indikator bear belum cukup sebelum keputusan awal.')
    bars = pd.read_parquet(DATA_DIR/'Master_OHLCV_15Tahun.parquet',
        columns=['Date','Ticker','Close','Volume'],
        filters=[('Date','>=',earliest[0]),('Date','<=',pd.Timestamp(last))])
    bars['Date'] = pd.to_datetime(bars['Date'])
    bars = bars.set_index('Date',drop=False).sort_index()
    results = {}
    for day in sorted(decision_dates):
        window = ihsg.loc[:str(day)].index[-CONFIG.slow_ma:]
        universe = breadth_universe(listings,window)
        breadth,details = calculate_breadth(bars.loc[window[0]:window[-1]],window,universe,
            ma_sessions=CONFIG.slow_ma,min_stocks=CONFIG.breadth_min_stocks,
            min_coverage=CONFIG.breadth_min_coverage)
        assessment = detect_market_regime(ihsg,day,vix=vix,usd_idr=fx,breadth=breadth,
            breadth_details=details,config=CONFIG)
        if assessment['status'] == 'unknown':
            raise AllocationReplayUnavailable(f'Saran pada {day} belum dapat dihitung: '+assessment['reason'])
        results[day] = assessment
    return results


def prepare_allocation_replay(db, portfolio, performance, start_date=None, end_date=None,
                              risk_profile=None, fallback_buy_fee=.0015, fallback_sell_fee=.0025):
    from src.backend.controller.portfolio_controller import _portfolio_lineage, _snapshot_positions, _snapshot_date, _as_date
    from src.preprocessing.bear_market_detection import _historical_context, _series, _clean
    if performance.get('price_accounting_status',performance.get('accounting_status')) != 'verified':
        raise AllocationReplayUnavailable('Arus dana dan NAV sumber belum cukup terverifikasi untuk replay alokasi.')
    rows = [r for r in performance['series'] if r.get('is_market_session') is not False
            and (not start_date or r['date'][:10]>=str(start_date))
            and (not end_date or r['date'][:10]<=str(end_date))]
    if len(rows)<2:
        raise AllocationReplayUnavailable('Periode harus memuat minimal dua sesi valuasi.')
    profile = PROFILE_ALIASES.get(str(risk_profile or portfolio.risk_profile).strip().lower())
    if profile is None:
        raise AllocationReplayUnavailable('Profil risiko tidak dikenal.')
    policy_profile = {'Conservative':'Konservatif','Moderate':'Moderat','Aggressive':'Agresif'}[profile]
    first,last = rows[0]['date'][:10],rows[-1]['date'][:10]
    chain = _portfolio_lineage(db,portfolio,portfolio.user_id)
    snapshots = []
    for node in chain:
        execution = _as_date(node.execution_date) if node.execution_date is not None else _snapshot_date(node)
        positions = _snapshot_positions(db,node.id)
        lots = {c:v['lots'] for c,v in positions.items() if v['lots']>0}
        snapshots.append(dict(date=execution.isoformat(),lots=lots,node=node))
    active = [s for s in snapshots if s['date']<=first]
    if not active:
        raise AllocationReplayUnavailable('Komposisi GA awal belum tersedia.')
    schedule = {first:active[-1]}
    schedule.update({s['date']:s for s in snapshots if first<s['date']<=last})
    row_dates = {r['date'][:10] for r in rows}
    if set(schedule)-row_dates:
        raise AllocationReplayUnavailable('Ada tanggal rebalance tanpa valuasi sesi yang lengkap.')
    all_codes = sorted({c for s in snapshots for c in s['lots']})
    bars = pd.read_parquet(DATA_DIR/'Master_OHLCV_15Tahun.parquet',
        columns=['Date','Ticker','Close'],filters=[('Date','in',[pd.Timestamp(d) for d in schedule]),
            ('Ticker','in',[c+'.JK' for c in all_codes])])
    bars['Date'] = pd.to_datetime(bars['Date']).dt.strftime('%Y-%m-%d')
    bars['Ticker'] = bars['Ticker'].str.removesuffix('.JK')
    quotes = {(r.Date,r.Ticker):r.Close for r in bars.itertuples()}
    market = _clean(_series(_historical_context(),'^JKSE'),date.fromisoformat(last))
    decisions, previous_lots = [], {}
    for day,snapshot in sorted(schedule.items()):
        node = snapshot['node']; meta = node.rebalance_metadata or {}
        old_lots = previous_lots
        if day != first:
            earlier = [s for s in snapshots if s['date']<day]
            old_lots = earlier[-1]['lots'] if earlier else {}
        def weights(lots):
            values = {c:q*100*_value(quotes.get((day,c)),f'Harga {c} pada {day}') for c,q in lots.items()}
            total = sum(values.values())
            if lots and total<=0:
                raise AllocationReplayUnavailable('Nilai keranjang saham tidak positif.')
            return {c:v/total for c,v in values.items()}
        old_weights,new_weights = weights(old_lots),weights(snapshot['lots'])
        completed = market.loc[market.index<pd.Timestamp(day)]
        if completed.empty:
            raise AllocationReplayUnavailable('Sesi sinyal sebelum eksekusi tidak tersedia.')
        signal = completed.index[-1].date()
        stored = (meta.get('fitness_diagnostics') or {}).get('as_of') if snapshot['date']==day else None
        if stored:
            signal = date.fromisoformat(str(stored)[:10])
        if signal>=date.fromisoformat(day):
            raise AllocationReplayUnavailable('Tanggal sinyal tersimpan tidak mendahului eksekusi.')
        buy_fee = _value(meta.get('buy_fee_rate',fallback_buy_fee),'Tarif beli')
        sell_fee = _value(meta.get('sell_fee_rate',fallback_sell_fee),'Tarif jual')
        if not 0<=buy_fee<=.05 or not 0<=sell_fee<=.05:
            raise AllocationReplayUnavailable('Tarif biaya di luar batas simulasi.')
        decisions.append(dict(date=day,signal_date=signal.isoformat(),old_weights=old_weights,
            new_weights=new_weights,buy_fee=buy_fee,sell_fee=sell_fee,
            fee_source='saved' if 'buy_fee_rate' in meta and 'sell_fee_rate' in meta else 'fallback'))
        previous_lots = snapshot['lots']
    assessments = historical_assessments(db,{date.fromisoformat(d['signal_date']) for d in decisions})
    for decision in decisions:
        assessment = assessments[date.fromisoformat(decision['signal_date'])]
        decision.update(status=assessment['status'],severity=assessment['severity'],
            fraction=recommend_fraction(assessment['status'],assessment['severity'],policy_profile),
            detector_version=assessment['config_version'])
    source_fees = {}
    for event in performance.get('rebalance_events',[]):
        day = str(event['execution_date'])[:10]
        source_fees[day] = source_fees.get(day,0)+_value(event.get('transaction_costs'),'Biaya sumber')
    points = []
    for i,row in enumerate(rows):
        day = row['date'][:10]
        point = dict(date=day,price_return=0.,dividend_yield=0.,contribution=0.,
            **{k:row.get(k) for k in ('ihsg_return','ihsg_observed','ihsg_previous_observed_date',
                'stale_price_tickers','unreconciled_split_tickers')})
        if i:
            prior = rows[i-1]
            if row.get('previous_market_session_date') not in (None,prior['date'][:10]):
                raise AllocationReplayUnavailable('Ada sesi NAV hilang; return beberapa hari tidak digunakan sebagai return harian.')
            prior_equity = _value(prior.get('holdings_market_value'),'Nilai saham sebelumnya')
            if prior_equity<=0:
                raise AllocationReplayUnavailable('Keranjang sumber sedang tanpa saham; return skenario 100% tidak dapat direkonstruksi.')
            flow = _value(row.get('capital_contribution',0),'Setoran')
            pnl = (_value(row.get('portfolio_value_excluding_dividends'),'NAV harga')
                -_value(prior.get('portfolio_value_excluding_dividends'),'NAV harga sebelumnya')
                -flow+source_fees.get(day,0))
            point.update(price_return=pnl/prior_equity,
                dividend_yield=_value(row.get('dividend_income'),'Dividen harian')/prior_equity,
                contribution=flow)
        points.append(point)
    initial = (performance['initial_capital'] if first==performance['start_date']
               else _value(rows[0].get('portfolio_value'),'NAV awal periode'))
    receivable = 0. if first==performance['start_date'] else _value(rows[0].get('dividend_receivable'),'Piutang awal')
    return points,decisions,_value(initial,'Modal awal'),profile,receivable


def evaluate_bear_impact(db,portfolio,performance,**options):
    points,decisions,capital,profile,receivable = prepare_allocation_replay(db,portfolio,performance,**options)
    result = run_allocation_comparison(points,decisions,capital,profile,receivable)
    result['profile'] = profile
    return result
