"""Controlled cash-allocation experiment on one frozen, recorded GA basket path.

No GA rerun, orders, DB writes, or integer-lot claims. Both arms use the same
stock-basket price returns, dividend entitlements, composition changes and fees.
Only the target equity fraction differs. Recommendations precede execution.
"""
import math

from src.backend.services.risk_evaluation import evaluate_risk
from src.backend.services.portfolio_return_metrics import calculate_return_metrics


class AllocationReplayUnavailable(ValueError):
    pass


def rebalance_fraction(equity, cash, old_weights, new_weights, fraction, buy_fee, sell_fee):
    """Solve the target fraction AFTER fees, accounting for composition turnover."""
    if (not 0 <= fraction <= 1 or not 0 <= buy_fee < 1 or not 0 <= sell_fee < 1
            or min(equity, cash) < 0 or not all(math.isfinite(v) for v in (equity,cash,fraction,buy_fee,sell_fee))):
        raise AllocationReplayUnavailable("Input alokasi atau biaya tidak valid.")
    for weights in (old_weights, new_weights):
        if weights and (any(not math.isfinite(v) or v < 0 for v in weights.values())
                        or abs(sum(weights.values()) - 1) > 1e-8):
            raise AllocationReplayUnavailable("Proporsi keranjang saham tidak valid.")
    if not new_weights or (equity > 0 and not old_weights):
        raise AllocationReplayUnavailable("Keranjang saham untuk alokasi belum tersedia.")
    codes = set(old_weights) | set(new_weights)
    investable = equity + cash
    def cost(target):
        delta = [target*new_weights.get(c,0) - equity*old_weights.get(c,0) for c in codes]
        return sum(max(v,0)*buy_fee + max(-v,0)*sell_fee for v in delta)
    left, right = 0., investable
    for _ in range(64):
        middle = (left + right)/2
        if middle > fraction*(investable-cost(middle)):
            right = middle
        else:
            left = middle
    target = (left+right)/2
    fee = cost(target)
    remaining = investable-target-fee
    tolerance = max(1e-7, investable*1e-12)
    if remaining < -tolerance:
        raise AllocationReplayUnavailable("Biaya membuat kas simulasi negatif.")
    return target, max(0.,remaining), fee


def run_allocation_comparison(points, decisions, initial_capital, profile, initial_receivable=0.):
    if (len(points) < 2 or not all(math.isfinite(v) for v in (initial_capital,initial_receivable))
            or initial_capital <= 0 or not 0 <= initial_receivable <= initial_capital):
        raise AllocationReplayUnavailable("Perbandingan membutuhkan modal dan minimal dua sesi.")
    by_date = {event['date']:event for event in decisions}
    if points[0]['date'] not in by_date:
        raise AllocationReplayUnavailable("Keputusan alokasi awal tidak tersedia.")
    if len(by_date) != len(decisions) or [p['date'] for p in points] != sorted({p['date'] for p in points}):
        raise AllocationReplayUnavailable("Tanggal replay harus unik dan berurutan.")
    for event in decisions:
        if event['signal_date'] >= event['date']:
            raise AllocationReplayUnavailable("Sinyal harus berasal dari sesi sebelum eksekusi.")
        if event['status'] not in ('normal','bear') or not 0 <= event['fraction'] <= 1:
            raise AllocationReplayUnavailable("Saran alokasi belum dapat ditentukan.")
    summaries, paths = {}, {}
    for name in ('recommended','full'):
        equity, cash, receivable = 0., float(initial_capital-initial_receivable), float(initial_receivable)
        growth, prior_nav, total_cost = 1., initial_capital, 0.
        rows, costs, exposures = [], [], []
        for i, point in enumerate(points):
            flow = float(point.get('contribution',0)) if i else 0.
            if i:
                price_return, dividend_yield = point['price_return'], point['dividend_yield']
                if (not all(math.isfinite(v) for v in (price_return,dividend_yield,flow))
                        or price_return < -1 or dividend_yield < 0 or flow < 0):
                    raise AllocationReplayUnavailable("Return keranjang/dividen/setoran tidak valid.")
                receivable += equity*dividend_yield
                equity *= 1+price_return
                cash += flow
            if point['date'] in by_date:
                event = by_date[point['date']]
                fraction = event['fraction'] if name == 'recommended' else 1.
                equity,cash,fee = rebalance_fraction(equity,cash,event['old_weights'] if i else {},
                    event['new_weights'],fraction,event['buy_fee'],event['sell_fee'])
                total_cost += fee
                costs.append(dict(execution_date=point['date'],transaction_costs=fee))
            nav = equity+cash+receivable
            if prior_nav <= 0:
                raise AllocationReplayUnavailable("NAV simulasi sebelumnya tidak positif.")
            daily_return = (nav-flow)/prior_nav-1
            growth *= 1+daily_return
            exposures.append(equity/nav if nav else 0.)
            rows.append(dict(date=point['date'],portfolio_value=nav,cash_balance=cash,
                holdings_market_value=equity,portfolio_value_excluding_dividends=equity+cash,
                dividend_receivable=receivable,capital_contribution=flow,daily_return=daily_return,
                portfolio_return=growth-1,is_market_session=True,
                previous_market_session_date=points[i-1]['date'] if i else None,
                stale_price_tickers=point.get('stale_price_tickers') or [],
                unreconciled_split_tickers=point.get('unreconciled_split_tickers') or [],
                ihsg_return=point.get('ihsg_return'),ihsg_observed=point.get('ihsg_observed',False),
                ihsg_previous_observed_date=point.get('ihsg_previous_observed_date')))
            prior_nav = nav
        risk = evaluate_risk(dict(series=rows,rebalance_events=costs,accounting_status='verified'),profile)
        returns = calculate_return_metrics(initial_capital=initial_capital,final_nav=rows[-1]['portfolio_value'],
            start_date=rows[0]['date'],end_date=rows[-1]['date'],twr_return_pct=(growth-1)*100,
            contributions=[(r['date'],r['capital_contribution']) for r in rows])
        summaries[name] = dict(risk=risk,net_return_pct=(growth-1)*100,final_nav=rows[-1]['portfolio_value'],
            transaction_costs=total_cost,average_equity_pct=sum(exposures)/len(exposures)*100,
            return_metrics=returns)
        paths[name] = rows
    def reduction(key):
        a,b = summaries['recommended']['risk'], summaries['full']['risk']
        a,b = (a['drawdown']['mdd_pct'],b['drawdown']['mdd_pct']) if key == 'mdd' else (a[key],b[key])
        return b-a if a is not None and b is not None else None
    deltas = dict(mdd_reduction_pp=reduction('mdd'),
        volatility_reduction_pp=reduction('annualized_volatility_pct'),
        cvar_reduction_pp=reduction('daily_cvar_95_pct'),
        return_difference_pp=summaries['recommended']['net_return_pct']-summaries['full']['net_return_pct'])
    observed = [v for key,v in deltas.items() if key != 'return_difference_pp' and v is not None]
    bear_count = sum(d['status']=='bear' and d['fraction'] < 1 for d in decisions)
    verdict = ('no_bear_reduction' if not bear_count else
               'recommended_lower_risk' if observed and all(v >= -1e-8 for v in observed) and any(v > 1e-8 for v in observed) else
               'full_lower_risk' if observed and all(v <= 1e-8 for v in observed) and any(v < -1e-8 for v in observed) else 'mixed_or_equal')
    series = [dict(date=p['date'],recommended_return_pct=paths['recommended'][i]['portfolio_return']*100,
                   full_return_pct=paths['full'][i]['portfolio_return']*100)
              for i,p in enumerate(points)]
    return dict(status='available',method='controlled_same_ga_basket_allocation_replay',
        start_date=points[0]['date'],end_date=points[-1]['date'],initial_capital=initial_capital,
        contributions=sum(p.get('contribution',0) for p in points[1:]),
        recommended=summaries['recommended'],full=summaries['full'],differences=deltas,
        verdict=verdict,bear_decisions=bear_count,decisions=decisions,series=series,
        assumptions=[
            'Kedua skenario memakai emiten dan proporsi relatif keranjang GA yang sama; GA tidak dijalankan ulang.',
            'Saran modal memakai data sampai sesi sebelum eksekusi dan diterapkan pada tanggal pembentukan/rebalance tersimpan.',
            'Simulasi memakai unit saham kontinu; pembulatan lot, likuiditas dan market impact belum dimodelkan.',
            'Kas tidak berbunga; dividen tetap piutang yang tidak dibelanjakan atau direinvestasikan tanpa tanggal pembayaran.',
            'Modal awal, setoran, jadwal, dan tarif biaya identik. Biaya nominal dapat berbeda karena volume transaksi berbeda.',
            'MDD memakai NAV pertama sesudah biaya awal sebagai basis, sama dengan evaluator portofolio. Net return tetap memasukkan biaya awal.',
            'Hasil mengisolasi efek kebijakan alokasi saham/kas pada simulasi, bukan membuktikan timing detector optimal atau hasil dua run GA berbeda.',
        ])
