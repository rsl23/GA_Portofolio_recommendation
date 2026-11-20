"""Owner corrections to the latest live position snapshot, without new cash flows."""
import math
from datetime import datetime, timezone
from uuid import UUID

from src.backend.models.portofolios import Portofolio
from src.backend.models.portofolio_items import PortofolioItem
from src.backend.controller import portfolio_controller as pc
from src.gaengine.guarded_rebalance import transaction_account


def update_live_holding(db, user_id, portfolio_id, ticker, body):
    try:
        owner, identifier = UUID(user_id), UUID(portfolio_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise pc.ItemNotFoundError("Portofolio tidak ditemukan.") from exc
    portfolio = db.query(Portofolio).filter(
        Portofolio.id == identifier, Portofolio.user_id == owner,
    ).with_for_update().first()
    if portfolio is None:
        raise pc.ItemNotFoundError("Portofolio tidak ditemukan.")
    if portfolio.status_portofolio != pc.STATUS_ACTIVE:
        raise ValueError("Hanya posisi portofolio live terbaru yang dapat diubah.")
    if db.query(Portofolio.id).filter(Portofolio.parent_portofolio_id == identifier).first():
        raise ValueError("Portofolio sudah memiliki rebalance berikutnya; muat ulang posisi terbaru.")
    pairs = pc._query_items_with_ticker(db, identifier)
    current_items = {code: item for item, code in pairs}
    parent_items = {}
    if portfolio.parent_portofolio_id:
        parent = db.query(Portofolio).filter(
            Portofolio.id == portfolio.parent_portofolio_id, Portofolio.user_id == owner,
        ).first()
        if parent is None:
            raise ValueError("Riwayat induk portofolio tidak lengkap.")
        parent_items = {code: item for item, code in pc._query_items_with_ticker(db, parent.id)}
    item = current_items.get(ticker)
    previous = parent_items.get(ticker)
    if item is None and previous is None:
        raise pc.ItemNotFoundError("Saham tidak ditemukan dalam posisi atau rebalance sebelumnya.")
    current_lots = int(item.jumlah_lot or 0) if item else 0
    if current_lots != body.expected_lots:
        raise ValueError("Jumlah lot sudah berubah. Muat ulang portofolio sebelum menyimpan.")
    quantity = body.jumlah_lot
    price = float(body.harga_beli)
    basis_value = quantity * 100 * price
    if not math.isfinite(basis_value) or basis_value * 100 > 2**53 - 1:
        raise ValueError("Total harga beli terlalu besar untuk dicatat dengan akurat.")

    metadata = dict(portfolio.rebalance_metadata or {})
    new_cash, fees = float(portfolio.sisa_budget), float(portfolio.transaction_costs or 0)
    execution_prices = dict(portfolio.execution_prices or {})
    old_lots = {c: int(i.jumlah_lot) for c, i in parent_items.items() if i.jumlah_lot > 0}
    before = {c: int(i.jumlah_lot) for c, i in current_items.items() if i.jumlah_lot > 0}
    after = dict(before)
    if quantity:
        after[ticker] = quantity
    else:
        after.pop(ticker, None)
    reference = float(item.harga_acuan) if item else float(execution_prices.get(ticker) or 0)
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Harga acuan saham pada snapshot ini tidak tersedia.")
    if quantity != current_lots:
        if portfolio.execution_prices is None:
            raise ValueError("Snapshot lama tidak menyimpan rincian transaksi. Buat rebalance baru sebelum mengubah lot.")
        prices = {c: float(v) for c, v in execution_prices.items()}
        prices.setdefault(ticker, reference)
        # Recover this snapshot's opening cash from its original trade deltas.
        # A correction never consumes pending deposits or creates a contribution.
        cash_before = float(portfolio.sisa_budget) + fees
        for code in set(old_lots) | set(before):
            delta = before.get(code, 0) - old_lots.get(code, 0)
            if delta:
                mark = prices.get(code)
                if mark is None or not math.isfinite(mark) or mark <= 0:
                    raise ValueError(f"Rincian harga transaksi {code} tidak lengkap.")
                cash_before += delta * 100 * mark
        if fees and not all(key in metadata for key in ('buy_fee_rate', 'sell_fee_rate')):
            raise ValueError("Snapshot lama belum menyimpan tarif biaya. Buat rebalance baru sebelum mengubah lot.")
        buy_fee = metadata.get("buy_fee_rate", 0.)
        sell_fee = metadata.get("sell_fee_rate", 0.)
        new_cash, fees = transaction_account(old_lots, after, prices, cash_before, buy_fee, sell_fee)
        execution_prices = {c: str(prices[c]) for c in set(old_lots) | set(after)
                            if old_lots.get(c, 0) != after.get(c, 0)}
        metadata.update(buy_fee_rate=buy_fee, sell_fee_rate=sell_fee,
                        execution_transaction_costs=fees,
                        execution_delta_lots={c: after.get(c, 0) - old_lots.get(c, 0)
                            for c in set(old_lots) | set(after) if old_lots.get(c, 0) != after.get(c, 0)})

    if item is None:
        item = PortofolioItem(portofolio_id=identifier, stock_id=previous.stock_id,
            harga_acuan=reference, start_date=datetime.combine(portfolio.execution_date, datetime.min.time()),
            end_date=None)
        db.add(item)
        current_items[ticker] = item
    item.jumlah_lot = quantity
    item.harga_beli = price
    item.total_investasi = basis_value
    delta = quantity - old_lots.get(ticker, 0)
    item.action_type = "buy" if delta > 0 else "sell" if delta < 0 else "hold"
    total = sum(int(i.jumlah_lot or 0) * 100 * float(i.harga_acuan) for i in current_items.values())
    for position in current_items.values():
        position.bobot_persentase = (int(position.jumlah_lot or 0) * 100 * float(position.harga_acuan) / total * 100
                                     if total else 0.)
    portfolio.total_terpakai = total
    portfolio.sisa_budget = new_cash
    portfolio.transaction_costs = fees
    if quantity != current_lots:
        portfolio.execution_prices = execution_prices
    metadata.update(holdings_edited=True, holdings_edited_at=datetime.now(timezone.utc).isoformat())
    metadata['manual_cost_basis'] = {**metadata.get('manual_cost_basis', {}), ticker: price}
    portfolio.rebalance_metadata = metadata
    db.flush()
    result = pc._to_portfolio_response(db, portfolio)
    db.commit()
    return result
