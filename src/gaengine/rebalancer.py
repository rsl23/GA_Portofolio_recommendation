"""
Rebalancing helper functions.
"""
from typing import List, Dict, Any, Tuple
import math


def fit_buys_to_available_cash(
    old_lots: Dict[str, int],
    desired_lots: Dict[str, int],
    cash_before: float,
    execution_prices: Dict[str, float],
    priorities: Dict[str, float],
    min_stocks: int = 3,
    buy_fee_rate: float = 0.0,
    sell_fee_rate: float = 0.0,
) -> Tuple[Dict[str, int], float]:
    """Trim only new BUY lots if execution prices exceed signal-day estimates.

    ``execution_prices`` are per share. Sales are preserved and their proceeds
    fund purchases. Lower-conviction buys are reduced first; the minimum
    diversification constraint is retained. No existing HOLD is sold here.
    """
    result = {code: int(qty) for code, qty in desired_lots.items() if qty > 0}
    if buy_fee_rate < 0 or sell_fee_rate < 0:
        raise ValueError("Tarif biaya transaksi tidak boleh negatif.")
    needed = set(old_lots) | set(result)
    for code in needed:
        price = execution_prices.get(code)
        if price is None or not math.isfinite(price) or price <= 0:
            raise ValueError(f"Harga eksekusi tidak valid untuk {code}.")

    cash_after = float(cash_before)
    for code in needed:
        delta = result.get(code, 0) - old_lots.get(code, 0)
        price = 100 * execution_prices[code]
        cash_after -= (delta * price * (1 + buy_fee_rate) if delta > 0
                       else delta * price * (1 - sell_fee_rate))
    if cash_after >= -0.01:
        return result, max(0.0, cash_after)

    buys = [code for code in result if result[code] > old_lots.get(code, 0)]
    buys.sort(key=lambda code: (priorities.get(code, 0.0), code))
    for code in buys:
        if cash_after >= -0.01:
            break
        max_trim = result[code] - old_lots.get(code, 0)
        if result[code] == max_trim and len(result) <= min_stocks:
            max_trim -= 1
        if max_trim <= 0:
            continue
        lot_cost = 100 * execution_prices[code] * (1 + buy_fee_rate)
        trim = min(max_trim, math.ceil((-cash_after - 0.01) / lot_cost))
        result[code] -= trim
        cash_after += trim * lot_cost
        if result[code] == 0:
            del result[code]

    if cash_after < -0.01:
        raise ValueError("Kas tidak mencukupi pada harga eksekusi tanpa melanggar batas diversifikasi.")
    return result, max(0.0, cash_after)

def compute_nav(old_lots: Dict[str, int], current_prices: Dict[str, float], leftover_cash: float) -> float:
    """
    Computes Net Asset Value (NAV) of the old portfolio based on current market prices.
    NAV = SUM(lots * current_price * 100) + leftover_cash
    """
    total_stock_value = 0.0
    for code, lots in old_lots.items():
        if code not in current_prices or not math.isfinite(current_prices[code]) or current_prices[code] <= 0:
            raise ValueError(f"Harga eksekusi tidak tersedia untuk posisi lama {code}.")
        # current_prices is expected to be price_per_lot
        total_stock_value += lots * current_prices[code]
    return total_stock_value + leftover_cash

def align_old_weights_to_market(old_lots: Dict[str, int], current_prices: Dict[str, float], leftover_cash: float, market_stock_codes: List[str]) -> Tuple[List[float], float]:
    """
    Calculates the weight of each stock in the old portfolio based on current NAV,
    and returns a list of weights aligned with the new MarketData stock_codes list.
    
    Returns:
        Tuple[List[float], float]: 
            - Aligned weights list (summing to <= 1.0)
            - The calculated NAV
    """
    nav = compute_nav(old_lots, current_prices, leftover_cash)
    if nav <= 0:
        return [0.0] * len(market_stock_codes), nav
        
    weights = []
    for code in market_stock_codes:
        if code in old_lots and code in current_prices:
            stock_value = old_lots[code] * current_prices[code]
            weights.append(stock_value / nav)
        else:
            weights.append(0.0)
            
    return weights, nav

def compute_rebalance_deltas(
    old_lots: Dict[str, int], 
    new_lots: Dict[str, int], 
    old_weights: Dict[str, float],
    new_weights: Dict[str, float],
    eliminated_stocks: List[str],
    old_cost_basis: Dict[str, float | None] | None = None,
    execution_prices: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    """
    Computes the delta (Buy, Sell, Hold) for each stock.
    The reason describes the GA/filter decision; realized profit or loss is
    reported separately and never treated as the decision's cause.
    """
    all_codes = set(old_lots.keys()).union(set(new_lots.keys()))
    deltas = []
    old_cost_basis = old_cost_basis or {}
    execution_prices = execution_prices or {}
    
    for code in sorted(all_codes):
        old_l = old_lots.get(code, 0)
        new_l = new_lots.get(code, 0)
        basis = old_cost_basis.get(code)
        price = execution_prices.get(code)
        has_basis_and_price = (
            basis is not None and price is not None and
            math.isfinite(basis) and basis > 0 and
            math.isfinite(price) and price > 0
        )
        unrealized_pct_before = (
            100.0 * (price / basis - 1.0)
            if old_l > 0 and has_basis_and_price else None
        )
        sold_lots = max(0, old_l - new_l)
        realized_pnl = (
            sold_lots * 100.0 * (price - basis)
            if sold_lots and has_basis_and_price else None
        )
        realization_type = (
            "REALIZED_GAIN" if realized_pnl > 0 else
            "REALIZED_LOSS" if realized_pnl < 0 else "BREAKEVEN"
        ) if realized_pnl is not None else None
        
        if old_l == new_l:
            # Action: Hold
            if old_l > 0:
                deltas.append({
                    "kode": code,
                    "action": "HOLD",
                    "old_lot": old_l,
                    "new_lot": new_l,
                    "delta_lot": 0,
                    "reason": "Dipertahankan oleh optimasi GA; tidak ada transaksi",
                    "action_detail": "HOLD",
                    "unrealized_return_pct_before": unrealized_pct_before,
                    "realized_pnl": None,
                    "realization_type": None,
                })
            continue
            
        action = "BUY" if new_l > old_l else "SELL"
        delta_l = abs(new_l - old_l)
        
        action_detail = (
            "BUY" if old_l == 0 else "ADD" if new_l > old_l else
            "FULL_SELL" if new_l == 0 else "REDUCE"
        )
        reason = ("Penambahan posisi berdasarkan optimasi GA" if action == "BUY"
                  else "Pengurangan posisi berdasarkan optimasi GA")
        if action == "SELL" and new_l == 0 and code in eliminated_stocks:
            reason = "Keluar dari kandidat saham; posisi dilepas saat rebalance"
            
        deltas.append({
            "kode": code,
            "action": action,
            "old_lot": old_l,
            "new_lot": new_l,
            "delta_lot": delta_l,
            "reason": reason,
            "action_detail": action_detail,
            "unrealized_return_pct_before": unrealized_pct_before,
            "realized_pnl": realized_pnl,
            "realization_type": realization_type,
        })
        
    return deltas

def apply_no_trade_band(
    best_chrom_weights: List[float], 
    old_weights_aligned: List[float], 
    threshold: float = 0.03,
    min_active: int | None = None,
    max_active: int | None = None,
) -> List[float]:
    """
    Revert small weight changes without breaking the GA's stock-count bounds.

    A small old position can otherwise be reactivated alongside all of the
    GA-selected positions, creating an invalid extra holding. If the banded
    portfolio would violate the supplied bounds, retain the GA decision.
    """
    if len(best_chrom_weights) != len(old_weights_aligned):
        raise ValueError("Bobot portofolio lama dan baru harus sejajar.")
    new_w = list(best_chrom_weights)
    for i in range(len(new_w)):
        diff = abs(new_w[i] - old_weights_aligned[i])
        if diff < threshold:
            new_w[i] = old_weights_aligned[i]

    active = sum(w > 0 for w in new_w)
    if ((min_active is not None and active < min_active) or
        (max_active is not None and active > max_active)):
        return list(best_chrom_weights)
            
    # Re-normalize if needed to ensure we don't exceed 1.0 (though lots calculation will handle cash leftover)
    s = sum(new_w)
    if s > 1.0:
        new_w = [w / s for w in new_w]
        
    return new_w
