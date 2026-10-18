"""
Rebalancing helper functions.
"""
from typing import List, Dict, Any, Tuple
import math

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
    eliminated_stocks: List[str]
) -> List[Dict[str, Any]]:
    """
    Computes the delta (Buy, Sell, Hold) for each stock.
    Categorizes the reason as "Eliminasi Kualitas" or "Optimasi Algoritma / Take Profit".
    """
    all_codes = set(old_lots.keys()).union(set(new_lots.keys()))
    deltas = []
    
    for code in all_codes:
        old_l = old_lots.get(code, 0)
        new_l = new_lots.get(code, 0)
        
        if old_l == new_l:
            # Action: Hold
            if old_l > 0:
                deltas.append({
                    "kode": code,
                    "action": "HOLD",
                    "old_lot": old_l,
                    "new_lot": new_l,
                    "delta_lot": 0,
                    "reason": "Dipertahankan (Hold)"
                })
            continue
            
        action = "BUY" if new_l > old_l else "SELL"
        delta_l = abs(new_l - old_l)
        
        reason = "Optimasi Algoritma / Take Profit"
        if action == "SELL" and new_l == 0 and code in eliminated_stocks:
            reason = "Eliminasi Kualitas (Fundamental/Likuiditas memburuk)"
            
        deltas.append({
            "kode": code,
            "action": action,
            "old_lot": old_l,
            "new_lot": new_l,
            "delta_lot": delta_l,
            "reason": reason
        })
        
    return deltas

def apply_no_trade_band(
    best_chrom_weights: List[float], 
    old_weights_aligned: List[float], 
    threshold: float = 0.03
) -> List[float]:
    """
    Reverts weights to their old values if the absolute change is less than the threshold (3%).
    """
    new_w = list(best_chrom_weights)
    for i in range(len(new_w)):
        diff = abs(new_w[i] - old_weights_aligned[i])
        if diff < threshold:
            new_w[i] = old_weights_aligned[i]
            
    # Re-normalize if needed to ensure we don't exceed 1.0 (though lots calculation will handle cash leftover)
    s = sum(new_w)
    if s > 1.0:
        new_w = [w / s for w in new_w]
        
    return new_w
