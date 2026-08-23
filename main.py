"""End-to-end runner for the portfolio-optimisation genetic algorithm."""

from __future__ import annotations

import argparse

from src.gaengine.data_loader import build_market_data
from src.gaengine.engine import GeneticEngine
from src.gaengine.ga_config import GAConfig


def _parse_args():
    p = argparse.ArgumentParser(description="Portfolio optimisation with a Genetic Algorithm")
    p.add_argument("--budget", type=float, default=100_000_000.0,
                   help="Total capital (IDR)")
    p.add_argument("--risk-profile", default="Moderate",
                   choices=["Conservative", "Moderate", "Aggressive"])
    p.add_argument("--population", type=int, default=200)
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--max-stocks", type=int, default=10)
    p.add_argument("--universe", type=int, default=None,
                   help="Cap the candidate universe to the most liquid N stocks")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("Loading market data ...")
    data = build_market_data(min_price=50.0, max_stocks=args.universe)

    print(f"  universe     : {data.n_stocks} stocks")
    print(f"  risk-free rf : {data.risk_free_rate:.4f} ({data.risk_free_rate*100:.2f} %)")

    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        budget=args.budget,
        risk_profile=args.risk_profile,
        min_stocks=3,
        max_stocks=args.max_stocks,
        risk_free_rate=data.risk_free_rate,
        seed=args.seed,
        data=data,
    )

    engine = GeneticEngine(config)
    solution = engine.run(verbose=True)

    # ---- report --------------------------------------------------------
    prices = data.prices_per_lot
    codes = data.stock_codes
    lots = solution.lots
    alloc = lots * prices
    total = float(alloc.sum())

    rows = sorted(
        ((codes[i], int(lots[i]), float(prices[i])) for i in range(len(codes))),
        key=lambda r: r[1] * r[2],
        reverse=True,
    )
    rows = [r for r in rows if r[1] > 0]

    print("\n================== BEST PORTFOLIO ==================")
    print(f"{'Ticker':<8}{'Lots':>6}{'LotPrice':>14}{'Cost':>16}{'Weight':>10}")
    for code, lot, price in rows:
        w = (lot * price) / total if total else 0.0
        print(f"{code:<8}{lot:>6}{price:>14,.0f}{lot*price:>16,.0f}{w:>10.2%}")
    print("-" * 56)
    print(f"{'TOTAL':<8}{sum(r[1] for r in rows):>6}{'-':>14}{total:>16,.0f}")
    print(f"\nn_active       : {solution.n_active}")
    print(f"fitness        : {solution.fitness:.4f}")
    print(f"budget_ok      : {solution.budget_ok}  (cost <= budget)")
    print(f"diversification : {solution.diversification_ok}  (3-10 active)")


if __name__ == "__main__":
    main()
