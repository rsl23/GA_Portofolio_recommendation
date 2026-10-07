"""
Uji Narasi LLM (prompt_builder + gemini_llm_caller)
===================================================

Skrip ini memvalidasi alur penjelasan portofolio oleh LLM tanpa bergantung pada
database / internet data pasar:

    1. Merakit MarketData SINTETIS (30 emiten x 400 hari return) - deterministik.
    2. Menjalankan GA singkat (populasi 40, 30 generasi) untuk mendapat satu
       solusi portofolio nyata (Chromosome + lots + metrik fitness).
    3. Menyusun konteks + prompt lewat src/gaengine/prompt_builder.py dan
       mencetaknya (tanpa jaringan).
    4. Memanggil src/backend/services/gemini_llm_caller.py untuk meminta narasi
       penjelasan dari model Gemini (default gemini-3.8-flash).

Cara pakai:
    python test_llm_narrative.py               # rakit prompt + panggil LLM
    python test_llm_narrative.py --prompt-only # hanya rakit prompt (tanpa API)
"""

import sys

import numpy as np

from src.backend.services.gemini_llm_caller import explain_portfolio, is_available
from src.gaengine.engine import GeneticEngine
from src.gaengine.ga_config import GAConfig
from src.gaengine.market_data import MarketData
from src.gaengine.prompt_builder import build_portfolio_context, render_context_text

# 30 kode emiten contoh (data di skrip ini SINTETIS, bukan harga pasar nyata).
CODES = [
    "BBCA", "BBRI", "BMRI", "TLKM", "ASII", "UNVR", "ICBP", "INDF", "KLBF", "ANTM",
    "ADRO", "PTBA", "SMGR", "INTP", "JSMR", "TOWR", "EXCL", "ISAT", "MEDC", "PGAS",
    "AKRA", "ERAA", "MAPI", "ACES", "CPIN", "JPFA", "MYOR", "SIDO", "TSPC", "HMSP",
]


def build_synthetic_market_data(seed: int = 0) -> MarketData:
    """Bikin MarketData sintetis: returns, korelasi, fundamental, dan harga/lot."""
    rng = np.random.default_rng(seed)
    n, t = len(CODES), 400

    # Return harian: sebagian emiten dibuat saling berkorelasi (faktor pasar).
    faktor = rng.normal(0.0004, 0.011, size=t)
    idio = rng.normal(0.0002, 0.018, size=(t, n))
    beta = rng.uniform(0.5, 1.5, size=n)
    returns = (faktor[:, None] * beta[None, :] + idio).T          # (n, T)

    correlation = np.nan_to_num(np.corrcoef(returns))
    prices_per_lot = rng.uniform(50_000, 1_500_000, size=n)        # IDR per lot
    fundamental_metrics = np.column_stack([
        rng.uniform(5.0, 25.0, n),      # PER (x)
        rng.uniform(0.5, 4.0, n),       # PBV (x)
        rng.uniform(5.0, 30.0, n),      # ROE (%)
        rng.uniform(0.1, 1.5, n),       # DER (rasio)
        rng.uniform(0.0, 0.08, n),      # Dividend yield (fraksi)
    ])
    # Skor komposit 0-1 (di skrip uji cukup uniform; di produksi dihitung loader).
    fundamental_scores = rng.uniform(0.35, 0.95, n)

    return MarketData(
        stock_codes=list(CODES),
        prices_per_lot=prices_per_lot,
        returns=returns,
        correlation=correlation,
        fundamental_scores=fundamental_scores,
        fundamental_metrics=fundamental_metrics,
        risk_free_rate=0.0625,
    )


def main() -> None:
    print("=== 1. Merakit MarketData sintetis & menjalankan GA ===")
    data = build_synthetic_market_data()
    config = GAConfig(
        population_size=40,
        generations=30,
        budget=10_000_000.0,
        risk_profile="Moderate",
        min_stocks=3,
        max_stocks=6,
        risk_free_rate=data.risk_free_rate,
        seed=7,
        data=data,
    )
    engine = GeneticEngine(config)
    solution = engine.run(verbose=False)
    print(f"GA selesai: fitness={solution.fitness:.4f}, n_active={solution.n_active}, "
          f"sharpe={solution.sharpe_ratio:.3f}, mdd={solution.max_drawdown:.2%}")

    print("\n=== 2. Konteks metrik yang dikirim ke LLM ===")
    context = build_portfolio_context(
        solution, data, config,
        mode="live",
        budget=config.budget,
        generations_run=len(engine.history),
    )
    print(render_context_text(context))

    if "--prompt-only" in sys.argv:
        print("\n[--prompt-only] Pemanggilan LLM dilewati.")
        return

    print("\n=== 3. Memanggil LLM untuk narasi penjelasan ===")
    print(f"LLM tersedia? {is_available()}")
    narasi = explain_portfolio(
        solution=solution,
        market_data=data,
        config=config,
        mode="live",
        budget=config.budget,
        generations_run=len(engine.history),
        fallback="[FALLBACK] LLM tidak tersedia / gagal, narasi template dipakai.",
    )

    print("\n======================= NARASI LLM =======================")
    print(narasi)
    print("==========================================================")


if __name__ == "__main__":
    main()
