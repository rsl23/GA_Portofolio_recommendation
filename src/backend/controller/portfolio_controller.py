from uuid import uuid4

import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.backend.models.schemas.portfolio_schema import PortfolioGenerateRequest
from src.gaengine.data_loader_live import build_market_data
from src.gaengine.engine import GeneticEngine
from src.gaengine.ga_config import GAConfig


def generate_new_portfolio(db: Session, request: PortfolioGenerateRequest, market_data=None):
    """
    Controller Otak Utama:
    1. Memakai MarketData live (bisa reused dari app.state atau dibangun ulang).
    2. Menjalankan Algoritma Genetika sesuai modal & profil risiko pengguna.
    3. Menyusun laporan hasil (tanpa menyimpan ke database untuk sekarang).
    """
    print(f"Menjalankan GA untuk profil {request.risk_profile} dengan modal Rp{request.budget:,.2f}")

    # 1. Siapkan MarketData. Kalau tidak diberi (None) dari luar, bangun sendiri.
    if market_data is None:
        print("Membentuk MarketData menggunakan data LIVE...")
        market_data = build_market_data(min_price=50.0)

    if market_data is None or market_data.n_stocks == 0:
        raise HTTPException(
            status_code=500,
            detail="Gagal membentuk MarketData. Pastikan tabel filtered_stock_cache sudah terisi.",
        )

    # 2. Konfigurasi GA mengikuti profil & modal dari payload request
    config = GAConfig(
        population_size=200,        # jumlah kromosom
        generations=300,            # jumlah generasi evolusi
        budget=request.budget,      # modal pengguna (IDR)
        risk_profile=request.risk_profile,
        min_stocks=3,
        max_stocks=10,
        risk_free_rate=market_data.risk_free_rate,
        seed=42,
        data=market_data,
    )

    # 3. Evolusi genetika
    print("=== MEMULAI EVOLUSI GENETIKA ===")
    engine = GeneticEngine(config)
    solution = engine.run(verbose=True)

    # 4. Susun laporan hasil
    prices = market_data.prices_per_lot
    codes = market_data.stock_codes
    lots = solution.lots
    alloc = lots * prices
    total = float(alloc.sum())

    rows = sorted(
        ((codes[i], int(lots[i]), float(prices[i])) for i in range(len(codes))),
        key=lambda r: r[1] * r[2],
        reverse=True,
    )
    allocations = [
        {
            "ticker": code,
            "lots": lot,
            "price_per_lot": price,
            "allocation": float(lot * price),
            "weight": float(lot * price / total) if total else 0.0,
        }
        for code, lot, price in rows
        if lot > 0
    ]

    # Estimasi return tahunan dari alokasi aktual (bobot = alokasi / total)
    expected_return = None
    if total > 0 and market_data.returns.shape[0] >= solution.n_active:
        aw = alloc / total
        p_daily = (market_data.returns * aw[:, None]).sum(axis=0)
        p_daily = p_daily[~np.isnan(p_daily) & ~np.isinf(p_daily)]
        if p_daily.size:
            expected_return = float(p_daily.mean() * 252.0)

    narasi = (
        f"Portofolio dengan profil risiko {request.risk_profile} memilih {solution.n_active} "
        f"saham dari {market_data.n_stocks} kandidat. Total dana terpakai "
        f"Rp{total:,.0f} dari budget Rp{request.budget:,.0f}. Skor fitness yang dicapai "
        f"{solution.fitness:.4f} dengan Sharpe ratio {solution.sharpe_ratio:.3f}. "
        f"Penurunan maksimum (max drawdown) tercatat {solution.max_drawdown:.2%} "
        f"dan korelasi rata-rata antar saham {solution.avg_correlation:.3f}. "
        f"Rekomendasi ini dihasilkan Algoritma Genetika dan belum disimpan ke database."
    )

    return {
        "id": str(uuid4()),
        "fitness_score": float(solution.fitness),
        "sharpe_ratio": float(solution.sharpe_ratio),
        "expected_return": expected_return,
        "max_drawdown": float(solution.max_drawdown),
        "avg_correlation": float(solution.avg_correlation),
        "skor_fundamental": float(solution.skor_fundamental),
        "total_terpakai": total,
        "sisa_budget": request.budget - total,
        "n_active": int(solution.n_active),
        "allocated_budget_ok": bool(solution.budget_ok),
        "risk_profile": request.risk_profile,
        "budget": request.budget,
        "allocations": allocations,
        "narasi_llm": narasi,
    }
