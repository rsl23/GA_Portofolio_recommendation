"""Skrip untuk mengetes Algoritma Genetika dengan data LIVE dari ZAPI & Yahoo Finance."""

from src.gaengine.data_loader_live import build_market_data
from src.gaengine.engine import GeneticEngine
from src.gaengine.ga_config import GAConfig

def run_live_test():
    print("Membentuk MarketData menggunakan data LIVE...")
    
    # max_stocks = 20 agar tidak terlalu berat saat uji coba, 
    # bisa dikosongkan (None) agar seluruh saham dimasukkan
    data = build_market_data(min_price=50.0)
    
    if data is None:
        print("Gagal membentuk MarketData. Uji coba dibatalkan.")
        return

    print("\nData berhasil dirakit!")
    print(f"Total Saham  : {data.n_stocks} saham")
    print(f"Suku Bunga BI: {data.risk_free_rate*100:.2f}%\n")

    # Konfigurasi GA
    config = GAConfig(
        population_size=200,  # Gunakan 100 kromosom
        generations=300,      # Evolusi selama 150 keturunan
        budget=10_000_000.0,  # Modal Rp 10 Juta
        risk_profile="Moderate", # Profil risiko moderat
        min_stocks=3,
        max_stocks=10,
        risk_free_rate=data.risk_free_rate,
        seed=42,
        data=data,
    )

    print("=== MEMULAI EVOLUSI GENETIKA ===")
    engine = GeneticEngine(config)
    solution = engine.run(verbose=True)

    # ---- Laporan Hasil --------------------------------------------------------
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

    print("\n================== PORTOFOLIO TERBAIK (LIVE DATA) ==================")
    print(f"{'Ticker':<8}{'Lots':>6}{'Harga Lot':>14}{'Alokasi Dana':>16}{'Bobot':>10}")
    for code, lot, price in rows:
        w = (lot * price) / total if total else 0.0
        print(f"{code:<8}{lot:>6}{price:>14,.0f}{lot*price:>16,.0f}{w:>10.2%}")
    print("-" * 56)
    print(f"{'TOTAL':<8}{sum(r[1] for r in rows):>6}{'-':>14}{total:>16,.0f}")
    
    print(f"\nJumlah Saham (Diversifikasi) : {solution.n_active}")
    print(f"Skor Fitness                 : {solution.fitness:.4f}")
    print(f"Anggaran Aman?               : {'Ya' if solution.budget_ok else 'Tidak'}")

if __name__ == "__main__":
    run_live_test()

