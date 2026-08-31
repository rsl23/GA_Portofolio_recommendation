import os
import yfinance as yf
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path

from src.backend.models.database import SessionLocal
from src.backend.models.filtered_stocks_cache import FilteredStockCache
from src.backend.services.api_bi import fetch_bi_rate

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("BASE_URL")

def take_ohlcv_data():
    """
    Mengambil data saham yang lolos preprocessing dari DB (filtered_stock_cache),
    lalu menarik data OHLCV 1 tahun terakhir menggunakan yfinance.
    """
    db = SessionLocal()
    try:
        # 1. Ambil daftar 'kode' saham dari tabel FilteredStockCache
        # Hasil db.query ini berupa list of tuples: [('BBCA',), ('BMRI',), ...]
        records = db.query(FilteredStockCache.kode).all()
        daftar_kode = [r[0] for r in records]
        
        if not daftar_kode:
            print("⚠️ Peringatan: Tabel filtered_stock_cache kosong! Anda belum menjalankan preprocessing.")
            return None
            
        print(f"Mempersiapkan unduhan OHLCV 1 tahun untuk {len(daftar_kode)} saham dari yfinance...")
        
        # 2. Tambahkan akhiran '.JK' untuk format Yahoo Finance (saham Indonesia)
        yf_tickers = [f"{kode}.JK" for kode in daftar_kode]
        
        # 3. Eksekusi proses download data 1 tahun (period="1y")
        # Tanpa group_by='ticker', formatnya akan MultiIndex (Level 0: Atribut, Level 1: Ticker)
        # yang mana format ini jauh lebih gampang untuk dihitung return-nya nanti
        df_ohlcv = yf.download(yf_tickers, period="1y", interval="1d")
        
        print("Data OHLCV berhasil diunduh!")
        return df_ohlcv
        
    except Exception as e:
        print(f"Terjadi kesalahan saat menarik data OHLCV: {e}")
        return None
    finally:
        # Wajib tutup koneksi ke database setelah selesai agar memori tidak bocor
        db.close()
        
# CATATAN: Import ThreadPoolExecutor & as_completed sementara TIDAK dipakai.
# Multithreading (10 workers) sengaja dinonaktifkan; fetch fundamental berjalan SEQUENTIAL (1-per-1).
# Kode threading versi lamanya masih disimpan (di-comment) di dalam take_fundamental_data().
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.backend.services.api_pluang import fetch_api_pluang_fundamentals

def _parse_float_string(val_str):
    """
    Mengubah string teks finansial seperti 'Rp466.74', '13.61x', '20.44%', atau '-'
    menjadi tipe data float bersih yang bisa disimpan di database.
    """
    if not val_str or val_str == "-" or val_str == "N/A":
        return np.nan
    
    # Hapus karakter mata uang, pengali, persen, koma ribuan, dan spasi kosong
    clean_str = str(val_str).replace("Rp", "").replace(",", "").replace("x", "").replace("%", "").strip()
    
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

def process_single_fundamental(kode):
    """Fungsi mandiri untuk dipanggil oleh tiap Thread saat mengeksekusi 1 saham."""
    data = fetch_api_pluang_fundamentals(kode)
    if not data:
        return None
        
    try:
        ratios = data.get("ratios", {})
        overview = data.get("overview", {})
        
        # Ekstraksi string dari JSON sesuai path
        eps_str = overview.get("eps", "0")
        per_str = ratios.get("valuation", {}).get("pe", "0")
        pbv_str = ratios.get("valuation", {}).get("pb", "0")
        roe_str = ratios.get("profitability", {}).get("roe", "0")
        der_str = ratios.get("solvency", {}).get("de", "0")
        div_str = ratios.get("dividend", {}).get("ttm", "0")
        
        # Parse semua ke bentuk Float bersih
        return {
            "kode": kode,
            "eps": _parse_float_string(eps_str),
            "per": _parse_float_string(per_str),
            "pbv": _parse_float_string(pbv_str),
            "roe": _parse_float_string(roe_str),
            "der": _parse_float_string(der_str),
            "dividend_yield": _parse_float_string(div_str)
        }
    except Exception as e:
        print(f"Error parsing data fundamental Pluang untuk {kode}: {e}")
        return None

def take_fundamental_data():
    """
    Mengambil saham yang lolos filter, menarik data fundamental lengkap dari 
    ZAPI Pluang menggunakan Multithreading (10 workers), dan mengupdate tabel.
    """
    db = SessionLocal()
    try:
        records = db.query(FilteredStockCache).all()
        if not records:
            print("Peringatan: Tabel filtered_stock_cache kosong!")
            return None
            
        print(f"Mulai mengambil data fundamental Pluang untuk {len(records)} saham secara paralel...")
        
        results = []
        
        # --- VERSI SEQUENTIAL (1 PER 1) ---
        # Mengambil data satu per satu agar tidak membebani server API ZAPI
        for i, record in enumerate(records, 1):
            print(f"[{i}/{len(records)}] Menarik data {record.kode}...")
            res = process_single_fundamental(record.kode)
            if res:
                results.append(res)
                
        # --- VERSI MULTITHREADING (SEMENTARA DINONAKTIFKAN) ---
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     futures = {executor.submit(process_single_fundamental, record.kode): record for record in records}
        #     for future in as_completed(futures):
        #         res = future.result()
        #         if res:
        #             results.append(res)
                    
        # Update seluruh saham di database
        print(f"Mengupdate data {len(results)} saham ke dalam database...")
        for res_dict in results:
            db_item = db.query(FilteredStockCache).filter(FilteredStockCache.kode == res_dict["kode"]).first()
            if db_item:
                db_item.eps = res_dict["eps"]
                db_item.per = res_dict["per"]
                db_item.pbv = res_dict["pbv"]
                db_item.roe = res_dict["roe"]
                db_item.der = res_dict["der"]
                db_item.dividend_yield = res_dict["dividend_yield"]
                
        db.commit()
        print("Proses update fundamental ZAPI Pluang selesai!")
        
        return results

    except Exception as e:
        print(f"Terjadi kesalahan saat take_fundamental_data: {e}")
        db.rollback()
        return None
    finally:
        db.close()
    
def take_bi_rate():
    """Take the latest BI rate data"""
    try:
        # Panggil fungsi fetch_bi_rate() dari service API BI
        return fetch_bi_rate()
    except Exception as e:
        print(f"Terjadi kesalahan saat mengambil data BI Rate: {e}")
        return None

import numpy as np
from src.gaengine.market_data import MarketData

def _normalize_metrics(metrics: np.ndarray) -> np.ndarray:
    """Min-Max normalise each metric to 0-1 respecting its direction, then
    average per stock into a single composite fundamental score 0-1."""
    # Arah metrik: PER (+1, lower better), PBV (+1), ROE (-1, higher better), DER (+1), DivYld (-1)
    METRIC_DIRECTION = [+1, +1, -1, +1, -1]
    
    n, m = metrics.shape
    composite = np.zeros(n)
    for j in range(m):
        col = metrics[:, j]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            sub = np.full(n, 0.5)
        else:
            lo, hi = float(valid.min()), float(valid.max())
            span = hi - lo
            if span <= 1e-12:
                sub = np.full(n, 1.0)
            else:
                norm = (np.clip(col, lo, hi) - lo) / span
                if METRIC_DIRECTION[j] > 0:   # lower is better
                    sub = 1.0 - norm
                else:                          # higher is better
                    sub = norm
            sub[np.isnan(col)] = 0.5
        composite += sub
    return composite / m


def build_market_data(min_price: float = 50.0, max_stocks: int = None):
    """
    Build the MarketData object for live program.
    Fungsi ini adalah JEMBATAN PERAKITAN data mentah menjadi data siap komputasi GA.
    """
    print("Mempersiapkan perakitan MarketData...")
    
    # 1. Tarik & perbarui data terbaru
    df_ohlcv = take_ohlcv_data()
    take_fundamental_data()
    
    # Ambil suku bunga acuan BI, jika gagal pakai default 6.25%
    bi_rate = take_bi_rate()
    if bi_rate is None:
        bi_rate = 0.0625  
        
    if df_ohlcv is None or df_ohlcv.empty:
        print("Error: Gagal mengambil data OHLCV. Perakitan dibatalkan.")
        return None

    # 2. Ambil data Close dari MultiIndex DataFrame yfinance
    df_close = df_ohlcv["Close"].ffill().fillna(0) # Forward fill jika ada libur bursa
    
    # Hapus suffix '.JK' dari nama kolom agar namanya rapi ('BBCA' bukan 'BBCA.JK')
    df_close.columns = [str(c).replace(".JK", "") for c in df_close.columns]
    
    # 3. Ambil data fundamental terbaru langsung dari Database Cache
    db = SessionLocal()
    try:
        db_stocks = db.query(FilteredStockCache).all()
        # Buat dictionary map: {'BBCA': Objek BBCA, 'BMRI': Objek BMRI}
        fund_map = {s.kode: s for s in db_stocks}
        valid_codes = list(fund_map.keys())
    finally:
        db.close()
        
    # 4. Validasi Kandidat (Pastikan saham ada di OHLCV & Fundamental, serta harga >= 50)
    candidates = [c for c in df_close.columns if c in valid_codes and df_close[c].iloc[-1] >= min_price]
    candidates = sorted(candidates)
    
    if not candidates:
        print("Error: Tidak ada kandidat saham yang valid tersisa.")
        return None
        
    if max_stocks and len(candidates) > max_stocks:
        candidates = candidates[:max_stocks]
        
    # --- MULAI PEMBENTUKAN MATRIKS ---
    
    # A. Harga Per Lot (Tarik baris paling bawah/terakhir dari OHLCV, lalu x 100)
    prices_per_lot = np.asarray([df_close[c].iloc[-1] * 100.0 for c in candidates], dtype=float)
    
    # B. Returns Matrix (N saham x T hari)
    # df_close.pct_change() menghitung daily return. 
    df_ret = df_close[candidates].pct_change().fillna(0.0)
    returns = df_ret.to_numpy(dtype=float).T  # Transpose agar (N x T)
    
    # C. Correlation Matrix
    correlation = np.corrcoef(returns)
    correlation = np.nan_to_num(correlation, nan=0.0) # Aman dari NaN
    
    # D. Fundamental Metrics (Bentuk matriks [PER, PBV, ROE, DER, DivYld])
    metrics_list = []
    for c in candidates:
        s = fund_map[c]
        metrics_list.append([s.per, s.pbv, s.roe, s.der, s.dividend_yield])
    fundamental_metrics = np.asarray(metrics_list, dtype=float)
    
    # E. Fundamental Scores (Normalisasi matriks menjadi skor 0.0 - 1.0)
    fundamental_scores = _normalize_metrics(fundamental_metrics)
    
    # 5. Kunci & Kirim ke Objek MarketData
    market_data = MarketData(
        stock_codes=candidates,
        prices_per_lot=prices_per_lot,
        returns=returns,
        correlation=correlation,
        fundamental_scores=fundamental_scores,
        fundamental_metrics=fundamental_metrics,
        risk_free_rate=bi_rate
    )
    
    print(f"MarketData sukses dirakit untuk {market_data.n_stocks} saham unggulan!")
    return market_data