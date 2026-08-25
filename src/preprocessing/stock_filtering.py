import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.backend.services.api_idx import fetch_api_idx_saham, fetch_api_stock_summary

def fetch_fundamental_minimal(ticker):
    """Hanya mengambil EPS, ROE, dan DER. Market Cap & Sektor sudah di-handle API"""
    try:
        info = yf.Ticker(ticker + ".JK").info
        return {
            'Kode': ticker, # Samakan key dengan master agar mudah di-join
            'EPS': info.get('trailingEps', 0) or 0,
            'ROE': info.get('returnOnEquity', 0) or 0,
            'DER': info.get('debtToEquity', 0) or 0
        }
    except Exception:
        return None

def run_live_preprocessing():
    print("1. Mengambil & Menyaring Data dari API IDX...")

    # Tarik data API secara paralel (di memori)
    df_companies = pd.DataFrame(fetch_api_idx_saham()) 
    df_summary = pd.DataFrame(fetch_api_stock_summary())
    
    print(f"Data API IDX berhasil diambil: {len(df_companies)} emiten, {len(df_summary)} ringkasan saham.")

    if df_companies.empty or df_summary.empty:
        raise ValueError("Data dari API ZPI kosong! Mohon cek koneksi atau konfigurasi API Key Anda.")

    # FILTER 1A: Usia IPO > 2 Tahun
    df_companies['TanggalPencatatan'] = pd.to_datetime(df_companies['TanggalPencatatan'])
    batas_usia = datetime.now() - timedelta(days=730)
    cond_age = df_companies['TanggalPencatatan'] <= batas_usia
    
    # --- LOGGING: Cek siapa yang gagal umur ---
    failed_age = df_companies[~cond_age]['Kode'].tolist()
    if failed_age:
        print(f"   [!] Gagal Filter IPO < 2 Tahun ({len(failed_age)} saham) -> Contoh: {failed_age[:7]}...")
        
    df_companies = df_companies[cond_age]

    # GABUNGKAN DATA (Companies + Summary)
    # df_master kini memiliki: Sektor, ListingDate, Close, dan ListedShares
    df_master = pd.merge(df_companies, df_summary, on='Kode', how='inner').set_index('Kode')

    # FILTER 1B: Market Cap & Harga Lantai (EKSEKUSI KILAT DI MEMORI)
    df_master['Market_Cap'] = df_master['ListedShares'] * df_master['Close']
    
    cond_mcap = df_master['Market_Cap'] >= 5_000_000_000_000      # > Rp 5 Triliun
    cond_floor = df_master['Close'] > 50                          # Bukan saham gocap

    # --- LOGGING: Cek siapa yang gagal ---
    failed_mcap = df_master[~cond_mcap].index.tolist()
    if failed_mcap:
        print(f"   [!] Gagal Market Cap < Rp5T ({len(failed_mcap)} saham) -> Contoh: {failed_mcap[:7]}...")
        
    failed_floor = df_master[~cond_floor].index.tolist()
    if failed_floor:
        print(f"   [!] Gagal Harga <= Rp50 / Gocap ({len(failed_floor)} saham) -> Contoh: {failed_floor[:7]}...")

    # Terapkan filter awal sebelum menyentuh yfinance!
    df_master = df_master[cond_mcap & cond_floor]
    
    print(f"Lolos Filter BEI (Usia, Papan, Market Cap, Harga): {len(df_master)} saham.")
    print("2. Mengunduh Fundamental & OHLCV via yfinance...")

    # HANYA unduh saham yang sudah lolos filter Market Cap 
    ticker_yf = [t + ".JK" for t in df_master.index] 

    # Tarik OHLCV (Hanya untuk keperluan ADTV 60 dan Vol Sporadis)
    df_tech = yf.download(ticker_yf, period="90d", interval="1d", group_by='ticker')

    # Eksekusi Multithreading untuk data fundamental (EPS, ROE, DER)
    fundamentals = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_fundamental_minimal, t): t for t in df_master.index}
        for future in as_completed(futures):
            res = future.result()
            if res: fundamentals.append(res)

    df_fund = pd.DataFrame(fundamentals).set_index('Kode')
    df_master = df_master.join(df_fund, how='left')

    print("3. Menghitung Metrik Teknikal (ADTV)...")
    tech_metrics = []
    for ticker in df_master.index:
        try:
            ticker_jk = ticker + ".JK"
            df_saham = df_tech[ticker_jk].dropna() if len(ticker_yf) > 1 else df_tech.dropna()

            if len(df_saham) == 0:
                continue

            # ADTV 60 Hari
            trading_value = df_saham['Close'] * df_saham['Volume']
            adtv_60 = trading_value.tail(60).mean()

            # Deteksi Sporadis
            vol_30d_median = df_saham['Volume'].tail(30).median()
            vol_today = df_saham['Volume'].iloc[-1]
            is_sporadic = (vol_30d_median < 1000) and (vol_today > 1000000)

            tech_metrics.append({
                'Kode': ticker,
                'ADTV_60': float(adtv_60),
                'Is_Sporadic': is_sporadic
            })
        except Exception:
            continue

    df_tech_summary = pd.DataFrame(tech_metrics).set_index('Kode')
    df_master = df_master.join(df_tech_summary, how='inner')

    # FILTER 2: Eksekusi Eliminasi Mikro (Fundamental & Likuiditas)
    cond_eps  = df_master['EPS'] > 0                              
    cond_roe  = df_master['ROE'] > 0                              
    cond_adtv = df_master['ADTV_60'] >= 1_000_000_000             
    cond_sporadic = ~df_master['Is_Sporadic']                     

    # Logika Bypass DER: Nilai DER < 200 ATAU Sektornya Keuangan
    cond_der = (df_master['DER'] < 200) | (df_master['Sektor'] == 'Keuangan') 

    # --- LOGGING: Rincian Eliminasi Mikro ---
    print("\n--- RINCIAN ELIMINASI MIKRO (FUNDAMENTAL & LIKUIDITAS) ---")
    
    failed_eps = df_master[~cond_eps].index.tolist()
    if failed_eps: print(f"   [!] Gagal Laba EPS <= 0 ({len(failed_eps)} saham) -> Contoh: {failed_eps[:7]}...")
    
    failed_roe = df_master[~cond_roe].index.tolist()
    if failed_roe: print(f"   [!] Gagal ROE <= 0 ({len(failed_roe)} saham) -> Contoh: {failed_roe[:7]}...")
    
    failed_adtv = df_master[~cond_adtv].index.tolist()
    if failed_adtv: print(f"   [!] Gagal ADTV < Rp1 Miliar/Hari ({len(failed_adtv)} saham) -> Contoh: {failed_adtv[:7]}...")
    
    failed_sporadic = df_master[df_master['Is_Sporadic']].index.tolist()
    if failed_sporadic: print(f"   [!] Gagal Volume Sporadis / Manipulasi ({len(failed_sporadic)} saham) -> Contoh: {failed_sporadic[:7]}...")
    
    failed_der = df_master[~cond_der].index.tolist()
    if failed_der: print(f"   [!] Gagal Utang DER >= 200% (Bukan Bank) ({len(failed_der)} saham) -> Contoh: {failed_der[:7]}...")

    df_lolos = df_master[cond_eps & cond_roe & cond_adtv & cond_sporadic & cond_der]

    print(f"Pipeline Live Selesai! Dari ratusan emiten, tersisa {len(df_lolos)} saham unggulan.")
    return df_lolos.index.tolist(), df_lolos