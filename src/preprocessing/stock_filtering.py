import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. FUNGSI UNTUK MENARIK INFO FUNDAMENTAL SECARA PARALEL
def fetch_fundamental(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'Ticker': ticker,
            'Market_Cap': info.get('marketCap', 0),
            'EPS': info.get('trailingEps', 0),
            'ROE': info.get('returnOnEquity', 0),
            'DER': info.get('debtToEquity', 0), # Ingat, ini format persentase (200 = 2.0)
            'Sector': info.get('sector', 'Unknown')
        }
    except Exception:
        return None

def run_live_preprocessing(daftar_ticker):
    print("Mengunduh data fundamental (Paralel)...")
    fundamentals = []
    
    # Multithreading agar 900 saham selesai dalam 1-2 menit
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_fundamental, t): t for t in daftar_ticker}
        for future in as_completed(futures):
            res = future.result()
            if res:
                fundamentals.append(res)
                
    df_fund = pd.DataFrame(fundamentals).set_index('Ticker')

    print("Mengunduh data teknikal OHLCV...")
    # Tarik data 90 hari kalender untuk memastikan dapat 60 hari bursa
    df_tech = yf.download(daftar_ticker, period="2y", interval="1d", group_by='ticker')
    
    # 2. MENGHITUNG METRIK TEKNIKAL SECARA VECTORIZED
    tech_metrics = []
    for ticker in daftar_ticker:
        try:
            # Handle multi-index yfinance
            if len(daftar_ticker) > 1:
                df_saham = df_tech[ticker].dropna()
            else:
                df_saham = df_tech.dropna()
                
            if len(df_saham) < 500:
                continue # Skip jika data belum 500 hari (IPO baru)
                
            # Hitung ADTV (Average Daily Trading Value) 60 hari terakhir
            trading_value = df_saham['Close'] * df_saham['Volume']
            adtv_60 = trading_value.tail(60).mean()
            
            # Harga penutupan terakhir
            last_close = df_saham['Close'].iloc[-1]
            
            # Deteksi Volume Sporadis (Contoh: Median volume 30 hari mati, tapi volume hari ini meledak)
            vol_30d_median = df_saham['Volume'].tail(30).median()
            vol_today = df_saham['Volume'].iloc[-1]
            is_sporadic = (vol_30d_median < 1000) and (vol_today > 1000000) # Bisa disesuaikan
            
            tech_metrics.append({
                'Ticker': ticker,
                'ADTV_60': adtv_60,
                'Last_Close': last_close,
                'Is_Sporadic': is_sporadic
            })
        except Exception:
            continue
            
    df_tech_summary = pd.DataFrame(tech_metrics).set_index('Ticker')

    # 3. GABUNGKAN DATA FUNDAMENTAL & TEKNIKAL
    df_master = df_fund.join(df_tech_summary, how='inner')

    # 4. TERAPKAN FILTER ELIMINASI MIKRO (Vectorized)
    cond_mcap = df_master['Market_Cap'] >= 5_000_000_000_000      # > Rp 5 Triliun
    cond_eps  = df_master['EPS'] > 0                              # EPS positif
    cond_roe  = df_master['ROE'] > 0                              # ROE positif
    cond_adtv = df_master['ADTV_60'] >= 1_000_000_000             # ADTV > Rp 1 Miliar
    cond_floor = df_master['Last_Close'] > 50                     # Hindari saham gocap
    cond_sporadic = ~df_master['Is_Sporadic']                     # Harus False (Bukan sporadis)
    
    # Logika DER: DER < 200 (karena yfinance pakai %) ATAU Sektor Finansial
    cond_der = (df_master['DER'] < 200) | (df_master['Sector'] == 'Financial Services')

    # Filter eksekusi
    df_lolos = df_master[cond_mcap & cond_eps & cond_roe & cond_adtv & cond_floor & cond_sporadic & cond_der]

    print(f"Preprocessing Live Selesai! Dari {len(daftar_ticker)} saham, tersisa {len(df_lolos)} saham unggulan.")
    return df_lolos.index.tolist()
