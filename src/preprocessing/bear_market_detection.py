from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

ROOT      = Path(__file__).resolve().parent.parent.parent
DATA_DIR  = ROOT / "data"

STOCKS_FILE   = DATA_DIR / "Daftar Saham - 20260401.xlsx"
PRICE_FILE    = DATA_DIR / "Master_OHLCV_15Tahun.parquet"

MARKET_CONTEXT_FILE   = DATA_DIR / "Master_Market_Context_15Tahun.parquet"

def _check_death_cross(date: date = None) -> bool:
    """Check if the market is in a bear condition using the Death Cross method.

        Returns:
            bool: True if the market is in a bear condition, False otherwise.
    """
    today = date
    two_years_ago = today.replace(year=today.year - 2)

    start_date = two_years_ago.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    df = yf.download("^JKSE", start=start_date, end=end_date, interval="1d")
    
    if(df.empty):
        raise ValueError("Failed to fetch data from Yahoo Finance. Please check your internet connection or the ticker symbol.")
    
    # 1. Menghitung MA-50
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    # 2. Menghitung MA-200
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # Tampilkan 5 baris terakhir untuk mengecek hasilnya
    print(df[['Close', 'MA_50', 'MA_200']].tail())
    
    is_death_cross = False
    for i in range(5):
        current_day = df.iloc[-(i+1)]
        if(current_day['MA_50'] > current_day['MA_200']):
            is_death_cross = False
            break
        else:
            is_death_cross = True

    if is_death_cross:
        print("ALERT: IHSG sedang mengalami Death Cross (Bear Market)!")
        # Di sini kamu bisa memicu perubahan 'market_trend_flag' menjadi 'Bearish'
    else:
        print("IHSG Aman (Bullish / Normal).")
    
    return is_death_cross


def _check_majority_stock(daftar_ticker: list, target_date: date = None) -> bool:
    """Check if majority (60% or more) of the indonesian stocks are below MA-200."""
    if target_date is None:
        target_date = date.today()
        
    start_date = target_date - timedelta(days=300)
    # yfinance butuh end_date + 1 hari agar data target_date ikut terambil
    end_date = target_date + timedelta(days=1)
    
    # PERHATIAN: Hapus argumen group_by='ticker' agar struktur DataFramenya 
    # terkelompok berdasarkan atribut (Close, Open) BUKAN berdasarkan Ticker. 
    # Ini syarat utama agar Vectorization berjalan mulus.
    df = yf.download(daftar_ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d")
    
    if df.empty:
        return False
        
    # 1. Ekstrak HANYA matriks 'Close' (Harga Penutupan) untuk SEMUA saham sekaligus
    # Bentuknya jadi Dataframe: Baris = Tanggal, Kolom = BBCA, TLKM, dst.
    df_close = df['Close']
    
    # --- MULAI VECTORIZATION ---
    
    # 2. Hitung MA-200 untuk SEMUA SAHAM secara serentak (1 baris kode!)
    df_ma200 = df_close.rolling(window=200).mean()
    
    # 3. Ambil data hari terakhir (paling ujung) dari harga dan MA-200
    latest_close = df_close.iloc[-1]
    latest_ma200 = df_ma200.iloc[-1]
    
    # Saham yang usianya belum 200 hari bursa, nilai MA-200-nya akan kosong (NaN).
    # Kita buat mask untuk membuang saham-saham NaN ini dari perhitungan 
    # agar persentasenya tidak bocor.
    valid_mask = latest_ma200.notna()
    
    # 4. Bandingkan secara Vektoral (Array vs Array)
    is_below_ma200 = latest_close[valid_mask] < latest_ma200[valid_mask]
    
    # 5. Hitung Persentase
    percentage_below = float(is_below_ma200.mean())
    
    print(f"[Bear Market] Saham di bawah MA-200: {percentage_below * 100:.2f}%")
    
    # Return True jika > 60% (0.60)
    return percentage_below > 0.60
    
    
def _check_macro_economy(target_date: date = None) -> bool:
    """Check if USD/IDR depreciate > 5% in the last 30 trading days OR VIX > 30."""
    
    # Jika tidak ada parameter, gunakan hari ini (Live Mode)
    if target_date is None:
        target_date = date.today()
        
    # Mundur 45 hari kalender pakai timedelta (anti-crash) untuk dapat ~30 hari bursa
    start_date = target_date - timedelta(days=45)
    
    # yfinance butuh end_date + 1 hari agar data target_date ikut terambil
    end_date = target_date + timedelta(days=1) 
    
    # HAPUS 'period', gunakan start dan end untuk kompatibilitas backtesting
    df = yf.download(["^VIX", "IDR=X"], start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d")
    
    if df.empty:
        raise ValueError("Failed to fetch data from Yahoo Finance.")
        
    # 1. Ekstraksi Data (Mengatasi format MultiIndex yfinance)
    df_close = df['Close']
    df_idr_30d = df_close['IDR=X'].dropna().tail(30)
    df_vix = df_close['^VIX'].dropna()
    
    if df_idr_30d.empty or df_vix.empty:
         return False
    
    # 2. Pengecekan VIX (Kondisi ATAU pertama)
    current_vix = float(df_vix.iloc[-1])
    if current_vix > 30:
        print(f"ALERT: VIX index is {current_vix:.2f} (Above 30).")
        return True
    
    # 3. Pengecekan Depresiasi Rupiah (Kondisi ATAU kedua)
    current_idr = float(df_idr_30d.iloc[-1])
    lowest_idr_30d = float(df_idr_30d.min())
    
    persentase_depresiasi = (current_idr - lowest_idr_30d) / lowest_idr_30d
    print(f"Depresiasi IDR 30 Hari: {persentase_depresiasi * 100:.2f}%")
    
    return bool(persentase_depresiasi > 0.05)