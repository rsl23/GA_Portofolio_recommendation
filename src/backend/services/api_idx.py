import requests
from datetime import datetime

import yfinance as yf

from src.backend.core.config import settings

BASE_URL = settings.BASE_URL

def fetch_api_idx_saham(length: int = 1000, boards: list = ["Utama", "Pengembangan"]):
    """
    Mengambil daftar saham dari API ZPI (Bursa Efek Indonesia) dan langsung menyaring papannya.
    """
    endpoint = f"{BASE_URL}/finance:idx/companies"
    
    api_key = settings.X_API_KEY
    if not api_key:
        print("WARNING: x-api-key tidak ditemukan di .env!")

    headers = {
        "x-api-key": api_key,
        "Accept": "application/json"
    }
    
    params = {
        "length": length,
    }
    
    hasil_format = []
    
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=45)
        response.raise_for_status() 
        
        data = response.json()
        
        # Perbaikan: Struktur API ZPI memiliki "data" bersarang (nested)
        # Contoh: {"data": {"provider": "idx", ..., "data": [{...}]}}
        nested_data = data.get('data', {})
        raw_list = nested_data.get('data', []) if isinstance(nested_data, dict) else []
        
        for item in raw_list:
            papan_pencatatan = item.get('PapanPencatatan', '')
            
            if papan_pencatatan not in boards:
                continue # Lewati (skip) jika bukan Utama/Pengembangan
                
            hasil_format.append({
                'Kode': item.get('KodeEmiten', ''), 
                'Nama': item.get('NamaEmiten', ''),
                'TanggalPencatatan': item.get('TanggalPencatatan', ''),
                'PapanPencatatan': papan_pencatatan,
                'Sektor': item.get('Sektor', ''),
                'SubSektor': item.get('SubSektor', ''),
                'Industri': item.get('Industri', ''),
                'SubIndustri': item.get('SubIndustri', ''),
            })
            
        return hasil_format

    except requests.exceptions.RequestException as e:
        print(f"Terjadi kesalahan saat memanggil API IDX: {e}")
        return []
    
def fetch_api_securities(length: int = 1000):
    """
    [SUMBER BARU] Mengambil daftar efek dari endpoint /finance:idx/securities.
    Dipakai sebagai pengganti /finance:idx/stock-summary selama endpoint
    tersebut down (status 10 Okt 2026).

    Field yang dipakai pipeline:
        Code   -> Kode
        Shares -> ListedShares
    Harga Close tidak tersedia di endpoint ini -> lihat _fetch_close_from_yfinance.
    """
    endpoint = f"{BASE_URL}/finance:idx/securities"

    api_key = settings.X_API_KEY
    if not api_key:
        print("WARNING: x-api-key tidak ditemukan di .env!")

    headers = {
        "x-api-key": api_key,
        "Accept": "application/json"
    }

    params = {
        "length": length,
    }

    hasil_format = []

    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=45)
        response.raise_for_status()

        data = response.json()

        # Struktur API ZPI sama dengan endpoint lain:
        # {"data": {"provider": "idx", ..., "data": [{...}]}}
        nested_data = data.get('data', {})
        raw_list = nested_data.get('data', []) if isinstance(nested_data, dict) else []

        for item in raw_list:
            code = item.get('Code', '')
            if not code:
                continue
            hasil_format.append({
                'Kode': code,
                'ListedShares': item.get('Shares', 0),
            })

        return hasil_format

    except requests.exceptions.RequestException as e:
        print(f"Terjadi kesalahan saat memanggil API IDX Securities: {e}")
        return []


def _fetch_close_from_yfinance(codes: list, chunk_size: int = 200) -> dict:
    """
    Ambil harga close terakhir tiap kode saham (.JK) dari yfinance, diunduh
    per batch (chunk_size ticker/request) agar hemat panggilan.

    auto_adjust=False sengaja dipakai agar 'Close' = harga penutupan asli
    (bukan adjusted close), sehingga Market_Cap = ListedShares x Close akurat.

    Returns: {kode: close}; ticker yang gagal diambil di-skip.
    """
    closes: dict = {}
    tickers = [f"{c}.JK" for c in codes if c]
    if not tickers:
        return closes

    total_batch = (len(tickers) + chunk_size - 1) // chunk_size
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            data = yf.download(
                tickers=chunk,
                period="5d",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as e:
            print(f"Terjadi kesalahan saat mengunduh close via yfinance "
                  f"(batch {i // chunk_size + 1}/{total_batch}): {e}")
            continue

        if data is None or len(data) == 0:
            continue

        # MultiIndex (Ticker, Price) bila > 1 ticker; kolom tunggal bila 1 ticker
        multi = getattr(data.columns, 'nlevels', 1) > 1
        for t in chunk:
            try:
                s = data[t]['Close'] if multi else data['Close']
                s = s.dropna()
                if len(s) == 0:
                    continue
                closes[t[:-3]] = float(s.iloc[-1])   # buang sufiks '.JK'
            except Exception:
                continue

    return closes


def fetch_api_stock_summary(length: int = 5000, target_date: str = None):
    """
    Mengambil ringkasan saham harian dari API ZPI (Bursa Efek Indonesia).

    [STATUS 10 Okt 2026] Endpoint /finance:idx/stock-summary sedang DOWN, maka
    yang AKTIF sekarang adalah fallback:
        1. /finance:idx/securities (length=1000) -> Code & Shares
        2. Close terakhir via yfinance (.JK)
    Kontrak keluaran TIDAK berubah: list of dict {Kode, Close, ListedShares},
    sehingga pemanggil (stock_filtering.run_live_preprocessing) tetap sama.

    CARA KEMBALI KE ENDPOINT LAMA (ketika sudah pulih):
        1. Uncomment (hapus tanda # pada) blok "NONAKTIF SEMENTARA:
           ENDPOINT LAMA" di bawah.
        2. Comment blok "AKTIF: FALLBACK".
       Blok ENDPOINT LAMA sengaja diletakkan DI ATAS fallback; bila di-
       uncomment, ia akan mengeksekusi dan return lebih dulu.
    """
    # ============== [NONAKTIF SEMENTARA] ENDPOINT LAMA ====================
    # /finance:idx/stock-summary — aktifkan kembali ketika endpoint pulih.

    """
    Mengambil ringkasan saham harian dari API ZPI (Bursa Efek Indonesia).
    """
   
        
    endpoint = f"{BASE_URL}/finance:idx/stock-summary"
    
    api_key = settings.X_API_KEY
    if not api_key:
        print("WARNING: x-api-key tidak ditemukan di .env!")

    headers = {
        "x-api-key": api_key,
        "Accept": "application/json"
    }
    
    params = {
        "length": length,
    }
    
    if target_date:
        # Default ke hari ini dengan format YYYYMMDD (format umum API ZPI/IDX)
        params["date"] = target_date
    
    hasil_format = []
    
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=45)
        response.raise_for_status() 
        
        data = response.json()
        
        # Perbaikan: Struktur API ZPI memiliki "data" bersarang (nested)
        # Contoh: {"data": {"provider": "idx", ..., "data": [{...}]}}
        nested_data = data.get('data', {})
        raw_list = nested_data.get('data', []) if isinstance(nested_data, dict) else []
        
        for item in raw_list:
            close_price = item.get('Close', 0)
            listed_shares = item.get('ListedShares', 0)
            
            hasil_format.append({
                'Kode': item.get('StockCode', ''),
                'Tanggal': item.get('Date', ''),
                
                # Mesin Valuasi
                'Close': close_price,
                'ListedShares': listed_shares,
                
                # Likuiditas & Volatilitas Dasar
                'Volume_Hari_Ini': item.get('Volume', 0),
                'Value_Hari_Ini': item.get('Value', 0),
                'High': item.get('High', 0),
                'Low': item.get('Low', 0),
                'Frequency': item.get('Frequency', 0),
                
                # Tabungan: Foreign Flow 
                'ForeignBuy': item.get('ForeignBuy', 0),
                'ForeignSell': item.get('ForeignSell', 0)
            })
            
        return hasil_format

    except requests.exceptions.RequestException as e:
        print(f"Terjadi kesalahan saat memanggil API IDX Stock Summary: {e}")
        return []
    # ======================= [AKTIF: FALLBACK] ============================
    # securities (Code & Shares) + yfinance (Close). Comment blok ini ketika
    # blok ENDPOINT LAMA di atas diaktifkan kembali.
    # print("   [i] Endpoint stock-summary nonaktif -> memakai fallback: "
    #       "securities (Code & Shares) + yfinance (Close).")

    # securities = fetch_api_securities(length=1000)
    # if not securities:
    #     print("   [!] Fallback gagal: data securities kosong/tidak terjangkau.")
    #     return []

    # closes = _fetch_close_from_yfinance([row['Kode'] for row in securities])
    # if not closes:
    #     print("   [!] Fallback gagal: tidak ada harga close yang berhasil diambil dari yfinance.")
    #     return []

    # hasil_format = []
    # for row in securities:
    #     close = closes.get(row['Kode'])
    #     if close is None:
    #         continue    # tanpa close, Market_Cap tidak bisa dihitung -> emiten dilewati
    #     hasil_format.append({
    #         'Kode': row['Kode'],
    #         'Close': close,
    #         'ListedShares': row['ListedShares'],
    #     })

    # print(f"   [i] Fallback summary siap: {len(hasil_format)} emiten "
    #       f"(securities: {len(securities)}, close terambil: {len(closes)}).")
    # return hasil_format
