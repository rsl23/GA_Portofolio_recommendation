import requests
from datetime import datetime

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
        response = requests.get(endpoint, headers=headers, params=params, timeout=20)
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
    
def fetch_api_stock_summary(length: int = 5000, target_date: str = None):
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
        response = requests.get(endpoint, headers=headers, params=params, timeout=20)
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