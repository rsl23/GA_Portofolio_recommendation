import os
from pathlib import Path
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load variabel dari file .env
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("BASE_URL")

def fetch_bi_rate():
    """Mengambil data BI rate terbaru dari API ZAPI"""
    endpoint = f"{BASE_URL}/finance:bi-kurs/policy-rate"
    
    api_key = os.getenv("X_API_KEY")
    
    if not api_key:
        print("WARNING: x-api-key tidak ditemukan di .env!")
    
    headers = {
        "x-api-key": api_key,
        "Accept" : "application/json"
    }
    
    try:
        response = requests.get(endpoint, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json().get('data', {}).get('items', [])
        latest = max(data, key=lambda x: x.get("date", "")) if data else None

        rate = latest.get("ratePercent") if latest else None
        if rate is None:
            return None

        try:
            rate = float(rate)
        except (TypeError, ValueError):
            return None

        # API ZAPI mengembalikan suku bunga dalam bentuk PERSEN (mis. 5.75 -> 5.75%).
        # GA membutuhkan fraksi desimal (contoh: 0.0575), jadi bagi 100 bila belum.
        # Guard "rate > 1" menjaga kalau nilainya ternyata sudah berupa fraksi (0.0575).
        return rate / 100.0 if rate > 1 else rate    
    except requests.RequestException as e:
        print(f"Gagal mengambil data BI rate: {e}")
        return None
    