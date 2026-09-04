import os
import requests
from dotenv import load_dotenv

# Load variabel dari file .env
load_dotenv()

BASE_URL = os.getenv("BASE_URL")

def fetch_api_pluang_fundamentals(code: str):
    """
    Mengambil data fundamental lengkap dari API ZPI dengan endpoint Pluang.
    Menerima parameter 'code' (misal: 'BBCA').
    """
    endpoint = f"{BASE_URL}/finance:pluang/fundamentals"
    
    api_key = os.getenv("X_API_KEY") 
    if not api_key:
        print("WARNING: x-api-key tidak ditemukan di .env!")

    headers = {
        "x-api-key": api_key,
        "Accept": "application/json"
    }
    
    params = {
        "code": code
    }
    
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Gagal mengambil fundamental Pluang untuk {code}: {e}")
        return None

