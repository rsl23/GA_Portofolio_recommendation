import os
import math
import re
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
        response = requests.get(endpoint, headers=headers, params=params, timeout=60)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Gagal mengambil fundamental Pluang untuk {code}: {e}")
        return None



def _parse_metric(value, *, percent=False):
    """Pluang display numbers: ROE/yield in percentage points, DER in x.

    Preserve missing or malformed values as NaN; do not invent zero earnings.
    """
    if value is None or isinstance(value, bool):
        return float("nan")
    raw = str(value).strip()
    if not raw or raw.lower() in {"-", "--", "n/a", "na", "null", "none"}:
        return float("nan")
    clean = re.sub(r"^Rp\s*", "", raw, flags=re.IGNORECASE).replace(",", "").strip()
    clean = re.sub(r"[xX%]$", "", clean).strip()
    try:
        number = float(clean)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(number):
        return float("nan")
    return number / 100.0 if percent else number


def parse_pluang_fundamentals(payload):
    """Normalize live Pluang metrics to the existing cache contract."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return None
    data = payload["data"]
    overview = data.get("overview") or {}
    ratios = data.get("ratios") or {}
    if not isinstance(overview, dict) or not isinstance(ratios, dict):
        return None
    def field(group, key):
        section = ratios.get(group)
        return section.get(key) if isinstance(section, dict) else None
    return {
        "eps": _parse_metric(overview.get("eps")),
        "per": _parse_metric(field("valuation", "pe")),
        "pbv": _parse_metric(field("valuation", "pb")),
        "roe": _parse_metric(field("profitability", "roe"), percent=True),
        "der": _parse_metric(field("solvency", "de")),
        "dividend_yield": _parse_metric(field("dividend", "ttm"), percent=True),
    }
