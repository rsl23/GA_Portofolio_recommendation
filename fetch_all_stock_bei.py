"""
Fetch data Bursa Efek Indonesia (BEI) dari API ZPI:
    1. new-listings : https://api.zpi.web.id/v1/finance:idx/new-listings  -> kolom listing_date
    2. delistings   : https://api.zpi.web.id/v1/finance:idx/delistings    -> kolom delisting_date
lalu menyimpannya ke tabel `stock_universe` di database.

Endpoint berjalan per-bulan, jadi skrip ini menggunakan loop for:
    - tahun : 1990 s.d. 2025
    - bulan : 1 s.d. 12
    - length: 200 (maksimum item per halaman)

Dilengkapi penanganan rate-limit (HTTP 429) dengan retry + exponential backoff.

Cara pakai:
    python fetch_all_stock_bei.py
"""

import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.backend.models.database import engine, SessionLocal
from src.backend.models.stock_universe import StockUniverse

# ---------------------------------------------------------------- konfigurasi
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("X_API_KEY")

ENDPOINT = f"{BASE_URL}/finance:idx/new-listings"
DELISTING_ENDPOINT = f"{BASE_URL}/finance:idx/delistings"
PAGE_LENGTH = 200          # length per permintaan
START_YEAR = 1990
END_YEAR = 2025            # inclusive
REQUEST_DELAY = 0.5        # jeda antar request agar tidak membebani server
MAX_RETRIES = 5            # percobaan ulang saat kena rate-limit (429)
BACKOFF_BASE = 2.0         # delay awal untuk exponential backoff (detik)

# --------------------------------------------------------- fungsi-fungsi inti
def _build_headers() -> dict:
    """Header otorisasi yang dibutuhkan API ZPI."""
    if not API_KEY:
        print("WARNING: x-api-key tidak ditemukan di .env!")
    return {
        "x-api-key": API_KEY,
        "Accept": "application/json",
    }


def fetch_new_listings(year: int, month: int, length: int = PAGE_LENGTH) -> list:
    """
    Ambil seluruh item new-listings untuk satu tahun & bulan tertentu.
    Mengikuti pagination (nextPage) hingga data habis (hasMore=False).
    Menangani rate-limit (HTTP 429) dengan retry + exponential backoff.

    Returns: list of dict {'code', 'name', 'listingDate'}.
    """
    headers = _build_headers()
    items = []
    page = 1

    while True:
        params = {"year": year, "month": month, "length": length}
        params["page"] = page

        payload = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.get(ENDPOINT, headers=headers, params=params, timeout=30)

                # Rate-limit (429): tunggu lalu coba lagi dengan backoff
                if resp.status_code == 429:
                    wait = BACKOFF_BASE ** attempt
                    print(f"  [429] year={year} month={month} page={page}: retry dalam {wait:.0f}s (percobaan {attempt+1}/{MAX_RETRIES+1})")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                payload = resp.json()
                break
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE ** attempt
                    print(f"  [ERR] year={year} month={month} page={page} (retry {attempt+1}): {e}")
                    time.sleep(wait)
                else:
                    print(f"  [ERR] year={year} month={month} page={page}: {e}")
                    payload = None

        if payload is None:
            break

        data = payload.get("data", {})
        if not isinstance(data, dict):
            break

        raw_items = data.get("items", []) or []
        items.extend(raw_items)

        has_more = data.get("hasMore", False)
        next_page = data.get("nextPage")
        if has_more and next_page:
            page = next_page
        else:
            break

        time.sleep(REQUEST_DELAY)

    return items


def fetch_delistings(year: int, month: int, length: int = PAGE_LENGTH) -> list:
    """
    Ambil seluruh item delistings untuk satu tahun & bulan tertentu.
    Mengikuti pagination (nextPage) hingga data habis (hasMore=False).
    Menangani rate-limit (HTTP 429) dengan retry + exponential backoff.

    Returns: list of dict {'code', 'name', 'delistingDate', ...}.
    """
    headers = _build_headers()
    items = []
    page = 1

    while True:
        params = {"year": year, "month": month, "length": length}
        params["page"] = page

        payload = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = requests.get(DELISTING_ENDPOINT, headers=headers, params=params, timeout=30)

                # Rate-limit (429): tunggu lalu coba lagi dengan backoff
                if resp.status_code == 429:
                    wait = BACKOFF_BASE ** attempt
                    print(f"  [429] delistings year={year} month={month} page={page}: retry dalam {wait:.0f}s (percobaan {attempt+1}/{MAX_RETRIES+1})")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                payload = resp.json()
                break
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES:
                    wait = BACKOFF_BASE ** attempt
                    print(f"  [ERR] delistings year={year} month={month} page={page} (retry {attempt+1}): {e}")
                    time.sleep(wait)
                else:
                    print(f"  [ERR] delistings year={year} month={month} page={page}: {e}")
                    payload = None

        if payload is None:
            break

        # Respons delistings menempatkan 'items' langsung di level paling atas:
        # { "page", "year", "items", "hasMore", "nextPage", ... }
        # Namun sebagian respons lain membungkusnya di dalam 'data'.
        # Ambil keduanya agar skrip tetap berjalan apa pun bentuknya.
        data = payload if "items" in payload else payload.get("data", {})
        if not isinstance(data, dict):
            break

        raw_items = data.get("items", []) or []
        items.extend(raw_items)

        has_more = data.get("hasMore", False)
        next_page = data.get("nextPage")
        if has_more and next_page:
            page = next_page
        else:
            break

        time.sleep(REQUEST_DELAY)

    return items


def fetch_all_delistings(start_year: int = START_YEAR, end_year: int = END_YEAR) -> list:
    """
    Loop seluruh kombinasi tahun (start s.d. end) dan bulan (1 s.d. 12),
    lalu gabungkan semua item menjadi satu daftar unik berdasarkan 'code'.
    """
    records = {}
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            items = fetch_delistings(year, month)
            if not items:
                continue

            for item in items:
                code = item.get("code")
                if not code:
                    continue
                records[code] = {
                    "ticker": code,
                    "nama_perusahaan": item.get("name", ""),
                    "listingDate": item.get("listingDate", ""),
                    "delistingDate": item.get("delistingDate", ""),
                }

            print(f"  {year}-{month:02d}: {len(items)} delisting (total unik {len(records)})")
            time.sleep(REQUEST_DELAY)

    return list(records.values())


def fetch_all_new_listings(start_year: int = START_YEAR, end_year: int = END_YEAR) -> list:
    """
    Loop seluruh kombinasi tahun (start s.d. end) dan bulan (1 s.d. 12),
    lalu gabungkan semua item menjadi satu daftar unik berdasarkan 'code'.
    """
    records = {}
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            items = fetch_new_listings(year, month)
            if not items:
                continue

            for item in items:
                code = item.get("code")
                if not code:
                    continue
                records[code] = {
                    "ticker": code,
                    "nama_perusahaan": item.get("name", ""),
                    "listingDate": item.get("listingDate", ""),
                }

            print(f"  {year}-{month:02d}: {len(items)} listing (total unik {len(records)})")
            time.sleep(REQUEST_DELAY)

    return list(records.values())


def parse_date(raw: str):
    """Konversi 'YYYY-MM-DD' menjadi objek date, None jika tidak valid."""
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def save_to_database(records: list) -> None:
    """
    Simpan (upsert) daftar saham ke tabel stock_universe.
    Menggunakan db.merge() berdasarkan primary key (ticker), sehingga aman
    dijalankan berulang kali tanpa menimbulkan duplikat.
    """
    # Pastikan tabel ada (aman jika masih kosong di database)
    StockUniverse.__table__.create(engine, checkfirst=True)

    db = SessionLocal()
    saved = 0
    skipped = 0

    try:
        for rec in records:
            listing_date = parse_date(rec.get("listingDate"))
            if listing_date is None:
                skipped += 1
                continue

            stock = StockUniverse(
                ticker=rec["ticker"],
                nama_perusahaan=rec.get("nama_perusahaan", ""),
                listing_date=listing_date,
                delisting_date=None,
                relisting_date=None,
            )
            db.merge(stock)
            saved += 1

        db.commit()
        print(f"\nBerhasil menyimpan {saved} saham (lewat {skipped} karena listingDate tidak valid).")
    except Exception as e:
        db.rollback()
        print(f"[ERR] Gagal menyimpan ke database: {e}")
        raise
    finally:
        db.close()


def save_delistings_to_database(records: list) -> None:
    """
    Masukkan ticker delisting ke tabel stock_universe.

    - Ticker yang SUDAH ada di tabel -> hanya colom delisting_date yg di-update
      (tidak memakai db.merge() supaya listing_date/nama_perusahaan yg sudah
      ada tidak tertimpa nilai None).
    - Ticker yang BELUM ada di tabel  -> di-insert sebagai baris baru
      (listing_date diambil dari respons API; bila kosong, pakai delisting_date
      agar kolom listing_date (nullable=False) tetap terisi).
    """
    # Pastikan tabel ada (aman jika masih kosong di database)
    StockUniverse.__table__.create(engine, checkfirst=True)

    db = SessionLocal()
    updated = 0
    inserted = 0
    skipped = 0

    try:
        for rec in records:
            ticker = rec.get("ticker")
            delisting_date = parse_date(rec.get("delistingDate"))
            if not ticker or delisting_date is None:
                skipped += 1
                continue

            stock = db.query(StockUniverse).filter(StockUniverse.ticker == ticker).first()

            if stock is None:
                # Ticker belum terdaftar -> buat baris baru.
                listing_date = parse_date(rec.get("listingDate")) or delisting_date
                stock = StockUniverse(
                    ticker=ticker,
                    nama_perusahaan=rec.get("nama_perusahaan", ""),
                    listing_date=listing_date,
                    delisting_date=delisting_date,
                    relisting_date=None,
                )
                db.add(stock)
                inserted += 1
            else:
                # Ticker sudah ada -> update delisting_date saja.
                stock.delisting_date = delisting_date
                updated += 1

        db.commit()
        print(f"\nBerhasil memperbarui delisting_date untuk {updated} saham, "
              f"insert {inserted} saham baru (hanya {skipped} yang dilewati "
              f"karena ticker/delistingDate tidak valid).")
    except Exception as e:
        db.rollback()
        print(f"[ERR] Gagal simpan delisting ke database: {e}")
        raise
    finally:
        db.close()


def main(start_year: int = START_YEAR, end_year: int = END_YEAR) -> None:
    """Orkestrasi utama: fetch data new-listings & delistings lalu simpan ke database."""
    # 1) New-listings
    print(f"Mengambil seluruh new-listings BEI dari tahun {start_year} s.d. {end_year}...")

    # records = fetch_all_new_listings(start_year, end_year)
    # print(f"\nTotal saham unik yang ditemukan: {len(records)}")

    # if records:
        # save_to_database(records)

    # 2) Delistings
    print(f"\nMengambil seluruh delistings BEI dari tahun {start_year} s.d. {end_year}...")

    delisted = fetch_all_delistings(start_year, end_year)
    print(f"\nTotal ticker yang mengalami delisting: {len(delisted)}")

    if delisted:
        save_delistings_to_database(delisted)

    # Ringkasan akhir
    db = SessionLocal()
    try:
        total = db.query(StockUniverse).count()
        print(f"Total baris di tabel stock_universe sekarang: {total}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
