from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.backend.services.api_idx import fetch_api_idx_saham, fetch_api_stock_summary
from src.backend.models.database import SessionLocal
from src.backend.models.stock_universe import StockUniverse


# ============================================================
# THRESHOLD MARKET CAP & SKALASINYA PADA MODE BACKTEST
# ============================================================
# Threshold Rp 5 Triliun ditetapkan berdasarkan kondisi pasar SAAT INI,
# sehingga tidak adil bila dipakai apa adanya untuk tahun-tahun lama
# (total kapitalisasi pasar IDX pada 2006 jauh lebih kecil daripada 2026).
# Pada mode backtest threshold diskalakan mengikuti total kapitalisasi
# pasar IDX pada tanggal pengujian:
#
#     threshold(date) = (IDX_MktCap(date) / IDX_MktCap(tahun_acuan)) x 5T
#
# Contoh: IDX_MktCap(2026) = 10.923.071,54 miliar IDR (baris tahun 2026,
# Month = NA, pada 'data/idx_activities.parquet'). Bila backtest dilakukan
# pada 2015 dengan IDX_MktCap(2015) = 4.872.702 miliar IDR, maka threshold
# menjadi (4.872.702 / 10.923.071,54) x Rp 5T = Rp 2,231 T.
#
# Sumber: 'data/idx_activities.parquet' (hasil transformasi Idx_Activities.xlsx,
# lihat idx_activities_to_parquet.py), kolom 'Market_Cap_bIDR' (miliar IDR).
# ============================================================

MARKET_CAP_THRESHOLD_NOW = 5_000_000_000_000   # Rp 5 Triliun (basis hari ini)
MARKET_CAP_BASE_YEAR = 2026                    # tahun acuan penetapan Rp 5T
IDX_ACTIVITIES_PARQUET = 'data/idx_activities.parquet'

# ============================================================
# THRESHOLD ADTV & SKALASINYA PADA MODE BACKTEST
# ============================================================
# Threshold Rp 1 Miliar/hari juga ditetapkan pada kondisi likuiditas pasar
# SAAT INI. Pada mode backtest threshold diskalakan mengikuti likuiditas IHSG
# pada tanggal pengujian:
#
#     threshold(date) = (ADTV_IHSG(date) / ADTV_IHSG(sekarang)) x Rp 1 Miliar
#
# ADTV IHSG (per hari bursa) dihitung dari 3 BULAN TERAKHIR sebelum bulan
# tanggal acuan, dengan langkah:
#   1. jumlah hari bursa bulan tsb = Total_Trading_Value_bIDR
#                                    / Avg_Daily_Trading_Value_bIDR
#   2. ADTV IHSG = sigma(Avg_Daily_Trading_Value_bIDR x hari bursa)
#                  / sigma(hari bursa)
#
# Sumber: 'data/idx_activities.parquet', baris bulanan (Period_Type='Month').
# ============================================================

ADTV_THRESHOLD_NOW = 1_000_000_000              # Rp 1 Miliar/hari (basis hari ini)
IDX_ADTV_LOOKBACK_MONTHS = 3                    # jumlah bulan rata-rata ADTV IHSG
_IDX_ACTIVITY_CACHE: dict[tuple[str, float], pd.DataFrame] = {}
_NAMA_BULAN = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',
               'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']

# Cache sederhana: {(path, mtime): DataFrame} agar file parquet tidak dibaca
# berulang, tetapi tetap ikut ter-update bila file berubah.
_IDX_MCAP_CACHE: dict[tuple[str, float], pd.DataFrame] = {}


def _format_triliun(value_idr: float) -> str:
    """Format nilai IDR ke satuan triliun, mis. 5e12 -> 'Rp 5,000 T'."""
    return f"Rp {value_idr / 1_000_000_000_000:,.3f} T"


def _load_idx_market_cap(path: str = IDX_ACTIVITIES_PARQUET) -> pd.DataFrame | None:
    """
    Muat total kapitalisasi pasar IDX dari parquet aktivitas IDX.

    Returns:
        DataFrame kolom ['Year', 'Month', 'Market_Cap_IDR'] (IDR penuh),
        atau None bila file tidak ada / gagal dibaca.
        Baris tahun (agregat 1 tahun) memiliki Month = <NA>.
    """
    p = Path(path)
    if not p.exists():
        print(f"   [!] {path} tidak ditemukan -> threshold market cap "
              f"memakai nilai default {_format_triliun(MARKET_CAP_THRESHOLD_NOW)}.")
        return None

    try:
        key = (str(p), p.stat().st_mtime)
        if key in _IDX_MCAP_CACHE:
            return _IDX_MCAP_CACHE[key]

        df = pd.read_parquet(p, columns=['Year', 'Month', 'Market_Cap_bIDR'])
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['Year', 'Market_Cap_bIDR']).copy()

        # 'Market Capt., b.IDR' -> satuan miliar IDR; kalikan 1e9 agar setara
        # dengan Market_Cap = ListedShares x Close yang dihitung di bawah.
        df['Market_Cap_IDR'] = df['Market_Cap_bIDR'] * 1_000_000_000

        df = (
            df[['Year', 'Month', 'Market_Cap_IDR']]
            .sort_values(['Year', 'Month'], na_position='first')
            .reset_index(drop=True)
        )
        _IDX_MCAP_CACHE[key] = df
        return df
    except Exception as exc:
        print(f"   [!] Gagal membaca {path} ({exc}) -> threshold market cap "
              f"memakai nilai default {_format_triliun(MARKET_CAP_THRESHOLD_NOW)}.")
        return None


def get_index_market_cap(
    date_ref: date,
    path: str = IDX_ACTIVITIES_PARQUET,
) -> tuple[float | None, str | None]:
    """
    Ambil total kapitalisasi pasar IDX (IDR penuh) yang relevan untuk `date_ref`.

    Urutan prioritas:
    1. Baris TAHUN pada tahun `date_ref` (Month = <NA>). Ini mengikuti cara
       basis Rp 5T ditetapkan (mis. 2026 -> 10.923.071,54 miliar IDR).
    2. Baris BULAN terakhir pada tahun yang sama dengan Month <= bulan date_ref
       (dipakai bila baris tahunnya belum tersedia).
    3. Baris TAHUN terakhir sebelum tahun date_ref (data tahun tsb belum ada).

    Returns:
        (market_cap_idr, label) atau (None, None) bila tidak tersedia.
    """
    if date_ref is None:
        raise ValueError("Skalasi market cap memerlukan parameter 'date' yang valid.")

    df = _load_idx_market_cap(path)
    if df is None or df.empty:
        return None, None

    year = int(date_ref.year)

    annual = df[(df['Year'] == year) & (df['Month'].isna())]
    if not annual.empty:
        return float(annual['Market_Cap_IDR'].iloc[-1]), f"tahun {year}"

    monthly = df[(df['Year'] == year) &
                 (df['Month'].notna()) &
                 (df['Month'] <= int(date_ref.month))]
    if not monthly.empty:
        row = monthly.iloc[-1]
        return float(row['Market_Cap_IDR']), f"bulan {int(row['Month'])} {year}"

    annual_prev = df[(df['Year'] < year) & (df['Month'].isna())]
    if not annual_prev.empty:
        row = annual_prev.iloc[-1]
        return float(row['Market_Cap_IDR']), f"tahun {int(row['Year'])}"

    monthly_prev = df[(df['Year'] < year) & (df['Month'].notna())]
    if not monthly_prev.empty:
        row = monthly_prev.iloc[-1]
        return float(row['Market_Cap_IDR']), f"bulan {int(row['Month'])} {int(row['Year'])}"

    return None, None


def get_scaled_market_cap_threshold(
    date_ref: date,
    base_threshold: float = MARKET_CAP_THRESHOLD_NOW,
    base_year: int = MARKET_CAP_BASE_YEAR,
    path: str = IDX_ACTIVITIES_PARQUET,
) -> dict:
    """
    Hitung threshold market cap yang sudah diskalakan untuk tanggal backtest.

        threshold(date) = (IDX_MktCap(date) / IDX_MktCap(base_year)) x base_threshold

    Returns:
        dict berisi:
          threshold         : nominal threshold (IDR) yang dipakai
          scale             : faktor pengali terhadap base_threshold
          base_year         : tahun acuan penetapan threshold
          base_label        : label baris IDX yang dipakai sebagai basis
          base_market_cap   : total market cap IDX tahun acuan (IDR)
          target_label      : label baris IDX untuk tanggal backtest
          target_market_cap : total market cap IDX pada tanggal backtest (IDR)
          scaled            : True bila skalasi berhasil diterapkan

    Bila data parquet tidak tersedia/kosong, fungsi mengembalikan threshold
    default (Rp 5T) dengan scaled=False agar pipeline tidak gagal berjalan.
    """
    info = {
        'threshold': float(base_threshold),
        'scale': 1.0,
        'base_year': base_year,
        'base_label': None,
        'base_market_cap': None,
        'target_label': None,
        'target_market_cap': None,
        'scaled': False,
    }

    base_mcap, base_label = get_index_market_cap(date(base_year, 12, 31), path)
    info['base_label'] = base_label
    info['base_market_cap'] = base_mcap

    if base_mcap is None or base_mcap <= 0:
        print(f"   [!] Market cap IDX tahun acuan {base_year} tidak tersedia -> "
              f"threshold tetap {_format_triliun(base_threshold)}.")
        return info

    if base_label != f"tahun {base_year}":
        print(f"   [!] Data market cap IDX tahun acuan {base_year} tidak ada, "
              f"dipakai data {base_label} sebagai basis.")
        base_year = int(str(base_label).split()[-1])
        info['base_year'] = base_year

    target_mcap, target_label = get_index_market_cap(date_ref, path)
    info['target_label'] = target_label
    info['target_market_cap'] = target_mcap

    if target_mcap is None or target_mcap <= 0:
        print(f"   [!] Market cap IDX untuk {date_ref} tidak tersedia -> "
              f"threshold tetap {_format_triliun(base_threshold)}.")
        return info

    scale = target_mcap / base_mcap
    info['scale'] = scale
    info['threshold'] = float(base_threshold) * scale
    info['scaled'] = True
    return info


def _format_miliar(value_idr: float) -> str:
    """Format nilai IDR ke satuan miliar, mis. 1e9 -> 'Rp 1,000 M'."""
    return f"Rp {value_idr / 1_000_000_000:,.3f} M"


def _load_idx_monthly_activity(path: str = IDX_ACTIVITIES_PARQUET) -> pd.DataFrame | None:
    """
    Muat aktivitas perdagangan IDX baris BULANAN dari parquet.

    Returns:
        DataFrame kolom ['Year', 'Month', 'Total_Trading_Value_bIDR',
        'Avg_Daily_Trading_Value_bIDR', 'Trading_Days'] terurut kronologis,
        atau None bila file tidak ada / gagal dibaca.
        Baris tahunan (Month = <NA>) dibuang karena ADTV dihitung per bulan.
    """
    p = Path(path)
    if not p.exists():
        print(f"   [!] {path} tidak ditemukan -> ADTV IHSG tidak dapat dihitung.")
        return None

    try:
        key = (str(p), p.stat().st_mtime)
        if key in _IDX_ACTIVITY_CACHE:
            return _IDX_ACTIVITY_CACHE[key]

        df = pd.read_parquet(p, columns=[
            'Year', 'Month', 'Total_Trading_Value_bIDR',
            'Avg_Daily_Trading_Value_bIDR', 'Trading_Days',
        ])
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['Year']).copy()
        # Hanya baris bulanan yang dipakai (baris tahunan Month = <NA>)
        df = df[df['Month'].notna()].copy()
        df = df.sort_values(['Year', 'Month']).reset_index(drop=True)
        _IDX_ACTIVITY_CACHE[key] = df
        return df
    except Exception as exc:
        print(f"   [!] Gagal membaca {path} ({exc}) -> ADTV IHSG tidak dapat dihitung.")
        return None


def get_idx_adtv(
    date_ref: date,
    lookback_months: int = IDX_ADTV_LOOKBACK_MONTHS,
    path: str = IDX_ACTIVITIES_PARQUET,
) -> dict:
    """
    Hitung ADTV IHSG (IDR/hari bursa) dari `lookback_months` bulan terakhir
    SEBELUM bulan `date_ref`.

    Contoh: date_ref = 2024-06-28 -> bulan yang dipakai Mei, Apr, Mar 2024.
    Bila salah satu bulan target tidak ada di parquet (mis. bulan berjalan
    belum terbit), bulan itu digantikan bulan terdekat sebelumnya sehingga
    jumlah bulan yang dirata-ratakan tetap `lookback_months`.

    Rumus per bulan:
        hari_bursa = Total_Trading_Value_bIDR / Avg_Daily_Trading_Value_bIDR
    Rumus gabungan:
        ADTV = sigma(Avg_Daily_Trading_Value_bIDR x hari_bursa)
               / sigma(hari_bursa)

    Returns:
        dict: adtv (IDR/hari), months (detail per bulan), label;
        adtv = None bila data tidak tersedia.
    """
    info = {'adtv': None, 'months': [], 'label': None}

    if date_ref is None:
        raise ValueError("Perhitungan ADTV IHSG memerlukan parameter 'date' yang valid.")

    df = _load_idx_monthly_activity(path)
    if df is None or df.empty:
        return info

    # Peta (tahun, bulan) -> baris, agar pencarian bulan O(1)
    avail = {(int(r.Year), int(r.Month)): r for r in df.itertuples(index=False)}

    # Kumpulkan bulan-bulan target (mundur dari bulan sebelum date_ref)
    y, m = int(date_ref.year), int(date_ref.month)
    used: list[tuple[int, int]] = []
    scanned = 0
    max_scan = 12 + lookback_months    # batas aman bila data bolong-bolong
    while len(used) < lookback_months and scanned < max_scan:
        m -= 1
        if m == 0:
            y -= 1
            m = 12
        scanned += 1
        if (y, m) in avail:
            used.append((y, m))

    if not used:
        print(f"   [!] Tidak ada data aktivitas IDX sebelum {date_ref} -> "
              f"ADTV IHSG tidak dapat dihitung.")
        return info

    total_value = 0.0    # sigma(Avg_Daily x hari bursa), dalam IDR
    total_days = 0.0     # sigma(hari bursa)
    months_info = []

    for (yy, mm) in used:
        row = avail[(yy, mm)]
        total_b = float(row.Total_Trading_Value_bIDR or 0)
        avg_b = float(row.Avg_Daily_Trading_Value_bIDR or 0)
        days_col = float(row.Trading_Days or 0)
        days = days_col       

        if days <= 0 or avg_b <= 0:
            continue

        avg_idr = avg_b * 1_000_000_000
        total_value += avg_idr * days
        total_days += days
        months_info.append({
            'year': yy,
            'month': mm,
            'label': f"{_NAMA_BULAN[mm]} {yy}",
            'avg_daily_idr': avg_idr,
            'trading_days': days,
        })

    if total_days <= 0:
        return info

    info['adtv'] = total_value / total_days
    info['months'] = months_info
    info['label'] = ' - '.join(mi['label'] for mi in reversed(months_info))
    return info


def get_scaled_adtv_threshold(
    date_ref: date,
    base_threshold: float = ADTV_THRESHOLD_NOW,
    now_date: date | None = None,
    path: str = IDX_ACTIVITIES_PARQUET,
) -> dict:
    """
    Hitung threshold ADTV yang sudah diskalakan untuk tanggal backtest.

        threshold(date) = (ADTV_IHSG(date) / ADTV_IHSG(sekarang)) x base_threshold

    ADTV_IHSG(sekarang) dihitung dengan rumus yang sama (3 bulan terakhir
    sebelum bulan `now_date`, default tanggal hari ini).

    Returns:
        dict berisi: threshold, scale, base_label, base_adtv, target_label,
        target_adtv, dan scaled (True bila skalasi berhasil).
        Bila data tidak tersedia, threshold default (Rp 1 Miliar) dipakai
        dengan scaled=False agar pipeline tidak gagal berjalan.
    """
    info = {
        'threshold': float(base_threshold),
        'scale': 1.0,
        'base_label': None,
        'base_adtv': None,
        'target_label': None,
        'target_adtv': None,
        'scaled': False,
    }

    if date_ref is None:
        raise ValueError("Skalasi ADTV memerlukan parameter 'date' yang valid.")

    now_ref = now_date or date.today()
    base_info = get_idx_adtv(now_ref, path=path)
    target_info = get_idx_adtv(date_ref, path=path)

    info['base_label'] = base_info['label']
    info['base_adtv'] = base_info['adtv']
    info['target_label'] = target_info['label']
    info['target_adtv'] = target_info['adtv']

    base_adtv = base_info['adtv']
    target_adtv = target_info['adtv']

    if base_adtv is None or base_adtv <= 0:
        print(f"   [!] ADTV IHSG acuan ({now_ref}) tidak tersedia -> "
              f"threshold ADTV tetap {_format_miliar(base_threshold)}.")
        return info

    if target_adtv is None or target_adtv <= 0:
        print(f"   [!] ADTV IHSG untuk {date_ref} tidak tersedia -> "
              f"threshold ADTV tetap {_format_miliar(base_threshold)}.")
        return info

    scale = target_adtv / base_adtv
    info['scale'] = scale
    info['threshold'] = float(base_threshold) * scale
    info['scaled'] = True
    return info


def sync_stock_universe(df_companies: pd.DataFrame) -> tuple[int, int]:
    """
    Pastikan semua emiten hasil fetch API IDX terdaftar di tabel stock_universe.
    - Ticker yang belum ada -> di-INSERT (listing_date dari TanggalPencatatan,
      nama_perusahaan dari Nama).
    - Ticker yang sudah ada -> dibiarkan (tidak di-update).
    Returns: (jumlah_baru, jumlah_sudah_ada)
    """
    if df_companies.empty:
        return 0, 0

    tickers_api = df_companies['Kode'].astype(str).str.strip().tolist()

    db = SessionLocal()
    inserted = 0
    skipped = 0
    try:
        # Satu query: ticker mana saja yang sudah terdaftar
        existing = {
            t for (t,) in (
                db.query(StockUniverse.ticker)
                .filter(StockUniverse.ticker.in_(tickers_api))
                .all()
            )
        }

        for _, row in df_companies.iterrows():
            ticker = str(row['Kode']).strip()
            if not ticker or ticker in existing:
                skipped += 1
                continue

            # TanggalPencatatan -> listing_date (boleh None bila API kosong,
            # kolom didefinisikan nullable=False -> fallback None ditolak DB)
            listing = row.get('TanggalPencatatan')
            if isinstance(listing, str) and listing:
                try:
                    listing = pd.to_datetime(listing).date()
                except (ValueError, TypeError):
                    listing = None

            db.add(StockUniverse(
                ticker=ticker,
                nama_perusahaan=str(row.get('Nama', ''))[:100],
                listing_date=listing,
            ))
            existing.add(ticker)
            inserted += 1

        if inserted:
            db.commit()
            print(f"   [+] Sync stock_universe: {inserted} emiten baru ditambahkan, {skipped} sudah terdaftar.")
        else:
            print(f"   [OK] Sync stock_universe: semua {skipped} emiten sudah terdaftar.")
        return inserted, skipped
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def get_active_tickers_on(date_ref: date) -> list[tuple[str, str | None]]:
    """
    Ambil daftar ticker yang valid (aktif & cukup umur) pada `date_ref` tertentu.
    Dipakai pada mode BACKTEST, di mana data API IDX hari ini tidak relevan.

    Kriteria dari tabel stock_universe:
    1. delisting_date IS NULL      -> emiten masih tercatat aktif
    2. date_ref > listing_date     -> emiten sudah tercatat pada tanggal tersebut
    3. Usia minimal 2 tahun        -> (date_ref - listing_date) >= 730 hari;
                                      emiten yang lebih muda dieliminasi.
    4. papan_pencatatan harus salah satu dari:
                                      Utama / Pengembangan / Akselerasi.

    Returns: list tuple (ticker, sektor).
    """
    if date_ref is None:
        raise ValueError("Mode backtest memerlukan parameter 'date' yang valid.")

    db = SessionLocal()
    try:
        rows = (
            db.query(StockUniverse.ticker, StockUniverse.sektor, StockUniverse.listing_date)
            .filter(
                StockUniverse.delisting_date.is_(None),
                StockUniverse.listing_date < date_ref,
            )
            .all()
        )
    finally:
        db.close()

    usia_min = timedelta(days=730)  # 2 tahun
    eligible = []
    failed_young = []
    for ticker, sektor, listing_date in rows:
        if listing_date is None:
            continue
        # Normalisasi: beberapa koneksi DB mengembalikan Date sebagai string
        if isinstance(listing_date, str):
            try:
                listing_date = datetime.strptime(listing_date[:10], '%Y-%m-%d').date()
            except ValueError:
                continue
        if (date_ref - listing_date) < usia_min:
            failed_young.append(ticker)
            continue
        eligible.append((ticker, sektor, listing_date))

    if failed_young:
        print(f"   [!] Backtest: Gagal Usia IPO < 2 Tahun pada {date_ref} "
              f"({len(failed_young)} saham) -> Contoh: {failed_young[:7]}...")

    print(f"   [OK] Backtest: {len(eligible)} emiten aktif & berusia >= 2 tahun pada {date_ref}.")
    return eligible

def get_fiscal_period(date_ref: date) -> tuple[int, int]:
    """
    Petakan tanggal backtest ke periode laporan fundamental yang relevan.
    Returns: (tahun, kuartal)
      - 1 Jan - 30 Apr   -> Q3 tahun sebelumnya
      - 1 Mei - 31 Agu   -> Q4 tahun sebelumnya (laporan tahunan)
      - 1 Sep - 30 Nov   -> Q1 tahun berjalan
      - 1 Des - 31 Des   -> Q2 tahun berjalan
    """
    if date_ref.month <= 4:
        return date_ref.year - 1, 3
    if date_ref.month <= 8:
        return date_ref.year - 1, 4
    if date_ref.month <= 11:
        return date_ref.year, 1
    return date_ref.year, 2


def load_backtest_data(date_ref: date):
    """
    Muat sumber data BACKTEST dari parquet lokal, mengembalikan objek
    dengan bentuk yang sama seperti sumber data live:
      df_companies : Kode, Sektor, TanggalPencatatan   (pengganti API IDX)
      df_summary   : Kode, Close, ListedShares          (pengganti API summary)
      df_tech      : MultiIndex kolom (Ticker, field)   (pengganti yf.download)
      df_fund      : EPS, ROE, DER per Kode             (pengganti yfinance info)
    """
    if date_ref is None:
        raise ValueError("Mode backtest memerlukan parameter 'date' yang valid.")

    # --- Universe valid per tanggal backtest (aktif, umur >= 2 th, papan valid) ---
    rows_valid = get_active_tickers_on(date_ref)  # list (ticker, sektor, listing_date)
    sektor_db = {t: s for t, s, _ in rows_valid}
    listing_db = {t: l for t, _, l in rows_valid}
    tickers_valid = set(sektor_db.keys())

    # --- df_companies (pengganti API IDX companies) ---
    df_companies = pd.DataFrame({'Kode': sorted(tickers_valid)})
    df_companies['Sektor'] = df_companies['Kode'].map(sektor_db)
    df_companies['TanggalPencatatan'] = pd.to_datetime(df_companies['Kode'].map(listing_db))

    # --- OHLCV historis (pengganti yf.download, window 120 hari) ---
    print("   [i] Backtest: memuat Master_OHLCV_15Tahun.parquet...")
    df_ohlcv = pd.read_parquet('data/Master_OHLCV_15Tahun.parquet')
    df_ohlcv['Date'] = pd.to_datetime(df_ohlcv['Date'])
    df_ohlcv['Ticker'] = df_ohlcv['Ticker'].astype(str).str.replace('.JK', '', regex=False)

    ts_ref = pd.Timestamp(date_ref)
    window = df_ohlcv[
        (df_ohlcv['Date'] <= ts_ref) &
        (df_ohlcv['Date'] > ts_ref - timedelta(days=120))
    ]
    window = window[window['Ticker'].isin(tickers_valid)]

    # Pivot ke struktur MultiIndex (Ticker, field) seperti yf.download(group_by='ticker')
    df_tech = (
        window.pivot(index='Date', columns='Ticker', values=['Close', 'Volume'])
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )

    # --- df_summary (pengganti API summary): Close terakhir + ListedShares ---
    window = window.sort_values(['Ticker', 'Date'])
    last_close = window.groupby('Ticker')['Close'].last()

    # --- Fundamental kuartalan (pengganti yfinance .info) ---
    print("   [i] Backtest: memuat fundamental_quarterly.parquet...")
    df_fq = pd.read_parquet('data/fundamental_quarterly.parquet')

    fy, fq = get_fiscal_period(date_ref)
    period_str = f"Q{fq} {fy}"

    def _period_key(s: str):
        q, y = s.split()
        return (int(y), int(q[1:]))

    target_key = (fy, fq)
    df_fq_keys = df_fq['Fiscal_Period'].map(_period_key)
    mask_past = df_fq_keys <= target_key

    # Data periode terpilih saja (untuk ROE, DER, dll.)
    df_p = df_fq[(df_fq['Fiscal_Period'] == period_str) & df_fq['Ticker'].isin(tickers_valid)]
    piv = df_p.pivot_table(index='Ticker', columns='Metric', values='Value', aggfunc='last')

    # --- EPS TTM: jumlahkan 12 bulan EPS terakhir (termasuk periode terpilih) ---
    # Menangani dua gaya pelaporan:
    #   a) Kuartalan   : Q1-Q4 tersedia -> tiap kuartal meng-cover 1 periode.
    #   b) Hanya Q4/annual : bila ticker TIDAK punya Q1-Q3 pada tahun yang sama,
    #      nilai Q4 dianggap EPS setahun penuh dan meng-cover 4 periode sekaligus
    #      (sehingga tidak terjadi penghitungan ganda, dan window tetap 12 bulan).
    df_eps = df_fq[mask_past & df_fq['Ticker'].isin(tickers_valid) &
                   (df_fq['Metric'] == 'EPS (Quarter)')].copy()
    df_eps['_key'] = df_fq_keys[mask_past].reindex(df_eps.index)

    def _prev_quarter(key: tuple[int, int]) -> tuple[int, int]:
        y, q = key
        return (y, q - 1) if q > 1 else (y - 1, 4)

    eps_dict = {}
    for tkr, grp in df_eps.groupby('Ticker'):
        eps_dict[tkr] = dict(zip(grp['_key'], grp['Value']))

    eps_rows = {}
    for tkr, kv in eps_dict.items():
        # Tahun bergaya annual: punya Q4, tapi tidak ada satu pun Q1/Q2/Q3 tahun itu
        annual_years = {
            y for (y, q) in kv
            if q == 4 and not any((y, qq) in kv for qq in (1, 2, 3))
        }
        ttm = 0.0
        counted = 0
        cur = target_key
        while counted < 4:
            if cur not in kv:
                cur = _prev_quarter(cur)
                counted += 1
                continue
            y, q = cur
            ttm += float(kv[cur])
            if q == 4 and y in annual_years:
                counted += 4            # Q4 annual meng-cover setahun penuh
                cur = (y - 1, 4)
            else:
                counted += 1
                cur = _prev_quarter(cur)
        eps_rows[tkr] = ttm

    eps = pd.Series(eps_rows, name='Value')
    eps.index.name = 'Ticker'

    # ROE & DER tetap memakai kuartal terpilih saja
    roe = piv.get('Return on Equity (Quarter)')
    total_debt = piv.get('Total Debt (Quarter)')
    total_equity_q = piv.get('Total Equity (Quarter)')
    total_equity_non_q = piv.get('Total Equity')

    if total_equity_q is None:
        total_equity = total_equity_non_q
    elif total_equity_non_q is None:
        total_equity = total_equity_q
    else:
        total_equity = total_equity_q.combine_first(total_equity_non_q)

    # DER diskalakan x100 agar setara dengan 'debtToEquity' yfinance (persen),
    # karena filter yang dipakai adalah DER < 200.
    der = (total_debt / total_equity * 100).replace([float('inf'), float('-inf')], None)
    

    # Share Outstanding: periode terpilih dulu; jika kosong, pakai periode
    # terbaru yang tersedia SEBELUM tanggal backtest (papan market cap).
    shares = piv.get('Share Outstanding')

    df_past = df_fq[mask_past & df_fq['Ticker'].isin(tickers_valid) &
                    (df_fq['Metric'] == 'Share Outstanding')].copy()
    df_past['_key'] = df_fq_keys[mask_past].reindex(df_past.index)
    shares_latest = df_past.sort_values('_key').groupby('Ticker')['Value'].last()
    if shares is None:
        shares = shares_latest
    else:
        shares = shares.fillna(shares_latest)

    # --- Rakit df_summary & df_fund dengan index 'Kode' ---
    df_summary = pd.DataFrame({
        'Close': last_close,
        'ListedShares': shares,
    }).dropna(subset=['Close']).reset_index().rename(columns={'Ticker': 'Kode'})

    df_fund = pd.DataFrame({
        'EPS': eps,
        'ROE': roe,
        'DER': der,
    }).reset_index().rename(columns={'Ticker': 'Kode'})

    print(f"   [OK] Backtest data siap: {len(df_companies)} emiten, "
          f"periode fundamental {period_str}, window OHLCV {len(df_tech)} hari.")
    return df_companies, df_summary, df_tech, df_fund


def fetch_fundamental_minimal(ticker):
    """Hanya mengambil EPS, ROE, dan DER. Market Cap & Sektor sudah di-handle API"""
    try:
        info = yf.Ticker(ticker + ".JK").info
        return {
            'Kode': ticker, # Samakan key dengan master agar mudah di-join
            'EPS': info.get('trailingEps', 0) or 0,
            'ROE': info.get('returnOnEquity', 0) or 0,
            'DER': info.get('debtToEquity', None) or 0
        }
    except Exception:
        return None

def run_live_preprocessing(backtest: bool = False, date: date = None):
    print(f"1. Mengambil & Menyaring Data... (backtest={backtest})")

    if backtest:
        # Mode BACKTEST: sumber data dari parquet lokal, bukan API IDX hari ini.
        df_companies, df_summary, _, _ = load_backtest_data(date)
        if df_companies.empty:
            raise ValueError(f"Tidak ada emiten valid pada {date}! Cek stock_universe / tanggal backtest.")
    else:
        # Tarik data API secara paralel (di memori)
        df_companies = pd.DataFrame(fetch_api_idx_saham())
        df_summary = pd.DataFrame(fetch_api_stock_summary())

        print(f"Data API IDX berhasil diambil: {len(df_companies)} emiten, {len(df_summary)} ringkasan saham.")

        if df_companies.empty or df_summary.empty:
            raise ValueError("Data dari API ZPI kosong! Mohon cek koneksi atau konfigurasi API Key Anda.")

        # SYNC STOCK_UNIVERSE: pastikan semua emiten terdaftar sebelum filtering.
        # Emiten baru dari API langsung dimasukkan (listing_date <- TanggalPencatatan),
        # sehingga generate portofolio tidak akan pernah gagal StockNotFoundError.
        print("1a. Sinkronisasi stock_universe...")
        sync_stock_universe(df_companies)

    # FILTER 1A: Usia IPO > 2 Tahun
    df_companies['TanggalPencatatan'] = pd.to_datetime(df_companies['TanggalPencatatan'])

    if backtest:
        # Mode BACKTEST: usia & keaktivan emiten sudah difilter di dalam
        # get_active_tickers_on (via load_backtest_data) memakai `date`.
        print("   [i] Mode BACKTEST: filter usia IPO diterapkan via stock_universe pada tanggal backtest.")
    else:
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
    
    # Threshold market cap:
    # - LIVE     : Rp 5T (ditetapkan pada kondisi pasar saat ini)
    # - BACKTEST : Rp 5T diskalakan mengikuti total kapitalisasi pasar IDX pada
    #              tanggal backtest, yaitu
    #              (IDX_MktCap(date) / IDX_MktCap(tahun_acuan)) x Rp 5T
    if backtest:
        mcap_info = get_scaled_market_cap_threshold(date)
        batas_mcap = mcap_info['threshold']
        if mcap_info['scaled']:
            print(f"   [i] Backtest: threshold Market Cap diskalakan dari "
                  f"{_format_triliun(MARKET_CAP_THRESHOLD_NOW)} "
                  f"(basis {mcap_info['base_label']}: "
                  f"{_format_triliun(mcap_info['base_market_cap'])}) -> "
                  f"{_format_triliun(batas_mcap)} "
                  f"untuk {mcap_info['target_label']} "
                  f"(rasio {mcap_info['scale']:.4f}).")
        else:
            print(f"   [i] Backtest: skalasi threshold Market Cap tidak tersedia, "
                  f"dipakai {_format_triliun(batas_mcap)}.")
    else:
        batas_mcap = MARKET_CAP_THRESHOLD_NOW

    cond_mcap = df_master['Market_Cap'] >= batas_mcap      # > Rp 5 Triliun (live) / terskalakan (backtest)
    
    # --- LOGGING: Cek siapa yang gagal ---
    failed_mcap = df_master[~cond_mcap].index.tolist()
    if failed_mcap:
        print(f"   [!] Gagal Market Cap < {_format_triliun(batas_mcap)} ({len(failed_mcap)} saham) -> Contoh: {failed_mcap[:7]}...")
     
    # Terapkan filter awal sebelum menyentuh yfinance!
    df_master = df_master[cond_mcap]
    
    print(f"Lolos Filter BEI (Usia, Papan, Market Cap, Harga): {len(df_master)} saham.")

    # HANYA unduh saham yang sudah lolos filter Market Cap 
    ticker_yf = [t + ".JK" for t in df_master.index] 

    if backtest:
        # Mode BACKTEST: OHLCV dari parquet (sudah dilimit window 120 hari),
        # disaring hanya untuk ticker yang lolos filter; fundamental dari
        # fundamental_quarterly.parquet pada periode fiskal yang relevan.
        print("2. Mode BACKTEST: memuat OHLCV & Fundamental dari parquet...")
        _, _, df_tech_full, df_fund_full = load_backtest_data(date)

        tickers_master = set(df_master.index)
        df_tech = df_tech_full.loc[:, df_tech_full.columns.get_level_values(0).isin(tickers_master)]
        # Samakan format kolom dengan yfinance: TICKER.JK
        df_tech.columns = df_tech.columns.set_levels(
            [t + '.JK' for t in df_tech.columns.levels[0]], level=0
        )
        df_fund = df_fund_full[df_fund_full['Kode'].isin(tickers_master)].set_index('Kode')
    else:
        print("2. Mengunduh Fundamental & OHLCV via yfinance...")

        # Tarik OHLCV (Hanya untuk keperluan ADTV 60 dan Vol Sporadis)
        df_tech = yf.download(ticker_yf, period="120d", interval="1d", group_by='ticker')

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

            if len(df_saham) == 0 or len(df_saham) < 60:
                continue

            # # 1. Kalkulasi Nilai Transaksi Harian (Trading Value)
            # trading_value = df_saham['Close'] * df_saham['Volume']
            
            # # 2. ADTV (Mean) dan MDTV (Median) 60 Hari
            # val_60d_mean = float(trading_value.tail(60).mean())
            # val_60d_median = float(trading_value.tail(60).median())
            
            # # Proteksi error division by zero jika median = 0
            # if val_60d_median == 0:
            #     is_sporadic = True
            # else:
            #     # 3. Rasio Skewness (Kemiringan Outlier)
            #     # Jika Mean > 3x lipat Median, berarti ada outlier likuiditas semu
            #     rasio_outlier = val_60d_mean / val_60d_median
                
            #     # Tambahkan Skenario Crossing dari diskusi sebelumnya
            #     freq_today = float(df_master.loc[ticker, 'Frequency'])
            #     vol_today = float(df_saham['Volume'].iloc[-1])
            #     crossing_palsu = (vol_today > 1_000_000) and (freq_today < 100)
                
            #     # Eksekusi Filter Sporadis
            #     is_sporadic = (rasio_outlier > 3.0) or crossing_palsu

            # tech_metrics.append({
            #     'Kode': ticker,
            #     'ADTV_60': val_60d_mean,
            #     'Is_Sporadic': is_sporadic
            # })
            
            # 1. Kalkulasi Estimasi Nilai Transaksi Harian (Estimated Trading Value)
            #    ETV = Close x Volume
            estimated_trading_value = df_saham['Close'] * df_saham['Volume']

            # 2. ADTV_60 (Mean) dan MDTV_60 (Median) 60 Hari
            adtv_60 = float(estimated_trading_value.tail(60).mean())
            mdtv_60 = float(estimated_trading_value.tail(60).median())
            vol_30d_median = float(df_saham['Volume'].tail(30).median())
            vol_30d_mean = float(df_saham['Volume'].tail(30).mean())

            # Proteksi division by zero: median transaksi 0 -> saham mati total,
            # langsung ditandai abnormal (juga akan tereliminasi filter ADTV)
            if mdtv_60 == 0:
                liquidity_outlier = False
                volume_spike = False
                indikasi_markup = False
                is_abnormal = True
            else:
                # --- SIGNAL 1: Liquidity Outlier (Likuiditas Semu) ---
                # Jika rata-rata ETV jauh di atas median-nya, ada outlier likuiditas semu
                rasio_outlier = adtv_60 / mdtv_60
                liquidity_outlier = (rasio_outlier > 3.0)

                # --- SIGNAL 2: Volume Spike ---
                # Volume hari ini meledak (> 3x rata-rata volume 30 hari)
                vol_today = float(df_saham['Volume'].iloc[-1])
                volume_spike = (vol_today > (vol_30d_mean * 3))

                # --- SIGNAL 3: Deteksi Kenaikan Harga Kosong (Fake Markup) ---
                harga_naik = df_saham['Close'].diff() > 0
                volume_tipis = df_saham['Volume'] < (vol_30d_median * 0.5)
                pola_markup = (harga_naik & volume_tipis).tail(3)
                indikasi_markup = pola_markup.sum() >= 2

                # KESIMPULAN:
                # Saham abnormal HANYA jika adanya manipulasi markup yang
                # didukung oleh outlier likuiditas ATAU lonjakan volume.
                is_abnormal = (liquidity_outlier and indikasi_markup) or (volume_spike and indikasi_markup)

            tech_metrics.append({
                'Kode': ticker,
                'ADTV_60': adtv_60,
                'Is_Sporadic': is_abnormal,
                'Sporadic_Reason': [k for k, v in {
                    'liquidity_outlier': liquidity_outlier,
                    'volume_spike': volume_spike,
                    'markup': indikasi_markup
                }.items() if v]
            })
        except Exception:
            continue

    if not tech_metrics:
        sumber = f"tanggal backtest {date}" if backtest else "kondisi pasar live"
        raise ValueError(
            f"Tidak ada saham yang mampu dihitung metrik teknikalnya pada {sumber} "
            f"(data OHLCV tidak tersedia / kurang dari 60 hari)."
        )

    df_tech_summary = pd.DataFrame(tech_metrics).set_index('Kode')
    df_master = df_master.join(df_tech_summary, how='inner')

    # FILTER 2: Eksekusi Eliminasi Mikro (Fundamental & Likuiditas)
    cond_eps  = df_master['EPS'] > 0                              
    cond_roe  = df_master['ROE'] > 0                              
    cond_sporadic = ~df_master['Is_Sporadic']       

    # Threshold ADTV:
    # - LIVE     : Rp 1 Miliar/hari (ditetapkan pada kondisi pasar saat ini)
    # - BACKTEST : Rp 1 Miliar diskalakan mengikuti likuiditas IHSG pada
    #              tanggal backtest, yaitu
    #              (ADTV_IHSG(date) / ADTV_IHSG(sekarang)) x Rp 1 Miliar
    if backtest:
        adtv_info = get_scaled_adtv_threshold(date)
        batas_adtv = adtv_info['threshold']
        if adtv_info['scaled']:
            print(f"   [i] Backtest: threshold ADTV diskalakan dari "
                  f"{_format_miliar(ADTV_THRESHOLD_NOW)} "
                  f"(IHSG {adtv_info['base_label']}: "
                  f"{_format_miliar(adtv_info['base_adtv'])}) -> "
                  f"{_format_miliar(batas_adtv)} "
                  f"untuk {adtv_info['target_label']} "
                  f"(IHSG {_format_miliar(adtv_info['target_adtv'])}, "
                  f"rasio {adtv_info['scale']:.4f}).")
        else:
            print(f"   [i] Backtest: skalasi threshold ADTV tidak tersedia, "
                  f"dipakai {_format_miliar(batas_adtv)}.")
    else:
        batas_adtv = ADTV_THRESHOLD_NOW

    cond_adtv = df_master['ADTV_60'] >= batas_adtv       
              

    # Logika Bypass DER: Nilai DER < 200 ATAU Sektornya Keuangan
    cond_der = (df_master['DER'] < 200) | (df_master['Sektor'] == 'Keuangan') 

    # --- LOGGING: Rincian Eliminasi Mikro ---
    print("\n--- RINCIAN ELIMINASI MIKRO (FUNDAMENTAL & LIKUIDITAS) ---")
    
    failed_eps = df_master[~cond_eps].index.tolist()
    if failed_eps: print(f"   [!] Gagal Laba EPS <= 0 ({len(failed_eps)} saham) -> Contoh: {failed_eps[:7]}...")
    
    failed_roe = df_master[~cond_roe].index.tolist()
    if failed_roe: print(f"   [!] Gagal ROE <= 0 ({len(failed_roe)} saham) -> Contoh: {failed_roe[:7]}...")
    
    failed_adtv = df_master[~cond_adtv].index.tolist()
    if failed_adtv: print(f"   [!] Gagal ADTV < {_format_miliar(batas_adtv)}/Hari ({len(failed_adtv)} saham) -> Contoh: {failed_adtv[:7]}...")
    
    failed_sporadic = df_master[df_master['Is_Sporadic']].index.tolist()
    if failed_sporadic: print(f"   [!] Gagal Volume Sporadis / Manipulasi ({len(failed_sporadic)} saham) -> Contoh: {failed_sporadic[:7]}...")
    
    failed_der = df_master[~cond_der].index.tolist()
    if failed_der: print(f"   [!] Gagal Utang DER >= 200% (Bukan Bank) ({len(failed_der)} saham) -> Contoh: {failed_der[:7]}...")

    df_lolos = df_master[cond_eps & cond_roe & cond_adtv & cond_sporadic & cond_der]

    print(f"Pipeline Live Selesai! Dari ratusan emiten, tersisa {len(df_lolos)} saham unggulan.")
    return df_lolos.index.tolist(), df_lolos