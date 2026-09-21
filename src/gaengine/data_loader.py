"""
Data Loader Terpadu - LIVE & BACKTEST
=====================================

Satu modul untuk merakit :class:`MarketData` yang dikonsumsi GA engine, baik
untuk mode LIVE (data hari ini) maupun BACKTEST (data historis lokal).

Yang berbeda hanyalah SUMBER DATA-nya; perakitan matriks (returns, korelasi,
skor fundamental) memakai SATU kode yang sama sehingga hasil live & backtest
apple-to-apple.

    mode LIVE      -> filtered_stock_cache (DB) + yfinance 1y + ZAPI Pluang
                      + BI rate API
    mode BACKTEST  -> run_live_preprocessing(backtest=True, date=date_ref)
                      + Master_OHLCV_15Tahun.parquet
                      + fundamental_quarterly.parquet
                      + dividend_events_20_tahun.xlsx
                      + BI-7Day-RR.xlsx  (baris terakhir <= date_ref)

Penggunaan:
    build_market_data()                                     # live
    build_market_data(backtest=True, date_ref=date(2024,6,28))   # backtest
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from src.backend.models.database import SessionLocal
from src.backend.models.filtered_stocks_cache import FilteredStockCache
from src.backend.services.api_bi import fetch_bi_rate
from src.gaengine.market_data import MarketData
from src.preprocessing.stock_filtering import get_fiscal_period, run_live_preprocessing

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("BASE_URL")

# ----------------------------------------------------------------------
# Path dataset lokal (untuk mode backtest)
# ----------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

PRICE_FILE       = DATA_DIR / "Master_OHLCV_15Tahun.parquet"
FUND_FILE        = DATA_DIR / "fundamental_quarterly.parquet"
DIVIDEND_FILE    = DATA_DIR / "dividend_events_20_tahun.xlsx"
RISKFREE_FILE    = DATA_DIR / "BI-7Day-RR.xlsx"

# Window return default: 1 tahun (paritas dengan period="1y" yfinance di LIVE).
DEFAULT_LOOKBACK_DAYS = 365

# Panjang window likuiditas: ADTV 60 hari bursa terakhir.
# Nilai ETV harian = Close x Volume; rata-ratanya dipakai untuk memilih
# N kandidat paling likuid (sama seperti metrik ADTV_60 di stock_filtering).
LIQUIDITY_WINDOW_DAYS = 60

# Fundamental metric columns: PER, PBV, ROE, DER, Dividend-Yield.
# ``+1`` berarti "lower is better", ``-1`` berarti "higher is better".
_METRIC_DIRECTION = [+1, +1, -1, +1, -1]

# Nama bulan (Indonesia) untuk parsing tanggal di BI-7Day-RR.xlsx
_NAMA_BULAN_ID = {
    'januari': 1, 'februari': 2, 'maret': 3, 'april': 4, 'mei': 5, 'juni': 6,
    'juli': 7, 'agustus': 8, 'september': 9, 'oktober': 10, 'november': 11,
    'desember': 12,
}


# ======================================================================
# LOGGING (observability saja — TIDAK mengubah alur / hasil perhitungan)
# ======================================================================
# Semua log memakai prefix "[DataLoader]" + nama fungsi, sehingga mudah
# dicari di terminal. Nilai yang dicetak adalah nilai yang benar-benar
# dipakai oleh fungsi tsb (parameter masuk, nilai antara, dan nilai balik).
logger = logging.getLogger(__name__)

TRACE_SAMPLE = 5          # jumlah contoh item yang ditampilkan di ringkasan
METRIC_NAMES = ["PER", "PBV", "ROE", "DER", "DivYld"]


def _ensure_logging_visible() -> None:
    """
    Pastikan log modul ini SELALU tampil di terminal, apa pun konfigurasi
    logging global (basicConfig root, uvicorn, dll.):
      - Pasang StreamHandler langsung di logger modul ini (bukan bergantung
        pada root logger yang level-nya mungkin WARNING / sudah dikonfigurasi
        pihak lain).
      - propagate=False agar pesan tidak dobel lewat handler root.
      - idempotent: bila handler sudah pernah dipasang, jangan tambah lagi.
    """
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)


_ensure_logging_visible()


def _probe_series(s, n: int = TRACE_SAMPLE) -> str:
    """Ringkasan Series untuk log: jumlah valid, min/max, dan contoh nilai."""
    if s is None:
        return "None"
    if not isinstance(s, pd.Series) or s.empty:
        return "kosong"
    num = pd.to_numeric(s, errors="coerce")
    valid = num.dropna()
    if valid.empty:
        return f"n={len(s)}, semua NaN"
    contoh = ", ".join(f"{k}={valid[k]:,.4g}" for k in valid.index[:n])
    return (f"n={len(s)}, valid={len(valid)}, "
            f"min={valid.min():,.6g}, max={valid.max():,.6g} | {contoh}")


def _probe_frame(df, n: int = TRACE_SAMPLE) -> str:
    """Ringkasan DataFrame untuk log: bentuk + rentang tanggal + contoh kolom."""
    if df is None:
        return "None"
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "kosong"
    dim = f"{df.shape[0]} baris x {df.shape[1]} kolom"
    kolom = ", ".join(str(c) for c in list(df.columns)[:n])
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index):
        return (f"{dim}, rentang {df.index.min().date()}..{df.index.max().date()}, "
                f"kolom[:{n}]={kolom}")
    return f"{dim}, kolom[:{n}]={kolom}"


def _probe_metrics_map(metrics_map: Dict[str, List[float]],
                       n: int = TRACE_SAMPLE) -> str:
    """Ringkasan metrics_map: jumlah emiten, contoh baris, dan NaN per metrik."""
    if not metrics_map:
        return "kosong"
    arr = np.asarray(list(metrics_map.values()), dtype=float)
    nan_per_metrik = np.isnan(arr).sum(axis=0) if arr.ndim == 2 else []
    detail = ", ".join(f"{METRIC_NAMES[j]}={int(nan_per_metrik[j])}"
                       for j in range(len(nan_per_metrik)))
    contoh = "; ".join(
        f"{k}=[{', '.join(f'{v:,.4g}' for v in vals)}]"
        for k, vals in list(metrics_map.items())[:n]
    )
    return (f"n_emiten={len(metrics_map)} | NaN per metrik: {detail} | "
            f"contoh (urut {METRIC_NAMES}): {contoh}")


def _fmt_rf(rf: Optional[float]) -> str:
    """Format risk-free rate untuk log (fraksi + persen)."""
    if rf is None:
        return "None"
    try:
        return f"{float(rf):.6f} ({float(rf) * 100:.4f}%)"
    except (TypeError, ValueError):
        return str(rf)


# ======================================================================
# HELPER MODE LIVE
# ======================================================================
def _parse_float_string(val_str):
    """
    Mengubah string teks finansial seperti 'Rp466.74', '13.61x', '20.44%', atau '-'
    menjadi tipe data float bersih yang bisa disimpan di database.
    """
    if not val_str or val_str == "-" or val_str == "N/A":
        return np.nan

    # Hapus karakter mata uang, pengali, persen, koma ribuan, dan spasi kosong
    clean_str = str(val_str).replace("Rp", "").replace(",", "").replace("x", "").replace("%", "").strip()

    try:
        return float(clean_str)
    except ValueError:
        return 0.0


def process_single_fundamental(kode):
    """Fungsi mandiri untuk 1 saham: tarik & parse data fundamental Pluang."""
    from src.backend.services.api_pluang import fetch_api_pluang_fundamentals

    data = fetch_api_pluang_fundamentals(kode)
    if not data:
        return None

    try:
        ratios = data.get("ratios", {})
        overview = data.get("overview", {})

        # Ekstraksi string dari JSON sesuai path
        eps_str = overview.get("eps", "0")
        per_str = ratios.get("valuation", {}).get("pe", "0")
        pbv_str = ratios.get("valuation", {}).get("pb", "0")
        roe_str = ratios.get("profitability", {}).get("roe", "0")
        der_str = ratios.get("solvency", {}).get("de", "0")
        div_str = ratios.get("dividend", {}).get("ttm", "0")

        # Parse semua ke bentuk Float bersih
        return {
            "kode": kode,
            "eps": _parse_float_string(eps_str),
            "per": _parse_float_string(per_str),
            "pbv": _parse_float_string(pbv_str),
            "roe": _parse_float_string(roe_str),
            "der": _parse_float_string(der_str),
            "dividend_yield": _parse_float_string(div_str)
        }
    except Exception as e:
        print(f"Error parsing data fundamental Pluang untuk {kode}: {e}")
        return None


def take_ohlcv_data():
    """
    LIVE: Ambil daftar saham lolos preprocessing dari DB (filtered_stock_cache),
    lalu tarik OHLCV 1 tahun terakhir dari yfinance.
    """
    db = SessionLocal()
    try:
        records = db.query(FilteredStockCache.kode).all()
        daftar_kode = [r[0] for r in records]

        if not daftar_kode:
            print("⚠️ Peringatan: Tabel filtered_stock_cache kosong! Anda belum menjalankan preprocessing.")
            return None

        print(f"Mempersiapkan unduhan OHLCV 1 tahun untuk {len(daftar_kode)} saham dari yfinance...")

        yf_tickers = [f"{kode}.JK" for kode in daftar_kode]
        df_ohlcv = yf.download(yf_tickers, period="1y", interval="1d")

        print("Data OHLCV berhasil diunduh!")
        return df_ohlcv

    except Exception as e:
        print(f"Terjadi kesalahan saat menarik data OHLCV: {e}")
        return None
    finally:
        db.close()


def take_fundamental_data():
    """
    LIVE: Tarik data fundamental Pluang untuk saham yang lolos filter dan
    update kolom eps/per/pbv/roe/der/dividend_yield di filtered_stock_cache.

    CATATAN: Multithreading (10 workers) sengaja dinonaktifkan; fetch berjalan
    SEQUENTIAL agar tidak membebani server API ZAPI.
    """
    db = SessionLocal()
    try:
        records = db.query(FilteredStockCache).all()
        if not records:
            print("Peringatan: Tabel filtered_stock_cache kosong!")
            return None

        print(f"Mulai mengambil data fundamental Pluang untuk {len(records)} saham...")

        results = []
        for i, record in enumerate(records, 1):
            print(f"[{i}/{len(records)}] Menarik data {record.kode}...")
            res = process_single_fundamental(record.kode)
            if res:
                results.append(res)

        print(f"Mengupdate data {len(results)} saham ke dalam database...")
        for res_dict in results:
            db_item = (
                db.query(FilteredStockCache)
                .filter(FilteredStockCache.kode == res_dict["kode"])
                .first()
            )
            if db_item:
                db_item.eps = res_dict["eps"]
                db_item.per = res_dict["per"]
                db_item.pbv = res_dict["pbv"]
                db_item.roe = res_dict["roe"]
                db_item.der = res_dict["der"]
                db_item.dividend_yield = res_dict["dividend_yield"]

        db.commit()
        print("Proses update fundamental ZAPI Pluang selesai!")
        return results

    except Exception as e:
        print(f"Terjadi kesalahan saat take_fundamental_data: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def take_bi_rate():
    """LIVE: Take the latest BI rate data (fraksi desimal, mis. 0.0575)."""
    try:
        rf = fetch_bi_rate()
        logger.info("take_bi_rate: API BI -> risk_free=%s", _fmt_rf(rf))
        return rf
    except Exception as e:
        logger.warning("take_bi_rate: GAGAL mengambil BI rate (%s) -> None", e)
        return None


# ======================================================================
# HELPER MODE BACKTEST
# ======================================================================
def _parse_number(val) -> Optional[float]:
    """Parse angka seperti 1234.5, '1,180.67 B' atau '2.50 %' menjadi float/None."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    if not isinstance(val, str):
        return None
    s = val.strip().replace(",", "").replace(" ", "")
    if s == "":
        return None
    mult = 1.0
    if s.endswith("B"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("T"):
        mult, s = 1e12, s[:-1]
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _parse_tanggal_id(text) -> Optional[date]:
    """Parse tanggal Indonesia seperti '22 Juli 2026' -> datetime.date."""
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s or s == "nan":
        return None
    parts = s.split()
    if len(parts) != 3:
        return None
    try:
        hari = int(parts[0])
        bulan = _NAMA_BULAN_ID.get(parts[1])
        tahun = int(parts[2])
    except (ValueError, TypeError):
        return None
    if bulan is None:
        return None
    try:
        return date(tahun, bulan, hari)
    except ValueError:
        return None


def _load_bi_rate_history(date_ref: date) -> Optional[float]:
    """
    BACKTEST: BI-7Day-RR pada/atau sebelum `date_ref` sebagai fraksi desimal.
    Mengembalikan None bila file tidak ada / tidak ada baris yang cocok.
    """
    logger.info("_load_bi_rate_history: date_ref=%s (ambil baris TERAKHIR <= date_ref)", date_ref)
    try:
        df = pd.read_excel(RISKFREE_FILE, header=None)
    except Exception as e:
        logger.warning("_load_bi_rate_history: Gagal membaca %s (%s) -> None", RISKFREE_FILE.name, e)
        return None

    try:
        dates = df.iloc[:, 1].map(_parse_tanggal_id)
        values = df.iloc[:, 2].astype(str).str.replace("%", "", regex=False) \
            .str.replace(",", ".", regex=False).str.strip()
        values = pd.to_numeric(values, errors="coerce")

        # Susun pasangan (tanggal, rate) lalu ambil baris terbaru <= date_ref.
        # Urutan file tidak diandalkan (bisa menurun) -> selalu sort ascending.
        data = (
            pd.DataFrame({"tanggal": dates, "rate": values})
            .dropna()
            .sort_values("tanggal")
        )
        data = data[data["tanggal"] <= date_ref]
        if data.empty:
            logger.warning("_load_bi_rate_history: tidak ada baris BI rate <= %s -> None", date_ref)
            return None
        tanggal_dipakai = data["tanggal"].iloc[-1]
        hasil = float(data["rate"].iloc[-1]) / 100.0
        logger.info("_load_bi_rate_history: baris terakhir <= date_ref -> tanggal=%s, "
                    "rate_raw=%.2f%%, risk_free=%s",
                    tanggal_dipakai, float(data["rate"].iloc[-1]), _fmt_rf(hasil))
        return hasil
    except Exception as e:
        logger.warning("_load_bi_rate_history: Gagal parsing BI rate historis (%s) -> None", e)
        return None


# def _load_dividend_yields(codes: List[str], date_ref: date) -> Dict[str, float]:
#     """
#     BACKTEST: dividend yield (fraksi) dari event ex-dividend TERAKHIR
#     pada/atau sebelum `date_ref` (anti look-ahead). Default 0.0.
#     """
#     try:
#         df = pd.read_excel(DIVIDEND_FILE)
#     except Exception as e:
#         print(f"   [!] Gagal membaca {DIVIDEND_FILE.name} ({e}) -> dividend yield dianggap 0.")
#         return {}

#     df["Ex_Dividend_Date"] = pd.to_datetime(df["Ex_Dividend_Date"], errors="coerce")
#     df = df[df["Ticker"].isin(codes) & (df["Ex_Dividend_Date"].dt.date <= date_ref)]
#     if df.empty:
#         return {}

#     df = df.assign(Dy=pd.to_numeric(df["Dividend_Yield_ExDate_%"], errors="coerce") / 100.0)
#     last = df.groupby("Ticker")["Ex_Dividend_Date"].transform("max")
#     df = df[df["Ex_Dividend_Date"] == last]
#     mean = df.groupby("Ticker")["Dy"].mean()
#     return {t: float(mean[t]) if t in mean.index else 0.0 for t in codes}

def _load_dividend_yields(
    codes: List[str],
    date_ref: date,
    prices: pd.Series
) -> Dict[str, float]:

    logger.info("_load_dividend_yields: n_codes=%d, window 1 tahun (%s .. %s), "
                "contoh prices=%s",
                len(codes), date_ref - timedelta(days=365), date_ref,
                _probe_series(prices))
    df = pd.read_excel(DIVIDEND_FILE)

    df["Ex_Dividend_Date"] = pd.to_datetime(
        df["Ex_Dividend_Date"],
        errors="coerce"
    )

    start_date = pd.Timestamp(date_ref) - pd.DateOffset(years=1)
    end_date = pd.Timestamp(date_ref)

    df = df[
        df["Ticker"].isin(codes)
        & (df["Ex_Dividend_Date"] > start_date)
        & (df["Ex_Dividend_Date"] <= end_date)
    ]

    dividend_ttm = (
        df.groupby("Ticker")["Dividend_Per_Saham"]
        .sum()
    )

    n_nonzero = int((dividend_ttm > 0).sum())
    logger.info("_load_dividend_yields: %d emiten punya dividen > 0 dalam window | contoh TTM: %s",
                n_nonzero, _probe_series(dividend_ttm))

    result = {}

    for ticker in codes:
        dividend = float(dividend_ttm.get(ticker, 0.0))
        price = prices.get(ticker, np.nan)

        if pd.isna(price) or price <= 0:
            result[ticker] = np.nan
        else:
            result[ticker] = dividend / price

    valid = {t: v for t, v in result.items() if pd.notna(v)}
    contoh = ", ".join(f"{t}={valid[t]:.4%}" for t in list(valid)[:TRACE_SAMPLE])
    logger.info("_load_dividend_yields: hasil -> n=%d, valid(non-NaN)=%d | contoh DivYld: %s",
                len(result), len(valid), contoh or "-")
    return result


def _to_float(value) -> float:
    """Konversi nilai kolom DB ke float (None/NaN -> NaN)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if np.isnan(v) else v


def _safe_value(series: Optional[pd.Series], key: str, default: float = np.nan) -> float:
    """Ambil nilai float dari Series dengan aman (None/abs/en -> default)."""
    if series is None or key not in series.index:
        return default
    return _to_float(series.get(key))


def _positive_or_nan(series: Optional[pd.Series]) -> Optional[pd.Series]:
    """Rasio 0 / negatif dari sumber data dianggap TIDAK TERSEDIA (NaN),
    supaya tidak terbaca sebagai 'saham termurah' saat min-max normalisasi."""
    if series is None:
        return None
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s > 0)


# ----------------------------------------------------------------------
# Provider BACKTEST
# ----------------------------------------------------------------------
def _provide_backtest_data(date_ref: date, lookback_days: int):
    """
    Sumber data BACKTEST (semua bebas look-ahead, dibatasi `date_ref`):
      - universe + ROE/DER  : run_live_preprocessing(backtest=True, date=date_ref)
      - OHLCV harian        : Master_OHLCV_15Tahun.parquet (window <= date_ref)
      - PER & PBV           : fundamental_quarterly.parquet (periode fiskal date_ref)
      - Dividend yield      : dividend_events_20_tahun.xlsx (ex-date <= date_ref)
      - Risk-free rate      : BI-7Day-RR.xlsx (baris terakhir <= date_ref)

    Returns: (df_close, metrics_map, risk_free, liquidity)
    """
    if date_ref is None:
        raise ValueError("Mode backtest memerlukan parameter 'date_ref' yang valid.")

    _t0 = time.perf_counter()
    logger.info("[BACKTEST] _provide_backtest_data: mulai | date_ref=%s, lookback_days=%d "
                "(window return %s .. %s)",
                date_ref, lookback_days, date_ref - timedelta(days=lookback_days), date_ref)

    print(f"[DataLoader] Mode BACKTEST - merakit data per {date_ref}...")

    # 1. Universe + fundamental dasar (EPS/ROE/DER/ADTV) dari pipeline backtest
    _, df_lolos = run_live_preprocessing(backtest=True, date=date_ref)
    if df_lolos is None or df_lolos.empty:
        raise ValueError(f"Tidak ada saham yang lolos preprocessing pada {date_ref}.")
    codes = list(df_lolos.index)
    print(f"[DataLoader] Universe backtest: {len(codes)} saham lolos filter.")
    logger.info("[BACKTEST] preprocessing selesai -> %d saham lolos | ROE: %s | DER: %s",
                len(codes), _probe_series(df_lolos["ROE"]), _probe_series(df_lolos["DER"]))

    # 2. OHLCV historis (window `lookback_days` hari ke belakang dari date_ref)
    df_price = pd.read_parquet(PRICE_FILE)
    df_price["Date"] = pd.to_datetime(df_price["Date"])
    df_price["Ticker"] = df_price["Ticker"].astype(str).str.replace(".JK", "", regex=False)

    ts_ref = pd.Timestamp(date_ref)
    window = df_price[
        (df_price["Date"] <= ts_ref) &
        (df_price["Date"] > ts_ref - pd.Timedelta(days=lookback_days)) &
        (df_price["Ticker"].isin(codes))
    ]
    if window.empty:
        raise ValueError(f"Tidak ada data OHLCV untuk window {lookback_days} hari sebelum {date_ref}.")

    df_close = (
        window.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
        .sort_index()
    )

    # Likuiditas = ADTV 60 HARI BURSA TERAKHIR (ETV = Close x Volume), bukan
    # rata-rata seluruh window. Urutkan kronologis per ticker dulu agar
    # tail(60) benar-benar 60 hari terakhir sebelum date_ref.
    etv = (
        pd.DataFrame({
            "Ticker": window["Ticker"],
            "Date": window["Date"],
            "ETV": window["Close"] * window["Volume"],
        })
        .sort_values(["Ticker", "Date"])
    )
    liquidity = etv.groupby("Ticker")["ETV"].apply(
        lambda s: s.tail(LIQUIDITY_WINDOW_DAYS).mean()
    )
    logger.info("[BACKTEST] df_close=%s", _probe_frame(df_close))
    top_liq = liquidity.sort_values(ascending=False).head(TRACE_SAMPLE)
    logger.info("[BACKTEST] liquidity (ADTV %d hari terakhir, IDR) -> n=%d | Top-%d: %s",
                LIQUIDITY_WINDOW_DAYS, len(liquidity), TRACE_SAMPLE,
                ", ".join(f"{t}={v:,.0f}" for t, v in top_liq.items()))

    # 3. PER & PBV dari fundamental kuartalan (periode fiskal relevan)
    try:
        df_fq = pd.read_parquet(FUND_FILE)
        fy, fq = get_fiscal_period(date_ref)
        period_str = f"Q{fq} {fy}"
        df_p = df_fq[(df_fq["Fiscal_Period"] == period_str) & df_fq["Ticker"].isin(codes)]
        piv = df_p.pivot_table(index="Ticker", columns="Metric", values="Value", aggfunc="last")
        per = _positive_or_nan(piv.get("PE Ratio (Quarter)"))
        pbv = _positive_or_nan(piv.get("Price to Book Value (Quarter)"))
        print(f"[DataLoader] Fundamental backtest: periode fiskal {period_str}.")
        logger.info("[BACKTEST] fundamental periode %s | PER: %s | PBV: %s",
                    period_str, _probe_series(per), _probe_series(pbv))
    except Exception as e:
        print(f"   [!] Gagal membaca {FUND_FILE.name} ({e}) -> PER/PBV dianggap tidak tersedia.")
        logger.warning("[BACKTEST] Gagal membaca %s (%s) -> PER/PBV = None", FUND_FILE.name, e)
        per = pbv = None

    # 4. Dividend yield & risk-free yang bebas look-ahead
    #    `prices` = harga terakhir per ticker (<= date_ref) dipakai sebagai
    #    penyebut dividend yield TTM: yield = sigma(dividen 1 tahun) / harga.
    last_prices = df_close.ffill().iloc[-1]
    logger.info("[BACKTEST] last_prices (penyebut DivYld)=%s", _probe_series(last_prices))
    div_map = _load_dividend_yields(codes, date_ref, last_prices)
    risk_free = _load_bi_rate_history(date_ref)
    logger.info("[BACKTEST] risk_free yang akan dipakai (dari BI-7Day-RR <= %s): %s",
                date_ref, _fmt_rf(risk_free))

    # 5. Rakit metrics_map: [PER, PBV, ROE, DER, DivYld]
    #    DER backtest dari pipeline berbentuk PERSEN -> dijadikan RASIO (÷100)
    #    agar setara dengan DER live (Pluang 'de' = rasio, mis. 0.53x).
    metrics_map: Dict[str, List[float]] = {}
    for t in codes:
        roe = _safe_value(df_lolos["ROE"], t)
        der_pct = _safe_value(df_lolos["DER"], t)
        der_ratio = der_pct / 100.0 if not np.isnan(der_pct) else np.nan
        metrics_map[t] = [
            _safe_value(per, t),
            _safe_value(pbv, t),
            roe,
            der_ratio,
            div_map.get(t, 0.0),
        ]

    logger.info("[BACKTEST] metrics_map selesai: %s", _probe_metrics_map(metrics_map))
    logger.info("[BACKTEST] _provide_backtest_data selesai dalam %.2fs -> "
                "df_close=%s, metrics_map(%d), risk_free=%s, liquidity(n=%d)",
                time.perf_counter() - _t0, _probe_frame(df_close), len(metrics_map),
                _fmt_rf(risk_free), len(liquidity))
    return df_close, metrics_map, risk_free, liquidity


# ----------------------------------------------------------------------
# Provider LIVE
# ----------------------------------------------------------------------
def _provide_live_data():
    """
    Sumber data LIVE:
      - OHLCV 1 tahun  : yfinance (daftar dari filtered_stock_cache)
      - Fundamental    : ZAPI Pluang -> kolom filtered_stock_cache
      - Risk-free rate : API BI (policy-rate terbaru)

    Returns: (df_close, metrics_map, risk_free, liquidity)
    """
    print("[DataLoader] Mode LIVE - menarik data OHLCV & fundamental...")
    df_ohlcv = take_ohlcv_data()
    take_fundamental_data()

    if df_ohlcv is None or df_ohlcv.empty:
        return None, {}, None, pd.Series(dtype=float)

    # Close: forward-fill untuk hari libur bursa, lalu buang suffix '.JK'
    df_close = df_ohlcv["Close"].ffill().fillna(0)
    df_close.columns = [str(c).replace(".JK", "") for c in df_close.columns]

    # Likuiditas = ADTV 60 HARI BURSA TERAKHIR (ETV = Close x Volume).
    # df_ohlcv sudah terurut berdasarkan tanggal (index=Date), sehingga
    # tail(60) mengambil 60 hari bursa terakhir per ticker.
    try:
        tv = df_ohlcv["Close"] * df_ohlcv["Volume"]
        tv.columns = [str(c).replace(".JK", "") for c in tv.columns]
        liquidity = tv.tail(LIQUIDITY_WINDOW_DAYS).mean()
    except Exception:
        liquidity = pd.Series(dtype=float)

    # Fundamental terbaru dari DB cache
    db = SessionLocal()
    try:
        metrics_map: Dict[str, List[float]] = {}
        for s in db.query(FilteredStockCache).all():
            metrics_map[s.kode] = [
                _to_float(s.per), _to_float(s.pbv), _to_float(s.roe),
                _to_float(s.der), _to_float(s.dividend_yield),
            ]
    finally:
        db.close()

    risk_free = take_bi_rate()
    logger.info("[LIVE] df_close=%s", _probe_frame(df_close))
    logger.info("[LIVE] liquidity (ADTV %d hari, IDR)=%s",
                LIQUIDITY_WINDOW_DAYS, _probe_series(liquidity))
    logger.info("[LIVE] metrics_map: %s", _probe_metrics_map(metrics_map))
    logger.info("[LIVE] risk_free dari API BI: %s", _fmt_rf(risk_free))
    return df_close, metrics_map, risk_free, liquidity


# ======================================================================
# PERAKITAN BERSAMA (SATU KODE UNTUK LIVE & BACKTEST)
# ======================================================================
def _normalize_metrics(metrics: np.ndarray) -> np.ndarray:
    """Min-Max normalise each metric to 0-1 respecting its direction, then
    average per stock into a single composite fundamental score 0-1."""
    n, m = metrics.shape
    composite = np.zeros(n)
    for j in range(m):
        col = metrics[:, j]
        valid = col[~np.isnan(col)]
        if valid.size == 0:
            sub = np.full(n, 0.5)
        else:
            lo, hi = float(valid.min()), float(valid.max())
            span = hi - lo
            if span <= 1e-12:
                sub = np.full(n, 1.0)
            else:
                norm = (np.clip(col, lo, hi) - lo) / span
                if _METRIC_DIRECTION[j] > 0:   # lower is better
                    sub = 1.0 - norm
                else:                          # higher is better
                    sub = norm
            sub[np.isnan(col)] = 0.5
        composite += sub
    return composite / m


def _assemble_market_data(
    df_close: pd.DataFrame,
    metrics_map: Dict[str, List[float]],
    risk_free: Optional[float],
    liquidity: pd.Series,
    min_price: float,
    max_stocks: Optional[int],
) -> Optional[MarketData]:
    """
    Rakit MarketData dari bahan mentah yang SUDAH dinormalkan bentuknya oleh
    provider (live maupun backtest):
        df_close    : index=Date, kolom=ticker (tanpa '.JK')
        metrics_map : {ticker: [PER, PBV, ROE, DER, DivYld]}
        risk_free   : fraksi tahunan (mis. 0.0575)
        liquidity   : ADTV 60 hari terakhir per ticker (IDR/hari), dipakai
                      untuk memilih `max_stocks` kandidat paling likuid
    """
    if df_close is None or df_close.empty or not metrics_map:
        print("Error: Data OHLCV / fundamental kosong. Perakitan dibatalkan.")
        return None

    # 1. Kandidat: ada di OHLCV & fundamental, harga terakhir >= min_price
    candidates = [
        c for c in df_close.columns
        if c in metrics_map and pd.notna(df_close[c].iloc[-1]) and df_close[c].iloc[-1] >= min_price
    ]
    candidates = sorted(candidates)

    if not candidates:
        print("Error: Tidak ada kandidat saham yang valid tersisa.")
        return None

    # 2. Batasi jumlah kandidat berdasarkan likuiditas (bila diminta)
    if max_stocks is not None and len(candidates) > max_stocks:
        liq = liquidity.reindex(candidates).fillna(0.0)
        candidates = liq.sort_values(ascending=False).head(max_stocks).index.tolist()
        candidates = sorted(candidates)

    # 3. Harga per lot (baris terakhir x 100 saham)
    prices_per_lot = np.asarray([df_close[c].iloc[-1] * 100.0 for c in candidates], dtype=float)

    # 4. Returns matrix (N saham x T hari)
    df_ret = df_close[candidates].pct_change().fillna(0.0)
    returns = df_ret.to_numpy(dtype=float).T

    # 5. Correlation matrix
    correlation = np.corrcoef(returns)
    correlation = np.nan_to_num(correlation, nan=0.0)

    # 6. Fundamental metrics & skor komposit 0-1
    fundamental_metrics = np.asarray([metrics_map[c] for c in candidates], dtype=float)
    fundamental_scores = _normalize_metrics(fundamental_metrics)

    # 7. Risk-free: fallback 6.25% bila gagal diambil
    rf = risk_free if risk_free is not None else 0.0625
    if risk_free is None:
        logger.warning("_assemble_market_data: risk_free=None -> fallback %.4f (6.25%%)", rf)
    logger.info("_assemble_market_data: n_candidates=%d, min_price=%.0f, max_stocks=%s, "
                "n_final=%d, rf=%s",
                len(candidates), min_price, max_stocks, len(candidates), _fmt_rf(rf))
    logger.info("_assemble_market_data: kandidat final: %s", candidates[:TRACE_SAMPLE])

    return MarketData(
        stock_codes=candidates,
        prices_per_lot=prices_per_lot,
        returns=returns,
        correlation=correlation,
        fundamental_scores=fundamental_scores,
        fundamental_metrics=fundamental_metrics,
        risk_free_rate=rf,
    )


# ======================================================================
# PUBLIC API
# ======================================================================
def build_market_data(
    min_price: float = 50.0,
    max_stocks: Optional[int] = None,
    backtest: bool = False,
    date_ref: Optional[date] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Optional[MarketData]:
    """
    Rakit MarketData untuk GA engine.

    Args:
        min_price     : harga minimum (IDR) saham kandidat.
        max_stocks    : batasi jumlah kandidat (None = semua).
        backtest      : False = data LIVE, True = data historis lokal.
        date_ref      : WAJIB untuk backtest; tanggal acuan backtest.
        lookback_days : panjang window return (default 365 hari = 1 tahun).

    Returns:
        MarketData, atau None bila data live tidak tersedia (mis. cache DB
        kosong / yfinance gagal). Mode backtest akan melempar ValueError bila
        tanggal acuan tidak ada isinya.
    """
    if backtest:
        df_close, metrics_map, risk_free, liquidity = _provide_backtest_data(date_ref, lookback_days)
    else:
        df_close, metrics_map, risk_free, liquidity = _provide_live_data()

    logger.info("build_market_data: parameter -> min_price=%.0f, max_stocks=%s, "
                "backtest=%s, date_ref=%s, lookback_days=%d",
                min_price, max_stocks, backtest, date_ref, lookback_days)
    logger.info("build_market_data: provider mengembalikan -> df_close=%s, "
                "metrics_map: %s, risk_free=%s, liquidity: %s",
                _probe_frame(df_close), _probe_metrics_map(metrics_map),
                _fmt_rf(risk_free), _probe_series(liquidity))

    market_data = _assemble_market_data(
        df_close, metrics_map, risk_free, liquidity, min_price, max_stocks
    )

    if market_data is not None:
        mode = f"BACKTEST {date_ref}" if backtest else "LIVE"
        print(f"[DataLoader] MarketData {mode} siap: {market_data.n_stocks} saham, "
              f"returns {market_data.returns.shape[0]}x{market_data.returns.shape[1]}, "
              f"rf={market_data.risk_free_rate:.4f}.")
    return market_data