import logging
from datetime import date

from fastapi import BackgroundTasks
from sqlalchemy import func
from sqlalchemy.orm import Session
from src.backend.models.schemas.market_schema import MarketFilterResponse
from src.backend.models.database import SessionLocal
from src.backend.models.filtered_stocks_cache import FilteredStockCache
from src.backend.models.market_data import MarketData
from src.backend.models.idx_composite import IdxComposite
from src.backend.models.portofolio_items import PortofolioItem
from src.backend.models.portofolios import Portofolio
from src.backend.models.stock_universe import StockUniverse
from src.preprocessing.stock_filtering import run_live_preprocessing

logger = logging.getLogger(__name__)


class StockFilteringError(Exception):
    """Exception domain: pipeline seleksi saham gagal dijalankan."""


class PortfolioNotFoundError(Exception):
    """Exception domain: user belum memiliki portofolio (tidak ada histori harga)."""


def run_and_cache_stock_filtering(background_tasks: BackgroundTasks) -> MarketFilterResponse:
    """
    Controller untuk endpoint /market/filter-stocks:
    1. Menjalankan pipeline seleksi saham secara live (1-2 menit).
    2. Mengirim tugas simpan ke DB sebagai background task (asynchronous).
    3. Mengembalikan hasil domain (MarketFilterResponse) untuk dibungkus envelope oleh routes.
    """
    try:
        daftar_saham, df_lolos = run_live_preprocessing()
    except Exception as e:
        logger.exception("Gagal menjalankan pipeline filtering saham")
        raise StockFilteringError(f"Filtering saham gagal: {e}")

    # 'Kode' tersembunyi sebagai index di df_lolos, jadi di-reset dulu
    df_json = df_lolos.reset_index().to_dict(orient="records")

    background_tasks.add_task(save_filtered_stocks_to_db, df_json)

    return MarketFilterResponse(
        total_saham=len(daftar_saham),
        data=df_json,
    )


def get_last_filter_update(db: Session):
    """
    Ambil updated_at dari salah satu baris filtered_stock_cache.
    Semua baris ditulis bersamaan oleh job filtering (truncate + insert),
    sehingga updated_at sinkron untuk semua row — cukup ambil MAX.
    Returns: datetime atau None jika tabel kosong.
    """
    from sqlalchemy import func
    from src.backend.models.filtered_stocks_cache import FilteredStockCache

    return db.query(func.max(FilteredStockCache.updated_at)).scalar()


def run_daily_pipeline() -> dict:
    """
    Pipeline harian (dipanggil scheduler jam 04:00 atau lifespan saat data basi):
      1. Filtering saham hari ini -> simpan ke filtered_stock_cache.
      2. build_market_data() -> muat data pasar + fundamental ke RAM untuk GA.
      3. sync_market_data -> tarik harga terbaru (yfinance) untuk market_data
         & idx_composite milik portofolio user.
    Returns: statistik ringkas; 'market_data' berupa objek MarketData untuk
    disimpan ke app.state oleh pemanggil.
    """
    from src.gaengine.data_loader_live import build_market_data
    from src.backend.services.price_history_service import sync_market_data

    db = SessionLocal()
    try:
        # 1. Filtering & simpan cache (SINKRON, bukan background task,
        #    karena scheduler tidak punya BackgroundTasks FastAPI)
        daftar_saham, df_lolos = run_live_preprocessing()
        df_json = df_lolos.reset_index().to_dict(orient="records")
        save_filtered_stocks_to_db(df_json)
        logger.info("Pipeline harian [1/3]: filtering selesai (%d saham lolos).", len(daftar_saham))

        # 2. Perakitan MarketData ke RAM (fundamental + OHLCV 1 tahun)
        market_data = build_market_data()
        logger.info("Pipeline harian [2/3]: build_market_data selesai (%s saham).",
                    market_data.n_stocks if market_data else 0)

        # 3. Sinkronisasi harga portofolio user + IHSG
        stats = sync_market_data(db)
        logger.info("Pipeline harian [3/3]: sync_market_data selesai (%s).", stats)

        return {
            "filtered_stocks": len(daftar_saham),
            "market_data": market_data,
            "sync_stats": stats,
        }
    finally:
        db.close()


def save_filtered_stocks_to_db(data_records: list):
    """
    Fungsi ini berjalan di background (Asynchronous Task).
    Tugasnya adalah menyimpan hasil dataframe yang sudah diubah menjadi dictionary
    ke dalam tabel filtered_stock_cache di database.
    """
    # Membuat sesi DB baru khusus untuk proses background
    db = SessionLocal()
    try:
        # Hapus data cache lama agar selalu fresh (Truncate)
        db.query(FilteredStockCache).delete()
        
        # Insert data baru satu per satu
        for row in data_records:
            cache_item = FilteredStockCache(
                kode=row.get('Kode', ''),
                nama=row.get('NamaEmiten', ''), # Sesuaikan dengan nama kolom dari API IDX
                sektor=row.get('Sektor', ''),
                close=float(row.get('Close', 0.0)),
                market_cap=float(row.get('Market_Cap', 0.0)),
                eps=float(row.get('EPS', 0.0)),
                roe=float(row.get('ROE', 0.0)),
                der=float(row.get('DER', 0.0)),
                adtv_60=float(row.get('ADTV_60', 0.0))
            )
            db.add(cache_item)
            
        # Simpan ke database
        db.commit()
        print(f"Background Task: {len(data_records)} saham berhasil disimpan ke tabel FilteredStockCache.")
    
    except Exception as e:
        db.rollback()
        print(f"Background Task Error saat menyimpan ke DB: {e}")
    finally:
        # Tutup sesi agar memori tidak bocor
        db.close()


def get_price_history(db: Session, user_id: str) -> dict:
    """
    Ambil histori harga harian (OHLCV + adj_close) saham yang dimiliki user
    pada portofolio BERSTATUS ACTIVE (selalu yang terbaru) DAN harga IHSG
    (benchmark) dari tabel idx_composite,
    dari tanggal pembuatan portofolio active tersebut sampai data terbaru.

    Struktur return:
    {
        "start_date": "...",   # tanggal portofolio active user
        "end_date": "...",     # tanggal data terbaru (saham maupun IHSG)
        "stocks": [
            {
                "ticker": "BBCA",
                "stock_id": "...",
                "prices": [
                    {"date": "2026-08-01", "open": ..., "high": ...,
                     "low": ..., "close": ..., "adj_close": ..., "volume": ...},
                    ...
                ]
            },
            ...
        ],
        "benchmark": {
            "ticker": "^JKSE",
            "prices": [
                {"date": "2026-08-01", "open": ..., "high": ...,
                 "low": ..., "close": ...},
                ...
            ]
        }
    }
    Raises: PortfolioNotFoundError jika user belum punya portofolio aktif.
    """
    from uuid import UUID

    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise PortfolioNotFoundError(f"User ID pada token tidak valid: {user_id}") from e

    # Tanggal awal = portofolio BERSTATUS ACTIVE (yang paling baru).
    # Portofolio lama berstatus 'replaced' tidak disertakan agar tampilan
    # selalu mengikuti portofolio terbaru.
    start_dt = (
        db.query(func.min(Portofolio.created_at))
        .filter(
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio == "active",
        )
        .scalar()
    )
    if start_dt is None:
        raise PortfolioNotFoundError("User belum memiliki portofolio aktif.")
    start_date = start_dt.date() if hasattr(start_dt, "date") else start_dt

    # Saham-saham yang dimiliki user pada portofolio ACTIVE
    stock_ids = [
        sid for (sid,) in (
            db.query(PortofolioItem.stock_id)
            .join(Portofolio, PortofolioItem.portofolio_id == Portofolio.id)
            .filter(
                Portofolio.user_id == user_uuid,
                Portofolio.status_portofolio == "active",
            )
            .distinct()
            .all()
        )
    ]
    if not stock_ids:
        raise PortfolioNotFoundError("User belum memiliki saham di portofolio aktifnya.")

    # Histori harga untuk saham-saham tsb, sejak tanggal portofolio terawal.
    # PENTING: bandingkan Date dengan Date (start_date), BUKAN dengan
    # start_dt (DateTime) — '2026-09-04' >= '2026-09-04 14:30' akan FALSE
    # karena date di-cast jadi timestamp 00:00:00.
    rows = (
        db.query(MarketData, StockUniverse.ticker)
        .join(StockUniverse, MarketData.stock_id == StockUniverse.id_stock)
        .filter(
            MarketData.stock_id.in_(stock_ids),
            MarketData.date >= start_date,
        )
        .order_by(MarketData.stock_id, MarketData.date.asc())
        .all()
    )

    # Grouping per ticker agar mudah digambar chart di frontend
    grouped: dict[str, dict] = {}
    for md, ticker in rows:
        entry = grouped.setdefault(ticker, {"ticker": ticker, "stock_id": str(md.stock_id), "prices": []})
        entry["prices"].append({
            "date": md.date.isoformat(),
            "open": md.open,
            "high": md.high,
            "low": md.low,
            "close": md.close,
            "adj_close": md.adj_close,
            "volume": md.volume,
        })

    all_dates = [md.date for md, _ in rows if isinstance(md.date, date)]

    # Benchmark IHSG dari idx_composite (rentang sama: sejak portofolio terawal)
    idx_rows = (
        db.query(IdxComposite)
        .filter(IdxComposite.date >= start_date)
        .order_by(IdxComposite.date.asc())
        .all()
    )
    benchmark_prices = [
        {
            "date": ic.date.isoformat(),
            "open": ic.open,
            "high": ic.high,
            "low": ic.low,
            "close": ic.close,
        }
        for ic in idx_rows
    ]
    all_dates.extend(ic.date for ic in idx_rows if isinstance(ic.date, date))

    max_date = max(all_dates, default=None)

    return {
        "start_date": start_dt.date().isoformat() if hasattr(start_dt, "date") else str(start_dt),
        "end_date": max_date.isoformat() if max_date else None,
        "stocks": [grouped[t] for t in sorted(grouped)],
        "benchmark": {
            "ticker": idx_rows[0].ticker if idx_rows else None,
            "prices": benchmark_prices,
        },
    }


def get_portfolio_performance(db: Session, user_id: str) -> dict:
    """
    Hitung performa (persentase return kumulatif harian) portofolio ACTIVE user
    vs IHSG, sejak tanggal portofolio aktif dibuat.

    Metode (berbasis KEPEMILIKAN LOT, bukan bobot investasi):
      nilai_portofolio(t) = SUM( jumlah_lot_i * adj_close_i(t) )
      return(t)           = nilai_portofolio(t) / nilai_portofolio(t0) - 1
    Kontribusi tiap emiten otomatis mengikuti pergerakan harganya, sehingga
    return portofolio total lebih akurat.

    Raises: PortfolioNotFoundError jika user belum punya portofolio aktif.
    """
    import pandas as pd
    from uuid import UUID

    try:
        user_uuid = UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise PortfolioNotFoundError(f"User ID pada token tidak valid: {user_id}") from e

    # 1. Portofolio ACTIVE user — ambil yang TERBARU secara eksplisit
    #    (bukan MIN/MAX sembarang active), jadi basis perhitungan jelas.
    portfolio = (
        db.query(Portofolio)
        .filter(
            Portofolio.user_id == user_uuid,
            Portofolio.status_portofolio == "active",
        )
        .order_by(Portofolio.created_at.desc(), Portofolio.id.desc())
        .first()
    )
    if portfolio is None:
        raise PortfolioNotFoundError("User belum memiliki portofolio aktif.")
    start_dt = portfolio.created_at
    start_date = start_dt.date() if hasattr(start_dt, "date") else start_dt

    # 2. Kepemilikan lot per emiten dari portofolio active tersebut (by id)
    holdings_rows = (
        db.query(
            PortofolioItem.stock_id,
            StockUniverse.ticker,
            PortofolioItem.jumlah_lot,
        )
        .join(Portofolio, PortofolioItem.portofolio_id == Portofolio.id)
        .join(StockUniverse, PortofolioItem.stock_id == StockUniverse.id_stock)
        .filter(PortofolioItem.portofolio_id == portfolio.id)
        .all()
    )
    if not holdings_rows:
        raise PortfolioNotFoundError("User belum memiliki saham di portofolio aktifnya.")

    stock_ids = [r.stock_id for r in holdings_rows]
    lots_map = {r.ticker: int(r.jumlah_lot) for r in holdings_rows}

    # 3. Harga per (date, stock_id) sejak start_date
    price_rows = (
        db.query(MarketData.stock_id, MarketData.date, MarketData.adj_close, MarketData.close)
        .filter(
            MarketData.stock_id.in_(stock_ids),
            MarketData.date >= start_date,
        )
        .all()
    )
    if not price_rows:
        raise PortfolioNotFoundError(
            "Belum ada data harga tersimpan untuk saham portofolio aktif. "
            "Jalankan sync-prices terlebih dahulu."
        )

    # 4. Vectorization: pivot date x stock (adj_close; fallback close bila adj kosong)
    recs = [
        {"date": r.date, "stock_id": str(r.stock_id),
         "price": r.adj_close if r.adj_close is not None else r.close}
        for r in price_rows
        if (r.adj_close is not None or r.close is not None)
    ]
    if not recs:
        raise PortfolioNotFoundError("Data harga (adj_close/close) belum tersedia untuk saham portofolio aktif.")

    df_price = pd.DataFrame(recs).pivot_table(
        index="date", columns="stock_id", values="price", aggfunc="last"
    ).sort_index()
    # dropna eksplisit bertahap:
    # a) ffill: isi tanggal bolong per saham selama saham tsb sudah punya harga
    #    sebelumnya (mis. saham suspension 1 hari)
    df_price = df_price.ffill()
    # b) buang baris yang KOSONG SEMUA (hari libur bersama sebelum saham manapun
    #    punya harga — biasanya di awal rentang)
    df_price = df_price.dropna(how="all")
    # c) basis t0 harus tanggal di mana SEMUA saham portofolio sudah punya harga,
    #    supaya nilai portofolio t0 benar-benar mencakup seluruh kepemilikan
    df_price = df_price.dropna(axis=0, how="any")
    if df_price.empty:
        raise PortfolioNotFoundError("Data harga belum cukup untuk menghitung performa.")

    id_to_ticker = {str(r.stock_id): r.ticker for r in holdings_rows}
    df_price.columns = [id_to_ticker[c] for c in df_price.columns]

    # 5. Nilai portofolio harian = SUM(lots * price).
    #    PENTING: adj_close dari yfinance adalah harga PER LEMBAR, sedangkan
    #    jumlah_lot = 100 lembar, jadi konversi lot -> lembar (x 100).
    LOTS_PER_SHARE = 100
    lots = pd.Series({t: lots_map[t] * LOTS_PER_SHARE for t in df_price.columns if t in lots_map})
    df_price = df_price[list(lots.index)]
    portfolio_value = df_price.mul(lots, axis=1).sum(axis=1)
    portfolio_return = portfolio_value / portfolio_value.iloc[0] - 1

    # 6. Return IHSG sejak tanggal yang sama
    idx_rows = (
        db.query(IdxComposite)
        .filter(IdxComposite.date >= df_price.index.min())
        .order_by(IdxComposite.date.asc())
        .all()
    )
    ihsg_return = None
    if idx_rows:
        idx_series = pd.Series(
            {r.date: r.close for r in idx_rows if r.close is not None}
        ).sort_index()
        ihsg_return = idx_series / idx_series.iloc[0] - 1

    # 7. Gabungkan (inner join tanggal agar kedua garis selalu punya nilai)
    combined = pd.DataFrame({"portfolio_return": portfolio_return})
    if ihsg_return is not None:
        combined["ihsg_return"] = ihsg_return
        combined = combined.dropna()

    series = [
        {
            "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
            "portfolio_value": float(portfolio_value.loc[d]),
            "portfolio_return": float(row["portfolio_return"]),
            "ihsg_return": (
                float(row["ihsg_return"])
                if "ihsg_return" in row and pd.notna(row["ihsg_return"])
                else None
            ),
        }
        for d, row in combined.iterrows()
    ]

    end = combined.index.max() if len(combined) else df_price.index.max()

    return {
        "start_date": start_dt.date().isoformat() if hasattr(start_dt, "date") else str(start_dt),
        "end_date": end.isoformat() if hasattr(end, "isoformat") else str(end),
        "holdings": {t: lots_map[t] for t in sorted(lots_map)},
        "series": series,
    }

