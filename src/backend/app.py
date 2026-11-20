from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from datetime import date
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from src.backend.routes.api import api_router
from src.backend.core.config import settings
from src.backend.core.analysis_shutdown import cancel_analyses_on_shutdown
from src.backend.core.analysis_logging import nonblocking_analysis_logging
from src.backend.models.database import engine, Base, SessionLocal
from sqlalchemy import inspect, text
from src.backend.controller.market_controller import (
    get_last_filter_update,
    run_daily_pipeline,
)
from fastapi.middleware.cors import CORSMiddleware

SCHEDULER_TIMEZONE = "Asia/Jakarta"  # pipeline harian jam 04:00 WIB


def _is_filter_stale() -> bool:
    """
    Cek apakah filtered_stock_cache sudah diperbarui HARI INI.
    True = belum pernah dijalankan / data masih milik hari sebelumnya.
    """
    db = SessionLocal()
    try:
        last_update = get_last_filter_update(db)
        return last_update is None or last_update.date() != date.today()
    except Exception:
        # Jika gagal membaca (mis. tabel belum ada), anggap basi agar pipeline jalan
        return True
    finally:
        db.close()


def scheduled_daily_refresh() -> None:
    """
    Job harian APScheduler (jam 04:00 WIB):
      1. Filtering saham unggulan hari ini -> filtered_stock_cache
      2. build_market_data -> MarketData (fundamental + OHLCV) ke RAM
      3. sync_market_data -> harga terbaru market_data & idx_composite
    Hasil MarketData disimpan ke app.state agar dipakai endpoint /generate.
    """
    try:
        result = run_daily_pipeline()
        app.state.market_data_today = result["market_data"]
        print(f"[Scheduler] Pipeline harian selesai: {result['filtered_stocks']} saham lolos, "
              f"sync={result['sync_stats']}")
    except Exception:
        # Scheduler thread tidak boleh mati karena exception satu job
        import traceback
        print("[Scheduler] Pipeline harian GAGAL:")
        traceback.print_exc()


def _ensure_portfolio_columns(database_engine) -> None:
    """Tambahkan kolom snapshot baru pada instalasi dengan tabel portofolios lama."""
    existing = {column["name"] for column in inspect(database_engine).get_columns("portofolios")}
    with database_engine.begin() as connection:
        optional = "IF NOT EXISTS " if database_engine.dialect.name == "postgresql" else ""
        if "execution_date" not in existing:
            connection.execute(text(f"ALTER TABLE portofolios ADD COLUMN {optional}execution_date DATE"))
        if "execution_prices" not in existing:
            price_type = "JSONB" if database_engine.dialect.name == "postgresql" else "JSON"
            connection.execute(text(f"ALTER TABLE portofolios ADD COLUMN {optional}execution_prices {price_type}"))
        if "pending_deposit" not in existing:
            connection.execute(text(
                f"ALTER TABLE portofolios ADD COLUMN {optional}pending_deposit "
                "DOUBLE PRECISION NOT NULL DEFAULT 0"
            ))
        if "capital_contribution" not in existing:
            connection.execute(text(
                f"ALTER TABLE portofolios ADD COLUMN {optional}capital_contribution "
                "DOUBLE PRECISION NOT NULL DEFAULT 0"
            ))
        if "name_portofolio" not in existing:
            connection.execute(text(f"ALTER TABLE portofolios ADD COLUMN {optional}name_portofolio VARCHAR(80)"))
        if "description" not in existing:
            connection.execute(text(f"ALTER TABLE portofolios ADD COLUMN {optional}description TEXT"))
        if "is_list_active" not in existing:
            connection.execute(text(f"ALTER TABLE portofolios ADD COLUMN {optional}is_list_active BOOLEAN"))
        if "transaction_costs" not in existing:
            connection.execute(text(f"ALTER TABLE portofolios ADD COLUMN {optional}transaction_costs DOUBLE PRECISION NOT NULL DEFAULT 0"))
        if "rebalance_metadata" not in existing:
            data_type="JSONB" if database_engine.dialect.name=="postgresql" else "JSON"
            connection.execute(text(f"ALTER TABLE portofolios ADD COLUMN {optional}rebalance_metadata {data_type}"))
        # Freeze the legacy list badges once. Later generations must not
        # silently deactivate a root the user already sees as active.
        connection.execute(text("""
            WITH RECURSIVE current_chains AS (
                SELECT id, parent_portofolio_id FROM portofolios
                WHERE status_portofolio IN ('active', 'active_backtest')
                UNION
                SELECT p.id, p.parent_portofolio_id FROM portofolios p
                JOIN current_chains c ON p.id = c.parent_portofolio_id
            )
            UPDATE portofolios SET is_list_active = id IN (
                SELECT id FROM current_chains WHERE parent_portofolio_id IS NULL
            )
            WHERE is_list_active IS NULL AND parent_portofolio_id IS NULL
        """))


def _ensure_filtered_stock_cache_columns(database_engine) -> None:
    """Preserve source-session provenance for cached Close fallback."""
    columns = {column['name'] for column in inspect(database_engine).get_columns('filtered_stock_cache')}
    if 'price_date' not in columns:
        with database_engine.begin() as connection:
            optional = 'IF NOT EXISTS ' if database_engine.dialect.name == 'postgresql' else ''
            connection.execute(text(f'ALTER TABLE filtered_stock_cache ADD COLUMN {optional}price_date DATE'))


def _ensure_user_columns(database_engine) -> None:
    existing = {column["name"] for column in inspect(database_engine).get_columns("users")}
    if "bear_allocation_policy" not in existing:
        optional = "IF NOT EXISTS " if database_engine.dialect.name == "postgresql" else ""
        with database_engine.begin() as connection:
            connection.execute(text(
                f"ALTER TABLE users ADD COLUMN {optional}bear_allocation_policy VARCHAR(20) "
                "NOT NULL DEFAULT 'manual'"
            ))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Generate tabel saat startup (lebih aman di sini daripada global scope)
    print("Mengecek dan membuat tabel database...")
    Base.metadata.create_all(bind=engine)
    _ensure_filtered_stock_cache_columns(engine)
    # create_all does not add columns to existing installations. Keep the
    # additive accounting migration idempotent for deployments without Alembic.
    _ensure_portfolio_columns(engine)
    _ensure_user_columns(engine)

    # 2. Load data pasar ke memori untuk engine Algoritma Genetika
    print("Memuat MarketData ke RAM...")
    # app.state.market_data_today = build_market_data()

    # 3. Scheduler harian (APScheduler) — jam 04:00 WIB:
    #    filtering -> build_market_data -> sync_prices
    scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)
    scheduler.add_job(
        scheduled_daily_refresh,
        CronTrigger(hour=17, minute=0),
        id="daily_market_refresh",
        replace_existing=True,
    )
    scheduler.start()
    app.state.scheduler = scheduler
    print("[Scheduler] Berjalan — job 'daily_market_refresh' dijadwalkan pukul 17:00 WIB.")

    # 4. Cek staleness saat startup: kalau filtered_stock_cache belum diperbarui
    #    hari ini, langsung jalankan pipeline sekali (jangan tunggu jam 04:00).
    if _is_filter_stale():
        print("[Startup] Data filtering basi/belum ada -> jalankan pipeline harian sekarang...")
        scheduled_daily_refresh()
    else:
        print("[Startup] Data filtering sudah terbaru (hari ini) -> pipeline dilewati.")

    with cancel_analyses_on_shutdown(), nonblocking_analysis_logging():
        logging.getLogger(__name__).info(
            "Analysis runtime ready: backend-clock-v3, GA thread HighQoS, cancellable narrative I/O, non-blocking logs, progress 1 Hz")
        from src.gaengine.composition_v2_experiment import LIMITS, POLICY_VERSION
        logging.getLogger(__name__).info("Composition v2 policy ready: %s | volatility caps=%s",
            POLICY_VERSION, {profile: limits.max_vol_ratio for profile, limits in LIMITS.items()})
        try:
            yield
        finally:
            # Always clean up, including forced cancellation during shutdown.
            app.state.market_data_today = None
            scheduler.shutdown(wait=False)
            print("[Scheduler] Dimatikan.")

# Gabungkan seluruh konfigurasi (metadata + lifespan) dalam SATU instance FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for GA Portfolio Recommendation with LLM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Standarisasi format ERROR: semua error (HTTPException & validasi Pydantic)
# dibungkus envelope ApiResponse {"status": "error", "message": ..., "data": null}
# sehingga frontend hanya perlu mengenali SATU format respons.
# ---------------------------------------------------------------------------
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Gunakan StarletteHTTPException agar menangkap SEMUA HTTP error,
# termasuk 404 "Not Found" dari router (bukan hanya HTTPException buatan kita).
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc.detail), "data": None},
        headers=getattr(exc, "headers", None),
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Rangkum semua kesalahan validasi body/query (mis. email invalid, password lemah)
    errors = [
        {"field": ".".join(str(loc) for loc in err.get("loc", []) if loc != "body"),
         "issue": err.get("msg", "invalid")}
        for err in exc.errors()
    ]
    message = "; ".join(f"{e['field']}: {e['issue']}" for e in errors) or "Input tidak valid."
    return JSONResponse(
        status_code=422,
        content={"status": "error", "message": message, "data": errors},
    )

import logging

# Ubah batas minimal logging menjadi INFO
logging.basicConfig(level=logging.INFO)

# Sekarang ini akan muncul di terminal!
logging.info("Memulai aplikasi backend...")

# Daftarkan semua route ke dalam API dengan prefix standar
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Selamat datang di API Sistem Rekomendasi Portofolio GA!"}

if __name__ == "__main__":
    from src.backend.core.local_server import run_local_backend
    run_local_backend()
