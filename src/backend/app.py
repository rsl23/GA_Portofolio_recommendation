from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from datetime import date
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from src.backend.routes.api import api_router
from src.backend.core.config import settings
from src.backend.models.database import engine, Base, SessionLocal
import src.backend.models.filtered_stocks_cache
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Generate tabel saat startup (lebih aman di sini daripada global scope)
    print("Mengecek dan membuat tabel database...")
    Base.metadata.create_all(bind=engine)

    # 2. Load data pasar ke memori untuk engine Algoritma Genetika
    print("Memuat MarketData ke RAM...")
    # app.state.market_data_today = build_market_data()

    # 3. Scheduler harian (APScheduler) — jam 04:00 WIB:
    #    filtering -> build_market_data -> sync_prices
    scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)
    scheduler.add_job(
        scheduled_daily_refresh,
        CronTrigger(hour=4, minute=0),
        id="daily_market_refresh",
        replace_existing=True,
    )
    scheduler.start()
    app.state.scheduler = scheduler
    print("[Scheduler] Berjalan — job 'daily_market_refresh' dijadwalkan pukul 04:00 WIB.")

    # 4. Cek staleness saat startup: kalau filtered_stock_cache belum diperbarui
    #    hari ini, langsung jalankan pipeline sekali (jangan tunggu jam 04:00).
    if _is_filter_stale():
        print("[Startup] Data filtering basi/belum ada -> jalankan pipeline harian sekarang...")
        scheduled_daily_refresh()
    else:
        print("[Startup] Data filtering sudah terbaru (hari ini) -> pipeline dilewati.")

    yield

    # Clean up saat server dimatikan
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

# Daftarkan semua route ke dalam API dengan prefix standar
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Selamat datang di API Sistem Rekomendasi Portofolio GA!"}

if __name__ == "__main__":
    import uvicorn
    # Menjalankan server secara lokal
    uvicorn.run("src.backend.app:app", host="127.0.0.1", port=8000, reload=True)