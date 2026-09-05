from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from src.backend.routes.api import api_router
from src.backend.core.config import settings
from src.backend.models.database import engine, Base
import src.backend.models.filtered_stocks_cache
from src.gaengine.data_loader_live import build_market_data
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Generate tabel saat startup (lebih aman di sini daripada global scope)
    print("Mengecek dan membuat tabel database...")
    Base.metadata.create_all(bind=engine)
    
    # 2. Load data pasar ke memori untuk engine Algoritma Genetika
    print("Memuat MarketData ke RAM...")
    app.state.market_data_today = build_market_data()
    
    yield
    
    # Clean up saat server dimatikan
    app.state.market_data_today = None

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