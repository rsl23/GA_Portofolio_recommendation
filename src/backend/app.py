from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.backend.routes.api import api_router
from src.backend.core.config import settings
from src.backend.models.database import engine, Base
import src.backend.models.filtered_stocks_cache
from src.gaengine.data_loader_live import build_market_data

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

# Daftarkan semua route ke dalam API dengan prefix standar
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Selamat datang di API Sistem Rekomendasi Portofolio GA!"}

if __name__ == "__main__":
    import uvicorn
    # Menjalankan server secara lokal
    uvicorn.run("src.backend.app:app", host="127.0.0.1", port=8000, reload=True)