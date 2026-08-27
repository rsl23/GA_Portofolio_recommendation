from fastapi import FastAPI
from src.backend.routes.api import api_router
from src.backend.core.config import settings
from src.backend.models.database import engine, Base
import src.backend.models.filtered_stocks_cache  # Memastikan model terdeteksi sebelum create_all

# Pastikan semua tabel di-generate ke database saat pertama kali jalan
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for GA Portfolio Recommendation with LLM",
    version="1.0.0"
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

