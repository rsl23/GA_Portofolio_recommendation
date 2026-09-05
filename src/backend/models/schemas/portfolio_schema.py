from pydantic import BaseModel, Field
from typing import List, Optional, Generic, TypeVar
from datetime import datetime

# Tipe generik untuk payload di dalam envelope ApiResponse
T = TypeVar("T")


# Envelope respons seragam untuk seluruh endpoint API.
# Dipakai dengan mengisi T, contoh: ApiResponse[PortfolioResponse]
class ApiResponse(BaseModel, Generic[T]):
    status: str = "success"          # "success" atau "error"
    message: str = ""                # deskripsi singkat hasil operasi
    data: Optional[T] = None         # payload utama (None jika error)

# Schema untuk Validasi Input (Request)
class PortfolioGenerateRequest(BaseModel):
    budget: float = Field(..., gt=0, description="Total modal investasi dalam IDR")
    risk_profile: str = Field(..., description="Konservatif, Moderat, atau Agresif")

# Satu baris alokasi saham di dalam portofolio hasil GA
class PortfolioItem(BaseModel):
    ticker: str                    # kode saham, misal BBCA
    lots: int                      # jumlah lot (1 lot = 100 lembar)
    price_per_lot: float           # harga 1 lot saat ini
    allocation: float              # alokasi dana = lots * price_per_lot
    weight: float                  # bobot alokasi terhadap total terpakai (0-1)


# Baris riwayat portofolio (ringkasan, tanpa alokasi detail)
class PortfolioHistoryItem(BaseModel):
    id: str
    budget: float
    total_terpakai: Optional[float]
    sisa_budget: Optional[float]
    fitness_score: float
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    risk_profile: str
    status_portofolio: str
    created_at: Optional[datetime] = None

# Schema untuk Standar Output (Response)
# Dipakai BERSAMA oleh generate (POST /generate), ambil aktif (GET /my-portofolio),
# dan ambil by id. Field yang tidak tersimpan di DB (expected_return, n_active,
# allocated_budget_ok) bersifat Optional — None saat dibaca dari database.
class PortfolioResponse(BaseModel):
    id: str
    user_id: Optional[str] = None                 # pemilik portofolio (dari DB)
    fitness_score: float
    sharpe_ratio: Optional[float]
    expected_return: Optional[float]              # hanya ada saat hasil GA (tidak disimpan)
    max_drawdown: Optional[float]
    avg_correlation: Optional[float]
    skor_fundamental: Optional[float]
    total_terpakai: Optional[float]
    sisa_budget: Optional[float]
    n_active: Optional[int]                       # jumlah saham aktif
    allocated_budget_ok: Optional[bool]           # apakah alokasi muat dalam budget
    risk_profile: str
    status_portofolio: Optional[str] = None       # active / replaced / dst (dari DB)
    created_at: Optional[datetime] = None         # dari DB (None saat hasil GA sebelum refresh)
    budget: float
    allocations: List[PortfolioItem] = []
    narasi_llm: Optional[str]

    class Config:
        from_attributes = True # Mengizinkan Pydantic membaca dari SQLAlchemy Model
