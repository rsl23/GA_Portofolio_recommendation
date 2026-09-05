from pydantic import BaseModel, Field
from typing import List, Optional, Generic, TypeVar

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

# Schema untuk Standar Output (Response)
class PortfolioResponse(BaseModel):
    id: str
    fitness_score: float
    sharpe_ratio: Optional[float]
    expected_return: Optional[float]
    max_drawdown: Optional[float]
    avg_correlation: Optional[float]
    skor_fundamental: Optional[float]
    total_terpakai: Optional[float]
    sisa_budget: Optional[float]
    n_active: Optional[int]
    allocated_budget_ok: Optional[bool]
    risk_profile: str
    budget: float
    allocations: List[PortfolioItem] = []
    narasi_llm: Optional[str]

    class Config:
        from_attributes = True # Mengizinkan Pydantic membaca dari SQLAlchemy Model
