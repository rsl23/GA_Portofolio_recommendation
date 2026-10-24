from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Generic, TypeVar, Literal
from datetime import date, datetime

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
    name_portofolio: Optional[str] = Field(None, max_length=80, description="Nama portofolio agar mudah dikenali")
    description: Optional[str] = Field(None, max_length=500, description="Catatan pribadi mengenai tujuan portofolio")
    backtest: bool = Field(False, description="Jika True, lakukan backtest portofolio terhadap data historis")
    date_ref: Optional[date] = Field(
        None,
        description="Tanggal acuan untuk backtest (format: YYYY-MM-DD). "
                    "Pydantic mengubah string ISO ini menjadi objek date, sehingga "
                    "langsung bisa dipakai oleh run_live_preprocessing/build_market_data.",
    )
    rebalance: bool = Field(False, description="Jika True, lakukan rebalancing dari portofolio aktif saat ini")
    contribution_amount: float = Field(0, ge=0, description="Setoran tambahan pada rebalance backtest (IDR)")
    bear_management: bool = False
    bear_choice: Optional[Literal["recommended", "full"]] = None
    bear_preview_as_of: Optional[date] = None
    bear_preview_fingerprint: Optional[str] = None

    @field_validator("name_portofolio")
    @classmethod
    def validate_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("Nama portofolio tidak boleh kosong.")
        return value.strip() if value is not None else None

    @field_validator("description")
    @classmethod
    def clean_description(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() or None if value is not None else None


class AddCapitalRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Setoran modal live dalam IDR; tersedia pada rebalance berikutnya")

# Satu baris alokasi saham di dalam portofolio hasil GA
class PortfolioItem(BaseModel):
    item_id: str                   # ID item di portofolio_items (untuk PATCH harga_beli)
    ticker: str                    # kode saham, misal BBCA
    lots: int                      # jumlah lot (1 lot = 100 lembar)
    price_per_lot: float           # harga 1 lot saat ini (100 lembar)
    harga_beli: Optional[float] = None  # harga beli aktual user per lembar (bisa diedit)
    allocation: float              # alokasi dana = lots * price_per_lot
    weight: float                  # bobot alokasi terhadap total terpakai (0-1)


# Request ubah harga beli milik user pada satu item portofolio
class UpdateHargaBeliRequest(BaseModel):
    harga_beli: float = Field(..., gt=0, description="Harga beli aktual per lembar (IDR)")


# Baris riwayat portofolio (ringkasan, tanpa alokasi detail)
class PortfolioHistoryItem(BaseModel):
    id: str
    name_portofolio: Optional[str] = None
    description: Optional[str] = None
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
# Baris detail perubahan (delta) saat rebalancing
class RebalanceDeltaItem(BaseModel):
    kode: str
    action: str
    old_lot: int
    new_lot: int
    delta_lot: int
    reason: str
    action_detail: Optional[str] = None
    unrealized_return_pct_before: Optional[float] = None
    realized_pnl: Optional[float] = None
    realization_type: Optional[str] = None


class PortfolioResponse(BaseModel):
    id: str
    parent_portofolio_id: Optional[str] = None
    execution_date: Optional[date] = None
    name_portofolio: Optional[str] = None
    description: Optional[str] = None
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
    date_ref: Optional[datetime] = None           # acuan tanggal portofolio (dari DB):
                                                  # backtest = tanggal simulasi, live = waktu generate
    budget: float
    pending_deposit: float = 0
    capital_contribution: float = 0
    allocations: List[PortfolioItem] = []
    narasi_llm: Optional[str]
    rebalance_deltas: Optional[List[RebalanceDeltaItem]] = None

    class Config:
        from_attributes = True # Mengizinkan Pydantic membaca dari SQLAlchemy Model


# ----------------------------------------------------------------------
# Rebalance series (banyak rebalance berantai pada rentang tanggal)
# ----------------------------------------------------------------------
# Request untuk POST /generate-rebalance-series: generate portofolio awal pada
# start_date, lalu rebalance berantai tiap interval_months sampai end_date.
# Jumlah portofolio yang dihasilkan TIDAK angka pasti -- melainkan berapa kali
# tanggal dalam rentang [start_date, end_date] kelipatan interval (mis. tiap 2
# bulan pada 2022-2026 -> lebih dari 10 portofolio).
class RebalanceSeriesGenerateRequest(BaseModel):
    budget: float = Field(..., gt=0, description="Modal awal portofolio pada start_date (IDR)")
    risk_profile: str = Field(..., description="Konservatif, Moderat, atau Agresif")
    name_portofolio: Optional[str] = Field(None, max_length=80)
    description: Optional[str] = Field(None, max_length=500)
    start_date: date = Field(
        ..., description="Tanggal generate portofolio awal (YYYY-MM-DD), mis. 2022-01-03"
    )
    end_date: date = Field(
        ..., description="Tanggal terakhir series (YYYY-MM-DD, inklusif), mis. 2026-12-31"
    )
    interval_months: int = Field(
        2, ge=1, le=12,
        description="Jeda antar rebalance dalam bulan (2 = tiap 2 bulan).",
    )
    contribution_amount: float = Field(0, ge=0, description="Nominal setoran berkala (IDR)")
    contribution_interval_months: Optional[int] = Field(None, ge=1, le=12, description="Setor tiap N bulan sejak start_date, dimulai bulan berikutnya")
    bear_management: bool = False
    bear_choices: Optional[dict[date, Literal["recommended", "full"]]] = None
    bear_preview_dates: Optional[dict[date, date]] = None
    bear_preview_fingerprints: Optional[dict[date, str]] = None

    @field_validator("name_portofolio")
    @classmethod
    def validate_name(cls, value: Optional[str]) -> Optional[str]:
        return PortfolioGenerateRequest.validate_name(value)

    @field_validator("description")
    @classmethod
    def clean_description(cls, value: Optional[str]) -> Optional[str]:
        return PortfolioGenerateRequest.clean_description(value)


class SkippedRebalanceStep(BaseModel):
    """Satu tanggal series yang dilewati karena MarketData tidak tersedia."""
    date: date
    reason: str


class RebalanceSeriesResponse(BaseModel):
    """Hasil series rebalance berantai: SEMUA portofolio urut kronologis."""
    start_date: date
    end_date: date
    interval_months: int
    scheduled_dates: List[date]              # seluruh tanggal terjadwal
    portfolios: List[PortfolioResponse]      # yang berhasil, urut dari start_date
    skipped: List[SkippedRebalanceStep] = [] # tanggal yang dilewati + alasan
    evaluation_mode: str = "legacy"  # automatic_strategy / what_if / legacy
