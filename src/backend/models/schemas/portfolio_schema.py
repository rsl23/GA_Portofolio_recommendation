from pydantic import BaseModel, Field
from typing import Optional

# Schema untuk Validasi Input (Request)
class PortfolioGenerateRequest(BaseModel):
    user_id: str
    budget: float = Field(..., gt=0, description="Total modal investasi dalam IDR")
    risk_profile: str = Field(..., description="Konservatif, Moderat, atau Agresif")

# Schema untuk Standar Output (Response)
class PortfolioResponse(BaseModel):
    id: str
    fitness_score: float
    sharpe_ratio: Optional[float]
    narasi_llm: Optional[str]
    
    class Config:
        from_attributes = True # Mengizinkan Pydantic membaca dari SQLAlchemy Model

