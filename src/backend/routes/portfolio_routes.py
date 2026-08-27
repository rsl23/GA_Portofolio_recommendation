from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.backend.models.database import get_db
from src.backend.models.schemas.portfolio_schema import PortfolioGenerateRequest, PortfolioResponse
from src.backend.controller.portfolio_controller import generate_new_portfolio

router = APIRouter()

@router.post("/generate", response_model=PortfolioResponse)
def api_generate_portfolio(request: PortfolioGenerateRequest, db: Session = Depends(get_db)):
    """
    Endpoint untuk membuat/mengenerate portofolio baru bagi pengguna.
    Menerima modal dan profil risiko, mengembalikan rekomendasi portofolio beserta narasi LLM.
    """
    hasil = generate_new_portfolio(db, request)
    return hasil

