from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from src.backend.models.database import get_db
from src.backend.models.schemas.portfolio_schema import PortfolioGenerateRequest, PortfolioResponse
from src.backend.controller.portfolio_controller import generate_new_portfolio

router = APIRouter()

@router.post("/generate", response_model=PortfolioResponse)
def api_generate_portfolio(request: Request, body: PortfolioGenerateRequest, db: Session = Depends(get_db)):
    """
    Endpoint untuk membuat/mengenerate portofolio baru bagi pengguna.
    Menerima modal dan profil risiko, mengembalikan rekomendasi portofolio beserta alokasi
    dan narasi singkat. Data market live diambil dari app.state (dimuat saat startup);
    jika kosong, controller akan membangun MarketData sendiri.
    """
    # Data market sudah dimuat di app.state saat startup oleh lifespan (app.py)
    market_data = request.app.state.market_data_today
    hasil = generate_new_portfolio(db, body, market_data)
    return hasil
