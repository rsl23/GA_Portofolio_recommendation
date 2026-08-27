from fastapi import APIRouter
from src.backend.routes import portfolio_routes, market_routes

api_router = APIRouter()

# Hubungkan semua routes yang terpisah-pisah di sini
api_router.include_router(portfolio_routes.router, prefix="/portfolios", tags=["Portfolios"])
api_router.include_router(market_routes.router, prefix="/market", tags=["Market Data"])

