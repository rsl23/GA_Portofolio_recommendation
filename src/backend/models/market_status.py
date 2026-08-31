from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.sql import func

from src.backend.models.database import Base

class DailyMarketStatus(Base):
    __tablename__ = "daily_market_status"

    id = Column(String, primary_key=True)
    is_bear_market = Column(Boolean, default=False)
    risk_free_rate = Column(Float, default=0.06) 
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())