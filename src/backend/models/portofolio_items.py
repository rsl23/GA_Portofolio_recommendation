from sqlalchemy import Column, Float, ForeignKey, Integer, String, DateTime 
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class PortofolioItem(Base):
    __tablename__ = "portofolio_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portofolio_id = Column(UUID(as_uuid=True), ForeignKey("portofolios.id", ondelete="CASCADE"), nullable=False)
    
    ticker = Column(String, nullable=False)
    bobot_persentase = Column(Float, nullable=False)
    jumlah_lot = Column(Integer, nullable=False)
    
    harga_acuan = Column(Float, nullable=False)
    total_investasi = Column(Float, nullable=False)
    
    action_type = Column(String, nullable=False)  # buy, sell, hold, add-more, reduce, etc.
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    portofolio = relationship("Portofolio", back_populates="items")