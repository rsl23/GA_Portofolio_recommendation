from sqlalchemy import Boolean, Column, Float, ForeignKey, String, DateTime, Text 
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class Portofolio(Base):
    __tablename__ = 'portofolios'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # metrik budget
    budget = Column(Float, nullable=False)
    total_terpakai = Column(Float, nullable=False)
    sisa_budget = Column(Float, nullable=False)
    
    # metrik fitness
    fitness_score = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    avg_correlation = Column(Float, nullable=False)
    skor_fundamental = Column(Float, nullable=False)
    
    # bobot pengali fitness
    mdd_lambda = Column(Float, nullable=False, default=1.0)
    avg_korelasi_gamma = Column(Float, nullable=False, default=1.0)
    funda_alpha = Column(Float, nullable=False, default=1.0)
    
    # llm
    narasi_llm = Column(Text, nullable=True)
    
    # status dan penjelasan
    status_portofolio = Column(String, nullable=False, default="active")  # active, rebalanced, inactive, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # rebalance
    is_rebalance = Column(Boolean, nullable=False, default=False)  # yes or no
    parent_portofolio_id = Column(UUID(as_uuid=True), ForeignKey("portofolios.id", ondelete="SET NULL"), nullable=True)
    turnover_rate = Column(Float, nullable=True)
    turnover_penalty_beta = Column(Float, nullable=True)  # penalty for turnover during rebalance
    
    user = relationship("User", back_populates="portofolios")
    items = relationship("PortofolioItem", back_populates="portofolio", cascade="all, delete-orphan")