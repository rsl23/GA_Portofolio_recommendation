from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from src.backend.models.database import Base
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    risk_profile = Column(String(50), nullable=True, default="Moderate")  # Conservative, Moderate, Aggressive
    
    portofolios = relationship("Portofolio", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"