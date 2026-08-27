from sqlalchemy import Column, String, Float, DateTime
from datetime import datetime
# Asumsi Base berasal dari konfigurasi database.py milikmu
from .database import Base 

class FilteredStockCache(Base):
    __tablename__ = "filtered_stock_cache"

    # Kode saham sebagai Primary Key agar tidak ada duplikasi
    kode = Column(String, primary_key=True, index=True)
    nama = Column(String)
    sektor = Column(String)
    
    # Metrik Valuasi & Harga
    close = Column(Float)
    market_cap = Column(Float)
    
    # Metrik Fundamental
    eps = Column(Float)
    roe = Column(Float)
    der = Column(Float)
    
    # Metrik Teknikal
    adtv_60 = Column(Float)
    
    # Waktu pembaruan data
    updated_at = Column(DateTime, default=datetime.now)