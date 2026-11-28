from sqlalchemy import Column, String, Float, DateTime, Date
from datetime import datetime
from sqlalchemy.sql import func
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
    price_date = Column(Date, nullable=True)  # tanggal Close dari IDX summary, bukan updated_at
    market_cap = Column(Float)
    
    # Metrik Fundamental
    # SATUAN (penting, dipakai bersama preprocessing & data_loader):
    #   per/pbv        : kelipatan (mis. 13.34)
    #   eps            : rupiah per saham (mis. 466.74)
    #   roe            : FRAKSI (mis. 0.2044 = 20.44%)
    #   der            : RASIO  (mis. 0.0753 = 7.53%); None bila tidak tersedia.
    # Filtering menormalkan satuan sumber sebelum menyimpan; data_loader LIVE
    # membaca rasio ini langsung, setara dengan jalur backtest.
    per = Column(Float)
    pbv = Column(Float)
    eps = Column(Float)
    roe = Column(Float)
    der = Column(Float)
    
    # dividend_yield : FRAKSI (mis. 0.0612 = 6.12%), sama seperti backtest.
    dividend_yield = Column(Float, nullable=True)
    
    # Metrik Teknikal
    adtv_60 = Column(Float)
    
    # Waktu pembaruan data
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
