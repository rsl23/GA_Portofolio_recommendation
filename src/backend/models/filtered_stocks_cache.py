from sqlalchemy import Column, String, Float, DateTime
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
    market_cap = Column(Float)
    
    # Metrik Fundamental
    # SATUAN (penting, dipakai bersama preprocessing & data_loader):
    #   per/pbv        : kelipatan (mis. 13.34)
    #   eps            : rupiah per saham (mis. 466.74)
    #   roe            : PERSEN   (mis. 20.44)
    #   der            : PERSEN   (mis. 7.53)  -> data_loader LIVE mengubahnya
    #                    menjadi RASIO (persen/100) saat merakit metrics_map
    #                    agar setara dengan jalur backtest.
    per = Column(Float)
    pbv = Column(Float)
    eps = Column(Float)
    roe = Column(Float)
    der = Column(Float)
    
    # dividend_yield : PERSEN (mis. 6.12) -> data_loader LIVE mengubahnya
    #                  menjadi FRAKSI (persen/100), sama seperti backtest.
    dividend_yield = Column(Float, default=0.0)
    
    # Metrik Teknikal
    adtv_60 = Column(Float)
    
    # Waktu pembaruan data
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    