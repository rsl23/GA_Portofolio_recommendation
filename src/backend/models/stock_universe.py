from sqlalchemy import Column, String, Date
from database import Base # Sesuaikan dengan file koneksi DB-mu

class StockUniverse(Base):
    __tablename__ = "stock_universe"

    ticker = Column(String(10), primary_key=True, index=True)
    nama_perusahaan = Column(String(100))
    listing_date = Column(Date, nullable=False)
    delisting_date = Column(Date, nullable=True) # Kosong jika masih aktif
    relisting_date = Column(Date, nullable=True)