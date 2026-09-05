from sqlalchemy import Column, String, Date
from sqlalchemy.dialects.postgresql import UUID
import uuid
from .database import Base # Sesuaikan dengan file koneksi DB-mu

class StockUniverse(Base):
    __tablename__ = "stock_universe"

    # UUID sebagai primary key (migrasi dari PK ticker)
    id_stock = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # ticker kini kolom biasa: wajib terisi & unik (dulunya PK)
    ticker = Column(String(10), nullable=False, unique=True, index=True)
    nama_perusahaan = Column(String(100))
    listing_date = Column(Date, nullable=False)
    delisting_date = Column(Date, nullable=True) # Kosong jika masih aktif
    relisting_date = Column(Date, nullable=True)
