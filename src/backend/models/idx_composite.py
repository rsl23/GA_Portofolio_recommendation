from sqlalchemy import Column, Float, Date, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from src.backend.models.database import Base
import uuid


class IdxComposite(Base):
    """
    Histori harga harian (OHLC) IHSG (^JKSE via yfinance), dipakai sebagai
    benchmark grafik performa portofolio vs IHSG.
    Rentang mengikuti tanggal pembentukan portofolio user hingga terbaru.
    """
    __tablename__ = "idx_composite"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # benchmark index; untuk saat ini selalu "^JKSE" (IHSG)
    ticker = Column(String(10), nullable=False, default="^JKSE")

    # tanggal data harga ini (1 baris per index per hari)
    date = Column(Date, nullable=False)

    # OHLC harian (IHSG tidak punya volume yang dipakai untuk perbandingan)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)

    # satu index hanya boleh punya satu baris harga per tanggal
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_idx_composite_ticker_date"),
    )
