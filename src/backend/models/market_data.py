from sqlalchemy import Column, Float, ForeignKey, Date, BigInteger, UniqueConstraint
from sqlalchemy.orm import relationship
from src.backend.models.database import Base
from sqlalchemy.dialects.postgresql import UUID
import uuid


class MarketData(Base):
    """Histori harga harian (OHLCV) untuk saham yang dimiliki user,
    dicatat mulai dari tanggal pembelian dan setiap harinya."""

    __tablename__ = "market_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # saham yang dimiliki user (master data di stock_universe)
    stock_id = Column(UUID(as_uuid=True), ForeignKey("stock_universe.id_stock", ondelete="CASCADE"), nullable=False)

    # tanggal data harga ini (1 baris per saham per hari)
    date = Column(Date, nullable=False)

    # OHLCV harian
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    # satu saham hanya boleh punya satu baris harga per tanggal
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_market_data_stock_date"),
    )

    stock = relationship("StockUniverse")
