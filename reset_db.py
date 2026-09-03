from src.backend.models.database import engine
from src.backend.models.filtered_stocks_cache import FilteredStockCache

print("Menjatuhkan tabel lama yang tidak memiliki kolom PER, PBV, Dividend Yield...")
FilteredStockCache.__table__.drop(engine, checkfirst=True)

print("Membuat ulang tabel dengan struktur baru...")
FilteredStockCache.__table__.create(engine)

print("Selesai! Tabel berhasil diperbarui.")

