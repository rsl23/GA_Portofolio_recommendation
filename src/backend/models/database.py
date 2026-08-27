from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from src.backend.core.config import settings

# Inisialisasi Engine Database (SQLite sebagai default/contoh)
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Pembuat Sesi Database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class untuk seluruh Model SQLAlchemy
Base = declarative_base()

# Dependency Injection untuk memastikan sesi DB ditutup setelah request selesai
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

