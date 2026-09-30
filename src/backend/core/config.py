import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "GA Portfolio Recommendation API"
    X_API_KEY: str = os.getenv("X_API_KEY", "")
    BASE_URL: str = os.getenv("BASE_URL", "https://api.zpi.web.id/v1")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./portfolio.db")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "genfolio_ta_secret_key_min_32_bytes_long!!")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ------------------------------------------------------------------
    # LLM (Gemini) untuk narasi penjelasan hasil GA.
    # Dipakai oleh src/backend/services/gemini_llm_caller.py; bila
    # GEMINI_API_KEY kosong / GEMINI_ENABLED=false, narasi LLM dilewati dan
    # controller memakai narasi template (lihat portfolio_controller).
    # ------------------------------------------------------------------
    GEMINI_API_KEY: str = ""
    GEMINI_ENABLED: bool = True
    GEMINI_MODEL: str = "gemini-3.8-flash"
    GEMINI_TEMPERATURE: float = 0.7
    GEMINI_MAX_OUTPUT_TOKENS: int = 8192
    GEMINI_THINKING_LEVEL: str = "low"
    GEMINI_TIMEOUT_MS: int = 120_000
    GEMINI_MAX_RETRIES: int = 3

    class Config:
        env_file = ".env"
        # Variabel .env yang tidak dideklarasikan di atas (mis. FUNDAMENTAL_MAX_RETRIES,
        # RETRY_BACKOFF_BASE, dsb.) TIDAK boleh menggagalkan startup aplikasi.
        extra = "ignore"

settings = Settings()

