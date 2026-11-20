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
    # LLM (OpenAI Responses API) untuk narasi penjelasan hasil GA.
    # OPENAI_ENABLED=false memakai narasi template.
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = ""
    OPENAI_ENABLED: bool = True
    OPENAI_MODEL: str = "gpt-6-luna"
    OPENAI_REASONING_EFFORT: str = "medium"
    OPENAI_MAX_OUTPUT_TOKENS: int = 4096
    OPENAI_TIMEOUT_MS: int = 120_000
    OPENAI_MAX_RETRIES: int = 3
    OPENAI_RETRY_ENABLED: bool = False
    OPENAI_RETURN_ERROR: bool = False
    OPENAI_FALLBACK_MODELS: str = ""
    OPENAI_BACKOFF_BASE: float = 2.0
    OPENAI_BACKOFF_CAP: float = 30.0

    class Config:
        env_file = ".env"
        # Variabel .env yang tidak dideklarasikan di atas (mis. FUNDAMENTAL_MAX_RETRIES,
        # RETRY_BACKOFF_BASE, dsb.) TIDAK boleh menggagalkan startup aplikasi.
        extra = "ignore"

settings = Settings()

