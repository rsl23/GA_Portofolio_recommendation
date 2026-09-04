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
    
    class Config:
        env_file = ".env"

settings = Settings()

