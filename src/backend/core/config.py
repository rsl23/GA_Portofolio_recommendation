import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "GA Portfolio Recommendation API"
    X_API_KEY: str = os.getenv("X_API_KEY", "")
    BASE_URL: str = os.getenv("BASE_URL", "https://api.zpi.web.id/v1")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./portfolio.db")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "genfolio_ta")
    
    class Config:
        env_file = ".env"

settings = Settings()

