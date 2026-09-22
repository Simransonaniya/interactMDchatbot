"""
InteractMD — Configuration.
Loads all settings from environment variables / .env file.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env from the same directory as this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)


class Settings(BaseSettings):
    # App
    APP_NAME: str = "InteractMD"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "postgresql+psycopg://interactmd:interactmd@localhost:5432/interactmd"

    # JWT
    JWT_SECRET_KEY: str = "CHANGE_THIS_SECRET_KEY_FOR_PRODUCTION_INTERACTMD_2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:5174"

    # LLM
    LLM_PROVIDER: str = "mock"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
