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
    DATABASE_URL: str = "sqlite:///./interactmd.db"
    MONGODB_URI: str = ""
    MONGODB_DB_NAME: str = "interactmd"

    # JWT
    JWT_SECRET_KEY: str = "CHANGE_THIS_SECRET_KEY_FOR_PRODUCTION_INTERACTMD_2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:5174,http://localhost:3000,http://127.0.0.1:5173"

    # LLM
    LLM_PROVIDER: str = "huggingface"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    HF_TOKEN: str = ""
    HF_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct"
    HF_API_URL: str = ""
    HF_TEMPERATURE: float = 0.2
    HF_MAX_NEW_TOKENS: int = 150
    HF_TOP_P: float = 0.9
    HF_REPETITION_PENALTY: float = 1.1

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

