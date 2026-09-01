"""Environment-based application configuration."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Settings loaded from environment variables without embedding secrets."""

    razorpay_key_id: str | None = os.getenv("RAZORPAY_KEY_ID")
    razorpay_key_secret: str | None = os.getenv("RAZORPAY_KEY_SECRET")
    database_url: str | None = os.getenv("DATABASE_URL")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")


settings = Settings()
