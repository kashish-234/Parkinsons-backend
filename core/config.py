from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    # Supabase
    supabase_url: str
    supabase_service_key: str

    # Gemini (GenAI)
    gemini_api_key: str

    # HuggingFace
    hf_token: str
    hf_model_repo: str = "your-org/pd-models"

    # Optional
    cron_secret: Optional[str] = None

    # App config
    environment: str = "development"
    debug: bool = False

    # Model cache
    model_cache_dir: str = "/tmp/pd_models"

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"


settings = Settings()