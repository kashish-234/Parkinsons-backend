from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    # Supabase
    supabase_url: str
    supabase_service_key: str
    supabase_anon_key: str        # needed for JWKS fetch (/auth/v1/keys requires apikey header)

    # Gemini (GenAI)
    gemini_api_key: str

    # HuggingFace — token is optional for public repos
    hf_token: Optional[str] = None
    hf_model_repo: str = "Kashish-jain/Parkinsons-trained-models"

    # Optional
    cron_secret: Optional[str] = None

    # App config
    environment: str = "development"
    debug: bool = False

    # Model cache
    # Render free tier: /tmp is ephemeral (reset on restart / spin-up).
    # To avoid re-downloading models on every cold start, attach a Render
    # Persistent Disk and set MODEL_CACHE_DIR=/mnt/pd_models in the env.
    model_cache_dir: str = "/tmp/pd_models"

    # Server
    port: int = 8000

    class Config:
        env_file = Path(__file__).resolve().parent.parent / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
