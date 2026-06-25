import os
import shutil
import logging
from threading import Lock
from huggingface_hub import hf_hub_download
from core.config import settings

logger = logging.getLogger(__name__)
lock = Lock()

class ModelStorageService:

    def __init__(self):
        os.makedirs(settings.model_cache_dir, exist_ok=True)

    def download_model(self, filename: str) -> str:
        local_name = filename.replace("/", "__")
        local_path = os.path.join(settings.model_cache_dir, local_name)

        with lock:
            if os.path.exists(local_path):
                logger.info(f"Cache hit: {filename}")
                return local_path

            logger.info(f"Downloading {filename}...")

            try:
                downloaded = hf_hub_download(
                    repo_id=settings.hf_model_repo,
                    filename=filename,
                    repo_type="model",
                    token=settings.hf_token,
                    local_dir=settings.model_cache_dir,
                )

                shutil.move(downloaded, local_path)
                return local_path

            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise

    def clear_cache(self):
        shutil.rmtree(settings.model_cache_dir, ignore_errors=True)
        os.makedirs(settings.model_cache_dir, exist_ok=True)
        logger.info("Model cache cleared")


model_storage_service = ModelStorageService()