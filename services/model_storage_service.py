import os
import shutil
import logging
import tempfile
import threading
from huggingface_hub import hf_hub_download
from core.config import settings

logger = logging.getLogger(__name__)

# One lock per HF filename; avoids blocking unrelated downloads.
_file_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _get_file_lock(filename: str) -> threading.Lock:
    with _registry_lock:
        if filename not in _file_locks:
            _file_locks[filename] = threading.Lock()
        return _file_locks[filename]


class ModelStorageService:

    def __init__(self):
        os.makedirs(settings.model_cache_dir, exist_ok=True)

    def _local_path(self, filename: str) -> str:
        local_name = filename.replace("/", "__")
        return os.path.join(settings.model_cache_dir, local_name)

    def download_model(self, filename: str) -> str:
        """
        Return local path to the cached artifact, downloading if needed.

        Thread-safe: uses a per-file lock so parallel requests for
        different files don't block each other.
        """
        local_path = self._local_path(filename)

        # Fast path: already cached (no lock needed for read)
        if os.path.exists(local_path):
            logger.debug(f"Cache hit: {filename}")
            return local_path

        lock = _get_file_lock(filename)
        with lock:
            # Double-check inside lock (another thread may have downloaded)
            if os.path.exists(local_path):
                return local_path

            logger.info(f"Downloading from HF: {filename}")
            try:
                downloaded = hf_hub_download(
                    repo_id=settings.hf_model_repo,
                    filename=filename,
                    repo_type="model",
                    token=settings.hf_token,
                    cache_dir=os.path.join(settings.model_cache_dir, ".hf_cache"),
                )

                # Atomic move: avoids a partial file being used if the
                # process is interrupted mid-copy.
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=settings.model_cache_dir, suffix=".tmp"
                )
                os.close(tmp_fd)
                shutil.copy2(downloaded, tmp_path)
                os.replace(tmp_path, local_path)  # atomic on POSIX

                logger.info(f"Downloaded and cached: {filename} → {local_path}")
                return local_path

            except Exception as e:
                logger.error(f"Download failed for {filename}: {e}")
                raise

    def clear_cache(self):
        shutil.rmtree(settings.model_cache_dir, ignore_errors=True)
        os.makedirs(settings.model_cache_dir, exist_ok=True)
        with _registry_lock:
            _file_locks.clear()
        logger.info("Model cache cleared")


model_storage_service = ModelStorageService()