from core.config import settings
from services.supabase_service import get_client

def download_file(path: str, local_path: str):
    supabase = get_client()  # use the singleton
    data = supabase.storage.from_("patient-data").download(path)
    with open(local_path, "wb") as f:
        f.write(data)