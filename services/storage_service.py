from supabase import create_client
from core.config import settings

def download_file(path: str, local_path: str):
    supabase = create_client(
        settings.supabase_url,
        settings.supabase_service_key
    )

    data = supabase.storage.from_("patient-data").download(path)

    with open(local_path, "wb") as f:
        f.write(data)