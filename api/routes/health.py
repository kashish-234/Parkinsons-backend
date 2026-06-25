from fastapi import APIRouter, HTTPException
import logging
from core.config import settings
from services.supabase_service import get_client

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
def health():
    status = {
        "status": "ok",
        "environment": settings.environment,
        "supabase": "unknown",
        "gemini": "unknown",
    }
    try:
        db = get_client()
        db.table("fused_results").select("job_id").limit(1).execute()
        status["supabase"] = "connected"
    except Exception as e:
        logger.error(f"Supabase check failed: {e}")
        status["supabase"] = "error"

    # 🔹 Check Gemini (basic check)
    if settings.gemini_api_key:
        status["gemini"] = "configured"
    else:
        status["gemini"] = "missing_key"

    return status