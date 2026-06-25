from fastapi import APIRouter, HTTPException, Depends
import logging

from api.schemas import ResultResponse
from services.supabase_service import get_result
from core.auth import verify_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/results/{job_id}", response_model=ResultResponse)
async def get_result_endpoint(job_id: str, user=Depends(verify_user)):

    # secure fetch (user isolation)
    result = get_result(job_id, user["user_id"])

    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    return ResultResponse(**result)