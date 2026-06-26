from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict
import logging

from api.schemas import ReportResponse
from services.supabase_service import (
    get_result,
    get_report,
    get_modality_embeddings_for_job,
)
from services.rag_service import retrieve_similar_cases
from services.report_service import generate_report
from core.auth import verify_user

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_report_response(job_id: str, existing: dict) -> ReportResponse:
    """Build a ReportResponse from a DB row returned by get_report()."""
    raw = existing.get("report_sections") or {}
    # Ensure it's Dict[str, str] whether stored as dict or Pydantic model
    if isinstance(raw, dict):
        sections: Dict[str, str] = {k: str(v) for k, v in raw.items()}
    else:
        sections = {}

    return ReportResponse(
        job_id=job_id,
        report_sections=sections,
        n_similar_cases_used=len(existing.get("retrieved_cases") or []),
        llm_model=existing.get("llm_model", "gemini"),
        generated_at=existing.get("generated_at"),
    )


async def _generate_report_for_job(job_id: str, user_id: str) -> ReportResponse:
    """Core logic shared by both POST endpoints."""

    # Return cached report if one already exists
    existing = get_report(job_id)
    if existing:
        return _build_report_response(job_id, existing)

    # Fetch the fused result (user-isolated)
    result = get_result(job_id, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get SHAP embeddings for RAG retrieval
    embeddings = get_modality_embeddings_for_job(job_id)
    if not embeddings:
        raise HTTPException(
            status_code=422,
            detail="No SHAP embeddings found for this job.",
        )

    # Retrieve similar historical cases
    retrieval_context = retrieve_similar_cases(
        job_id=job_id,
        modality_embeddings=embeddings,
        k_per_modality=5,
    )

    # Generate report via Gemini
    try:
        sections = generate_report(
            job_id=job_id,
            fused_result=result,
            modality_results=result.get("modality_results", []),
            retrieval_context=retrieval_context,
        )
    except Exception as e:
        logger.error(f"Report generation failed for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Report generation failed")

    return ReportResponse(
        job_id=job_id,
        report_sections=sections,
        n_similar_cases_used=retrieval_context.n_retrieved,
        llm_model="gemini-1.5-flash",
    )


# ── GET existing report ───────────────────────────────────────────────────────

@router.get("/report/{job_id}", response_model=ReportResponse)
async def get_report_endpoint(job_id: str, user=Depends(verify_user)):
    existing = get_report(job_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Report not found")
    return _build_report_response(job_id, existing)


# ── POST /api/report/{job_id}  (original route) ───────────────────────────────

@router.post("/report/{job_id}", response_model=ReportResponse)
async def generate_report_endpoint(job_id: str, user=Depends(verify_user)):
    return await _generate_report_for_job(job_id, user["user_id"])


# ── POST /api/reports/generate  (frontend-compatible alias) ───────────────────
# The frontend calls POST /api/reports/generate with body {"job_id": "..."}.

class GenerateReportBody(BaseModel):
    job_id: str


@router.post("/reports/generate", response_model=ReportResponse)
async def generate_report_alias(body: GenerateReportBody, user=Depends(verify_user)):
    return await _generate_report_for_job(body.job_id, user["user_id"])
