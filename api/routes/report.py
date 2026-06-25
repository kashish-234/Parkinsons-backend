from fastapi import APIRouter, HTTPException, Depends
import logging

from api.schemas import ReportResponse
from services.supabase_service import (
    get_result,
    get_report,
    get_modality_embeddings_for_job
)
from services.rag_service import retrieve_similar_cases
from services.report_service import generate_report
from core.auth import verify_user

logger = logging.getLogger(__name__)

router = APIRouter()


# -------------------------------
# GET EXISTING REPORT
# -------------------------------

@router.get("/report/{job_id}", response_model=ReportResponse)
async def get_report_endpoint(job_id: str, user=Depends(verify_user)):

    existing = get_report(job_id)

    if not existing:
        raise HTTPException(status_code=404, detail="Report not found")

    return ReportResponse(
        job_id=job_id,
        report_sections=existing["report"],
        n_similar_cases_used=len(existing.get("retrieved_cases") or []),
        llm_model=existing["llm_model"],
        generated_at=str(existing.get("created_at")),
    )


# -------------------------------
# GENERATE REPORT
# -------------------------------

@router.post("/report/{job_id}", response_model=ReportResponse)
async def generate_report_endpoint(job_id: str, user=Depends(verify_user)):

    # 🔹 return cached report if exists
    existing = get_report(job_id)
    if existing:
        return ReportResponse(
            job_id=job_id,
            report_sections=existing["report"],
            n_similar_cases_used=len(existing.get("retrieved_cases") or []),
            llm_model=existing["llm_model"],
            generated_at=str(existing.get("created_at")),
        )

    # 🔹 fetch result (secure)
    result = get_result(job_id, user["user_id"])
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    # 🔹 get embeddings for RAG
    embeddings = get_modality_embeddings_for_job(job_id)
    if not embeddings:
        raise HTTPException(
            status_code=422,
            detail="No SHAP embeddings found for this job."
        )

    # 🔹 retrieve similar cases
    retrieval_context = retrieve_similar_cases(
        job_id=job_id,
        modality_embeddings=embeddings,
        k_per_modality=5,
    )

    # 🔹 generate report using Gemini
    try:
        sections = generate_report(
            job_id=job_id,
            fused_result=result,
            modality_results=result.get("modality_results", []),
            retrieval_context=retrieval_context,
        )
    except Exception as e:
        logger.error(f"Report generation failed for {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Report generation failed"
        )

    return ReportResponse(
        job_id=job_id,
        report_sections=sections,
        n_similar_cases_used=retrieval_context.n_retrieved,
        llm_model="gemini-1.5-flash",
    )