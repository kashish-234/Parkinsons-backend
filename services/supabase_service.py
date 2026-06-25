from supabase import create_client
from models.base.contracts import FusedResult
import logging

logger = logging.getLogger(__name__)

from core.config import settings

_client = None


def get_client():
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _client


# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------

def persist_fused_result(result: FusedResult, user_id: str):
    db = get_client()
    try:
        db.table("fused_results").insert({
            "job_id":               result.job_id,
            "user_id":              user_id,
            "patient_id":           result.patient_id,
            "probability":          result.probability,
            "risk_label":           result.risk_label,
            "ci_low":               result.ci_low,
            "ci_high":              result.ci_high,
            "modality_weights":     result.modality_weights,
            "fusion_model_version": result.fusion_model_version,
        }).execute()

        for mr in result.modality_results:
            if not mr.available:
                continue

            db.table("modality_results").insert({
                "job_id":        result.job_id,
                "modality":      mr.modality,
                "probability":   mr.probability,
                "shap_features": [
                    {"name": f.name, "value": f.value, "rank": f.rank}
                    for f in mr.shap_features
                ],
                "available":     mr.available,
            }).execute()

            vec = [f.value for f in mr.shap_features[:10]]
            vec += [0.0] * (10 - len(vec))
            db.table("shap_embeddings").insert({
                "job_id":    result.job_id,
                "modality":  mr.modality,
                "content":   f"{mr.modality} SHAP features for job {result.job_id}",
                "embedding": vec,
            }).execute()

        logger.info(f"Persisted job {result.job_id} for user {user_id}")

    except Exception as e:
        logger.error(f"Persist failed for job {result.job_id}: {e}")
        raise


# ---------------------------------------------------------------------------
# FETCH RESULT
# FIX H4: Also fetch modality_results rows and attach them.
# ---------------------------------------------------------------------------

def get_result(job_id: str, user_id: str):
    db = get_client()

    res = (
        db.table("fused_results")
        .select("*")
        .eq("job_id", job_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not res.data:
        return None

    row = res.data[0]

    # Attach modality_results from the separate table
    mr_res = (
        db.table("modality_results")
        .select("*")
        .eq("job_id", job_id)
        .execute()
    )
    row["modality_results"] = mr_res.data if mr_res.data else []

    return row


# ---------------------------------------------------------------------------
# FETCH REPORT
# ---------------------------------------------------------------------------

def get_report(job_id: str):
    db = get_client()
    res = (
        db.table("clinical_reports")
        .select("*")
        .eq("job_id", job_id)
        .order("generated_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


# ---------------------------------------------------------------------------
# FETCH SHAP EMBEDDINGS FOR RAG
# ---------------------------------------------------------------------------

def get_modality_embeddings_for_job(job_id: str) -> dict:
    db = get_client()
    res = (
        db.table("shap_embeddings")
        .select("modality, embedding")
        .eq("job_id", job_id)
        .execute()
    )
    if not res.data:
        return {}
    return {row["modality"]: row["embedding"] for row in res.data}