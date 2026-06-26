import logging
import time
from supabase import create_client, Client
from models.base.contracts import FusedResult

logger = logging.getLogger(__name__)

from core.config import settings

_client: Client | None = None
_client_created_at: float = 0
# Re-create the client every 30 min to avoid stale connections on Render
_CLIENT_TTL_SECONDS = 1800


def get_client() -> Client:
    global _client, _client_created_at
    if _client is None or time.time() - _client_created_at > _CLIENT_TTL_SECONDS:
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
        _client_created_at = time.time()
    return _client


# ── Save results ────────────────────────────────────────────────────────────

def persist_fused_result(result: FusedResult, user_id: str):
    db = get_client()
    try:
        db.table("fused_results").insert({
            "job_id":  result.job_id,
            "user_id": user_id,
            "patient_id": result.patient_id,
            "patient_uuid": getattr(result, "patient_uuid", None),
            "probability":          result.probability,
            "risk_label":           result.risk_label,
            "ci_low":               result.ci_low,
            "ci_high":              result.ci_high,
            "modality_weights":     result.modality_weights,
            "fusion_model_version": result.fusion_model_version,
            "report_json":          result.report_json,
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

            # Build a fixed-length float vector for pgvector (dim=10)
            shap_vals = [float(f.value) for f in mr.shap_features[:10]]
            shap_vals += [0.0] * (10 - len(shap_vals))  # zero-pad to 10

            db.table("shap_embeddings").insert({
                "job_id":    result.job_id,
                "modality":  mr.modality,
                "content":   f"{mr.modality} SHAP features for job {result.job_id}",
                "embedding": shap_vals,  # list[float] — pgvector expects this
            }).execute()

        logger.info("Persisted job %s for patient %s",result.job_id,result.patient_id,)

    except Exception as e:
        logger.exception(f"Persist failed for job {result.job_id}: {e}")
        raise


# ── Fetch result ─────────────────────────────────────────────────────────────

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

    mr_res = (
        db.table("modality_results")
        .select("*")
        .eq("job_id", job_id)
        .execute()
    )
    row["modality_results"] = mr_res.data if mr_res.data else []

    return row


# ── Fetch report ──────────────────────────────────────────────────────────────

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


# ── Fetch SHAP embeddings for RAG ─────────────────────────────────────────────

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