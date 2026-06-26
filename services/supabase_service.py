import logging
import time
from supabase import create_client, Client
from models.base.contracts import FusedResult

logger = logging.getLogger(__name__)

from core.config import settings

_client: Client | None = None
_client_created_at: float = 0
_CLIENT_TTL_SECONDS = 1800  # re-create every 30 min to avoid stale connections


def get_client() -> Client:
    global _client, _client_created_at
    if _client is None or time.time() - _client_created_at > _CLIENT_TTL_SECONDS:
        _client = create_client(settings.supabase_url, settings.supabase_service_key)
        _client_created_at = time.time()
    return _client


# ── SHAP embedding dimension ───────────────────────────────────────────────────
# Must match the vector(N) dimension in the DB schema.
# We store the top-N SHAP values per modality, zero-padded to this length.
SHAP_VECTOR_DIM = 10


# ── Persist analysis results ──────────────────────────────────────────────────

def persist_fused_result(fused: FusedResult, user_id: str) -> None:
    """
    Write the complete analysis job to the DB.
    Called as a FastAPI background task after the HTTP response is sent.

    Stores:
      - fused_results      (one row per job)
      - modality_results   (one row per available modality)
      - shap_embeddings    (one row per available modality, for RAG retrieval)

    Raw patient files are NEVER stored — they live only in /tmp during inference.
    """
    db = get_client()

    # ── 1. fused_results ──────────────────────────────────────────────────────
    try:
        db.table("fused_results").insert({
            "job_id":               fused.job_id,
            "user_id":              user_id,
            "patient_id":           fused.patient_id,
            "patient_uuid":         getattr(fused, "patient_uuid", None),
            "probability":          fused.probability,
            "risk_label":           fused.risk_label,
            "ci_low":               fused.ci_low,
            "ci_high":              fused.ci_high,
            "modality_weights":     fused.modality_weights,
            "fusion_model_version": fused.fusion_model_version,
            "report_json":          fused.report_json,
        }).execute()
        logger.info("fused_results saved: job=%s patient=%s", fused.job_id, fused.patient_id)
    except Exception as e:
        logger.exception("Failed to save fused_results for job %s: %s", fused.job_id, e)
        return  # can't save child rows without the parent

    # ── 2. modality_results + shap_embeddings ─────────────────────────────────
    for mr in fused.modality_results:
        if not mr.available:
            continue

        # modality_results
        try:
            db.table("modality_results").insert({
                "job_id":      fused.job_id,
                "modality":    mr.modality,
                "probability": mr.probability,
                "shap_features": [
                    {"name": f.name, "value": f.value, "rank": f.rank}
                    for f in mr.shap_features
                ],
                "available":   True,
            }).execute()
        except Exception as e:
            logger.exception(
                "Failed to save modality_results for job %s modality %s: %s",
                fused.job_id, mr.modality, e
            )
            continue  # still try to save SHAP embedding

        # shap_embeddings — fixed-length float vector for pgvector
        try:
            shap_vals = [float(f.value) for f in mr.shap_features[:SHAP_VECTOR_DIM]]
            shap_vals += [0.0] * (SHAP_VECTOR_DIM - len(shap_vals))  # zero-pad

            db.table("shap_embeddings").insert({
                "job_id":    fused.job_id,
                "modality":  mr.modality,
                "content":   (
                    f"{mr.modality} SHAP: "
                    + ", ".join(
                        f"{f.name}={f.value:.4f}"
                        for f in mr.shap_features[:5]
                    )
                ),
                "embedding": shap_vals,
            }).execute()
        except Exception as e:
            logger.exception(
                "Failed to save shap_embeddings for job %s modality %s: %s",
                fused.job_id, mr.modality, e
            )

    logger.info("Persistence complete: job=%s", fused.job_id)


# ── Fetch fused result (with modality_results joined) ────────────────────────

def get_result(job_id: str, user_id: str) -> dict | None:
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
    row["modality_results"] = mr_res.data or []

    return row


# ── Fetch clinical report ──────────────────────────────────────────────────────

def get_report(job_id: str) -> dict | None:
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

def get_modality_embeddings_for_job(job_id: str) -> dict[str, list[float]]:
    db = get_client()
    res = (
        db.table("shap_embeddings")
        .select("modality, embedding")
        .eq("job_id", job_id)
        .execute()
    )
    if not res.data:
        return {}

    def _parse_vec(raw) -> list:
        # Supabase returns vector columns as strings '[0.1,0.2,...]' — parse to list
        if isinstance(raw, list):
            return [float(x) for x in raw]
        if isinstance(raw, str):
            import json as _json
            return [float(x) for x in _json.loads(raw)]
        return list(raw)

    return {row["modality"]: _parse_vec(row["embedding"]) for row in res.data}
