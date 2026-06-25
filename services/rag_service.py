from dataclasses import dataclass
import logging
from services.supabase_service import get_client

logger = logging.getLogger(__name__)


@dataclass
class SimilarCase:
    job_id: str
    modality: str
    distance: float
    content: str


@dataclass
class RetrievalContext:
    current_job_id: str
    similar_cases: list[SimilarCase]
    n_retrieved: int
    matched_modalities: list[str]


def retrieve_similar_cases(
    job_id: str,
    modality_embeddings: dict[str, list[float]],
    k_per_modality: int = 5,
) -> RetrievalContext:

    db = get_client()
    all_cases = []
    matched_modalities = []

    for modality, embedding in modality_embeddings.items():
        try:
            response = db.rpc("match_shap_embeddings", {
                "query_embedding": embedding,
                "match_count": k_per_modality,
            }).execute()

            if not response.data:
                continue

            for row in response.data:
                all_cases.append(SimilarCase(
                    job_id="unknown",
                    modality=modality,
                    distance=1 - row["similarity"],
                    content=row["content"],
                ))

            matched_modalities.append(modality)

        except Exception as e:
            logger.warning(f"RAG failed for {modality}: {e}")

    # Deduplicate
    seen = {}
    for case in all_cases:
        key = case.content
        if key not in seen or case.distance < seen[key].distance:
            seen[key] = case

    deduplicated = sorted(seen.values(), key=lambda c: c.distance)

    return RetrievalContext(
        current_job_id=job_id,
        similar_cases=deduplicated,
        n_retrieved=len(deduplicated),
        matched_modalities=matched_modalities,
    )