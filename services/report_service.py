import json
import logging
import google.generativeai as genai
from services.rag_service import RetrievalContext
from services.supabase_service import get_client
from core.config import settings

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


def _build_prompt(fused_result: dict, modality_results: list[dict],
                  retrieval_context: RetrievalContext) -> str:

    modality_lines = []
    for mr in modality_results:
        if not mr.get("modality"):
            continue

        feats = mr.get("shap_features", [])[:3]
        feat_text = ", ".join(
            f"{f['name']} (SHAP: {f['value']:.3f})" for f in feats
        )

        modality_lines.append(
            f"- {mr['modality']}: "
            f"P(PD)={mr['probability']:.3f}, "
            f"CI [{mr.get('ci_low',0):.3f}, {mr.get('ci_high',0):.3f}], "
            f"features: {feat_text}"
        )

    if retrieval_context.similar_cases:
        case_lines = []
        for i, c in enumerate(retrieval_context.similar_cases[:5], 1):
            case_lines.append(
                f"Case {i}: {c.modality}, dist={c.distance:.3f}, "
                f"features: {c.content}"
            )
        similar_text = "\n".join(case_lines)
    else:
        similar_text = "No similar cases found."

    return f"""
You are a clinical AI assistant for Parkinson's disease.

STRICT RULES:
- No hallucinations
- No diagnosis, only risk
- Use exact numbers

Patient:
Probability: {fused_result['probability']:.4f}
Risk: {fused_result['risk_label']}
CI: [{fused_result['ci_low']:.4f}, {fused_result['ci_high']:.4f}]

Modalities:
{chr(10).join(modality_lines)}

Similar cases:
{similar_text}

Return ONLY JSON:

{{
  "summary": "",
  "risk_interpretation": "",
  "biomarker_findings": "",
  "uncertainty_analysis": "",
  "similar_cases_context": "",
  "recommendations": "",
  "caveats": ""
}}
"""

def generate_report(job_id: str,
                    fused_result: dict,
                    modality_results: list[dict],
                    retrieval_context: RetrievalContext) -> dict:

    db = get_client()

    prompt = _build_prompt(fused_result, modality_results, retrieval_context)

    logger.info(f"Calling Gemini for report {job_id}")

    response = model.generate_content(prompt)

    raw = response.text

    # Safe JSON extraction
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        sections = json.loads(raw[start:end])
    except Exception:
        sections = {
            "summary": raw,
            "risk_interpretation": "",
            "biomarker_findings": "",
            "uncertainty_analysis": "",
            "similar_cases_context": "",
            "recommendations": "",
            "caveats": "Parsing failed"
        }

    # Store report
    db.table("clinical_reports").insert({
        "job_id": job_id,
        "report_sections": sections,
        "prompt_used": prompt,
        "llm_model": "gemini-1.5-flash",
    }).execute()

    logger.info(f"Report stored for job {job_id}")

    return sections