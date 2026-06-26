from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from datetime import datetime


# ── SHAP ─────────────────────────────────────────────────────────────────────

class ShapFeature(BaseModel):
    name: str
    value: float
    rank: Optional[int] = None


class ModalityResult(BaseModel):
    modality: str
    probability: float
    shap_features: List[ShapFeature]
    available: bool


# ── RESPONSES ────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    job_id: str
    patient_id: str
    probability: float
    risk_label: str
    ci_low: float
    ci_high: float
    modality_weights: Dict[str, float]
    available_modalities: List[str]
    fusion_model_version: Optional[str] = None
    warning: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    probability: float
    risk_label: str
    ci_low: float
    ci_high: float
    modality_weights: Dict[str, float]
    patient_id: Optional[str] = None
    modality_results: Optional[List[ModalityResult]] = None
    report_json: Optional[dict] = None


class ReportResponse(BaseModel):
    job_id: str
    report_sections: Dict[str, str]
    n_similar_cases_used: int
    llm_model: str
    generated_at: Optional[datetime] = None
