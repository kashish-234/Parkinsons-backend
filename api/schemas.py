from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime

# -------- REQUESTS --------

class PredictRequest(BaseModel):
    patient_id: str
    job_id: str
    files: Dict[str, List[str]]

class ReportRequest(BaseModel):
    job_id: str

# -------- SHAP --------

class ShapFeature(BaseModel):
    name: str
    value: float
    rank: Optional[int] = None


class ModalityResult(BaseModel):
    modality: str
    probability: float
    shap_features: List[ShapFeature]
    available: bool


# -------- RESPONSES --------

class PredictResponse(BaseModel):
    job_id: str
    patient_id: str
    probability: float
    risk_label: str
    ci_low: float
    ci_high: float
    modality_weights: Dict[str, float]
    available_modalities: List[str]
    warning: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    patient_id: str
    probability: float
    risk_label: str
    ci_low: float
    ci_high: float
    modality_weights: Dict[str, float]
    modality_results: Optional[List[ModalityResult]] = None
    report_json: Optional[dict] = None


class ReportSectionsSchema(BaseModel):
    summary: str
    risk_interpretation: str
    biomarker_findings: str
    uncertainty_analysis: str
    similar_cases_context: str
    recommendations: str
    caveats: str


class ReportResponse(BaseModel):
    job_id: str
    report_sections: ReportSectionsSchema
    n_similar_cases_used: int
    llm_model: str
    generated_at: Optional[datetime] = None