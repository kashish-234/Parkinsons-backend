from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SHAPFeature:
    name: str          # canonical feature name e.g. "jitter", "shimmer", "hnr"
    value: float       # SHAP value: positive = pushes toward PD
    rank: int          # 1 = most important


@dataclass
class ModelOutput:
    """
    What ONE sub-model returns. Every sub-model in the project returns this.
    """
    model_id: str              
    modality: str              
    dataset: str               
    probability: float         
    shap_features: list        
    raw_logit: float           
    mc_samples: list           
    metadata: dict = field(default_factory=dict)
    # speech metadata carries:
    # {"feature_names": [...], "feature_values": [...], "n_features": int}


@dataclass
class ModalityResult:
    """
    What IntraModalFuser returns after combining all speech sub-models.
    This is what LateFusionModel receives.
    """
    modality: str              # "speech"
    available: bool            # False if patient did not submit audio
    probability: float         # intra-fused P(PD)
    ci_low: float              # 2.5th percentile of pooled MC samples
    ci_high: float             # 97.5th percentile of pooled MC samples
    ci_width: float            # ci_high - ci_low (reliability signal)
    shap_features: list        # list[SHAPFeature], merged top 10
    model_ids: list            # list[str], which sub-models contributed
    metadata: dict = field(default_factory=dict)


@dataclass
class FusedResult:
    job_id: str
    patient_id: str
    patient_uuid: Optional[str] = None
    probability: float
    risk_label: str
    ci_low: float
    ci_high: float
    modality_weights: dict[str, float]
    fusion_model_version: str | None = None
    report_json: dict | None = None
    modality_results: list[ModalityResult]