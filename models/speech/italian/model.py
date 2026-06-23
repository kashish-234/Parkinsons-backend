from functools import lru_cache
import joblib
from services.model_storage_service import model_storage_service


@lru_cache(maxsize=1)
def load_components() -> dict:
    """
    Downloads and caches all Italian-speech artifacts from HuggingFace.
    Returns dict with keys:
        model, cal_model, imputer, selector, feature_cols_full,
        selected_feature_names, bootstrap_models,
        validation_auc, decision_threshold
    """
    files = {
        "model":                  "speech/italian/speech_italian_rf_v1.pkl",
        "cal_model":              "speech/italian/speech_italian_rf_v1_calibrated.pkl",
        "imputer":                "speech/italian/imputer.pkl",
        "selector":               "speech/italian/feature_selector.pkl",
        "feature_cols_full":      "speech/italian/feature_cols_full.pkl",
        "selected_feature_names": "speech/italian/selected_feature_names.pkl",
        "bootstrap_models":       "speech/italian/bootstrap_models.pkl",
        "validation_auc":         "speech/italian/validation_auc.pkl",
        "decision_threshold":     "speech/italian/decision_threshold.pkl",
    }
    components = {}
    for key, hf_path in files.items():
        local_path = model_storage_service.download_model(hf_path)
        components[key] = joblib.load(local_path)
    return components