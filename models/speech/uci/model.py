from functools import lru_cache
import joblib

from services.model_storage_service import (
    model_storage_service
)

@lru_cache(maxsize=1)
def load_components():

    files = {
        "model":
            "speech/uci/speech_uci_rf_v1.pkl",

        "cal_model":
            "speech/uci/speech_uci_rf_v1_calibrated.pkl",

        "selector":
            "speech/uci/feature_selector.pkl",

        "feature_cols_full":
            "speech/uci/feature_cols_full.pkl",

        "selected_feature_names":
            "speech/uci/selected_feature_names.pkl",

        "bootstrap_models":
            "speech/uci/bootstrap_models.pkl",

        "validation_auc":
            "speech/uci/validation_auc.pkl",

        "decision_threshold":
            "speech/uci/decision_threshold.pkl",
    }

    components = {}

    for key, hf_path in files.items():

        local_path = (
            model_storage_service
            .download_model(hf_path)
        )

        components[key] = joblib.load(
            local_path
        )

    return components