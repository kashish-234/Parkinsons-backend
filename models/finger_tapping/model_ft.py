from functools import lru_cache
import json
import joblib

from services.model_storage_service import (
    model_storage_service
)

@lru_cache(maxsize=1)
def load_components():

    pkl_files = {
        "cal_model":
            "finger_tapping/finger_tapping_model.pkl",

        "scaler":
            "finger_tapping/scaler.pkl",

        "selected_feature_names":
            "finger_tapping/selected_feature_names.pkl",
    }

    components = {}

    for key, hf_path in pkl_files.items():

        local_path = (
            model_storage_service
            .download_model(hf_path)
        )

        components[key] = joblib.load(
            local_path
        )

    metadata_path = (
        model_storage_service
        .download_model(
            "finger_tapping/metadata.json"
        )
    )

    with open(metadata_path, "r") as f:
        components["metadata"] = json.load(f)

    return components
