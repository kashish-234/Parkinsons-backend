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
            "gait/daphnet/gait_model.pkl",

        "scaler":
            "gait/daphnet/scaler.pkl",
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
            "gait/daphnet/metadata.json"
        )
    )

    with open(metadata_path, "r") as f:
        components["metadata"] = json.load(f)

    return components
