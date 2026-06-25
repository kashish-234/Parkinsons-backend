from functools import lru_cache
import json
import joblib
from sklearn.preprocessing import LabelEncoder

from services.model_storage_service import model_storage_service


@lru_cache(maxsize=1)
def load_components():
    pkl_files = {
        "cal_model":              "finger_tapping/finger_tapping_model.pkl",
        "scaler":                 "finger_tapping/scaler.pkl",
        "selected_feature_names": "finger_tapping/selected_feature_names.pkl",
    }

    try:
        components = {}
        for key, hf_path in pkl_files.items():
            local_path = model_storage_service.download_model(hf_path)
            components[key] = joblib.load(local_path)

        metadata_path = model_storage_service.download_model(
            "finger_tapping/metadata.json"
        )
        with open(metadata_path, "r") as f:
            components["metadata"] = json.load(f)

        # Try to load hand encoder — fall back to known categories
        try:
            hand_path = model_storage_service.download_model(
                "finger_tapping/hand_encoder.pkl"
            )
            components["hand_encoder"] = joblib.load(hand_path)
        except Exception:
            enc = LabelEncoder()
            enc.fit(["L", "R"])
            components["hand_encoder"] = enc

        # Try to load gender encoder — fall back to known categories
        try:
            gender_path = model_storage_service.download_model(
                "finger_tapping/gender_encoder.pkl"
            )
            components["gender_encoder"] = joblib.load(gender_path)
        except Exception:
            enc = LabelEncoder()
            enc.fit(["F", "M"])
            components["gender_encoder"] = enc

        return components

    except Exception:
        load_components.cache_clear()
        raise