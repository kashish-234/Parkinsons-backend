from functools import lru_cache
import joblib

from services.model_storage_service import (
    model_storage_service
)


@lru_cache(maxsize=1)
def load_components():

    model_path = model_storage_service.download_model(
        "speech/oxford/speech_oxford_lgbm_v1.pkl"
    )

    cal_model_path = model_storage_service.download_model(
        "speech/oxford/speech_oxford_lgbm_v1_calibrated.pkl"
    )

    feature_map_path = model_storage_service.download_model(
        "speech/oxford/feature_map.pkl"
    )

    return (
        joblib.load(model_path),
        joblib.load(cal_model_path),
        joblib.load(feature_map_path),
    )