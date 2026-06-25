from functools import lru_cache
import joblib

from services.model_storage_service import model_storage_service


@lru_cache(maxsize=1)
def load_components():
    """
    Returns (pipeline, cal_model, feature_map, shap_background).

    pipeline:         sklearn Pipeline (SimpleImputer → StandardScaler → SVC)
    cal_model:        CalibratedClassifierCV wrapping the frozen pipeline
    feature_map:      dict {feature_name: feature_name} -- identity map
                      (kept for API consistency with Oxford / Sakar models)
    shap_background:  np.ndarray, shape (n_background, n_features) --
                      representative training rows used by KernelExplainer
    """
    files = {
        "model":            "speech/mdvr/speech_mdvr_best.pkl",
        "cal_model":        "speech/mdvr/speech_mdvr_best_calibrated.pkl",
        "feature_map":      "speech/mdvr/feature_map.pkl",
        "shap_background":  "speech/mdvr/shap_background.pkl",
    }

    components = {}
    for key, hf_path in files.items():
        local_path = model_storage_service.download_model(hf_path)
        components[key] = joblib.load(local_path)

    return (
        components["model"],
        components["cal_model"],
        components["feature_map"],
        components["shap_background"],
    )