import shap
import pandas as pd
import numpy as np
import logging
 
from models.base.contracts import ModelOutput, SHAPFeature
from .model import load_components
from .canonicalize import canonicalize
 
logger = logging.getLogger(__name__)
 
 
class SpeechMDVRSVC:
    MODEL_ID = "speech_mdvr_svc_v1"
    VALIDATION_AUC = 0.8833
 
    def predict(self, raw_features: dict) -> ModelOutput:
        """
        raw_features: dict[str, float] — full speech feature dict.
        MDVR uses 60 acoustic features; keys are listed in feature_map.pkl.
        Keys not in the MDVR feature set are silently ignored.
        """
        model, cal_model, feature_map, shap_background = load_components()
 
        # Select MDVR's own feature columns from the unified input dict
        mdvr_cols = list(feature_map.keys())  # 60 features in training order
        row = {col: raw_features.get(col, 0.0) for col in mdvr_cols}
        sample_df = pd.DataFrame([row])[mdvr_cols]
 
        prob = float(cal_model.predict_proba(sample_df)[0, 1])
        eps = 1e-6
        p_clipped = min(max(prob, eps), 1 - eps)
        raw_logit = float(np.log(p_clipped / (1 - p_clipped)))
 
        shap_features = self._compute_shap(cal_model, sample_df, shap_background, feature_map)
        mc_samples    = self._bootstrap_mc(sample_df)
 
        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="speech",
            dataset="mdvr_kcl",
            probability=prob,
            shap_features=shap_features,
            raw_logit=raw_logit,
            mc_samples=mc_samples,
            metadata={
                "n_features_used": len(mdvr_cols),
                "model_type": "SVC_RBF_Pipeline",
            },
        )
 
    def _compute_shap(self, cal_model, sample_df, shap_background, feature_map) -> list:
        """KernelExplainer for SVC — TreeExplainer doesn't support SVC."""
        try:
            explainer = shap.KernelExplainer(
                lambda x: cal_model.predict_proba(
                    pd.DataFrame(x, columns=sample_df.columns)
                )[:, 1],
                shap_background,
                silent=True,
            )
            shap_values = explainer.shap_values(sample_df.values, nsamples=100, silent=True)
            shap_vals   = np.asarray(shap_values).ravel()
        except Exception as e:
            logger.warning(f"SpeechMDVRSVC SHAP failed: {e}. Returning empty shap_features.")
            return []
 
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
        return [
            SHAPFeature(
                name=canonicalize(feature_map.get(sample_df.columns[idx], sample_df.columns[idx])),
                value=float(shap_vals[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(top_idx, start=1)
        ]
 
    def _bootstrap_mc(self, sample_df: pd.DataFrame) -> list:
        """50 bootstrap SVC models for uncertainty estimation."""
        try:
            from services.model_storage_service import model_storage_service
            import joblib
            local_path = model_storage_service.download_model("speech/mdvr/bootstrap_models.pkl")
            bootstrap_models = joblib.load(local_path)
            return [float(m.predict_proba(sample_df)[0, 1]) for m in bootstrap_models]
        except Exception:
            logger.warning("speech/mdvr/bootstrap_models.pkl not found — using synthetic MC.")
            _, cal_model, _, _ = load_components()
            prob = float(cal_model.predict_proba(sample_df)[0, 1])
            rng  = np.random.default_rng(42)
            return [float(np.clip(prob + n, 0.0, 1.0)) for n in rng.normal(0, 0.08, 50)]
 