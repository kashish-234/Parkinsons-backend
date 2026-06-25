import shap
import pandas as pd
import numpy as np
 
from models.base.contracts import ModelOutput, SHAPFeature
from .model import load_components
from .canonicalize import canonicalize
 
 
class SpeechOxfordLGBM:
    MODEL_ID = "speech_oxford_lgbm_v1"
    VALIDATION_AUC = 0.9286
 
    def predict(self, raw_features: dict) -> ModelOutput:
        """
        raw_features: dict[str, float] — full speech feature dict.
        Oxford uses 22 features with cleaned column names (e.g. MDVP_Fo_Hz_,
        MDVP_Jitter_, HNR ...) as stored in the feature_map artifact.
        Keys not in the Oxford feature set are silently ignored.
        """
        model, cal_model, feature_map = load_components()
 
        # Build a single-row DataFrame in the exact column order the model
        # was trained on, selecting only Oxford's features from the full dict.
        oxford_cols = list(feature_map.keys())  # cleaned column names in training order
        row = {col: raw_features.get(col, 0.0) for col in oxford_cols}
        sample_df = pd.DataFrame([row])[oxford_cols]
 
        prob = float(cal_model.predict_proba(sample_df)[0, 1])
        raw_logit = float(model.predict(sample_df, raw_score=True)[0])
 
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_df)
 
        # Dimensionality-safe SHAP — handles both old list API and new 3D array API
        if isinstance(shap_values, list):
            shap_vals = np.asarray(shap_values[1][0])
        else:
            arr = np.asarray(shap_values)
            shap_vals = arr[0, :, 1] if arr.ndim == 3 else arr[0]
        shap_vals = shap_vals.ravel()
 
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
        shap_features = [
            SHAPFeature(
                name=canonicalize(sample_df.columns[idx]),
                value=float(shap_vals[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(top_idx, start=1)
        ]
 
        mc_samples = self._bootstrap_mc(sample_df, model, n=100)
 
        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="speech",
            dataset="oxford_pd",
            probability=prob,
            shap_features=shap_features,
            raw_logit=raw_logit,
            mc_samples=mc_samples,
            metadata={"n_features_used": len(oxford_cols)},
        )
 
    def _bootstrap_mc(self, sample_df: pd.DataFrame, model, n: int = 100) -> list:
        """Uncertainty via early-stopping ensemble across boosting rounds."""
        preds = []
        n_trees = model.booster_.num_trees()
        rng = np.random.default_rng(42)
        for _ in range(n):
            k = int(rng.integers(int(n_trees * 0.7), n_trees + 1))
            p = model.predict_proba(sample_df, num_iteration=k)[0, 1]
            preds.append(float(p))
        return preds