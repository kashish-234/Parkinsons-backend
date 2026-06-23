import numpy as np
import shap
from models.base.contracts import ModelOutput, SHAPFeature
from .model import load_components
from .canonicalize import canonicalize
 
 
class SpeechItalianLGBM:
 
    MODEL_ID = "speech_italian_lgbm"
    VALIDATION_AUC = 0.9167
 
    def predict(self, raw_features: dict) -> ModelOutput:
        c = load_components()
        model              = c["model"]
        cal_model          = c["cal_model"]
        imputer            = c["imputer"]
        selector           = c["selector"]
        feature_cols_full  = c["feature_cols_full"]
        selected_names     = c["selected_feature_names"]
        bootstrap_models   = c["bootstrap_models"]
        validation_auc     = c["validation_auc"]
        decision_threshold = c["decision_threshold"]
 
        # Build input row in training column order; missing keys → NaN
        row_raw = np.array(
            [[raw_features.get(col, np.nan) for col in feature_cols_full]],
            dtype=np.float64,
        )
 
        # Apply imputer then selector in the same order as train.py
        row_imp = imputer.transform(row_raw)
        row_sel = selector.transform(row_imp)
 
        prob            = float(cal_model.predict_proba(row_sel)[0, 1])
        predicted_label = int(prob >= decision_threshold)
 
        eps = 1e-6
        p_c = min(max(prob, eps), 1 - eps)
        raw_logit = float(np.log(p_c / (1 - p_c)))
 
        # SHAP — dimensionality-safe for all shap versions
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row_sel)
        if isinstance(shap_values, list):
            shap_vals = np.asarray(shap_values[1][0])
        else:
            arr = np.asarray(shap_values)
            shap_vals = arr[0, :, 1] if arr.ndim == 3 else arr[0]
        shap_vals = shap_vals.ravel()
 
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
        shap_features = [
            SHAPFeature(
                name=canonicalize(selected_names[idx]),
                value=float(shap_vals[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(top_idx, start=1)
        ]
 
        # Real MC uncertainty from 50 subject-bootstrap models
        mc_samples = [
            float(m.predict_proba(row_sel)[0, 1])
            for m in bootstrap_models
        ]
 
        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="speech",
            dataset="italian_pvs",
            probability=prob,
            shap_features=shap_features,
            raw_logit=raw_logit,
            mc_samples=mc_samples,
            metadata={
                "n_features_used":    row_sel.shape[1],
                "predicted_label":    predicted_label,
                "decision_threshold": float(decision_threshold),
                "validation_auc":     float(validation_auc),
            },
        )
 