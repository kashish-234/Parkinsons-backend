import numpy as np
import shap
from models.base.contracts import ModelOutput, SHAPFeature
from .model import load_components
from .canonicalize import canonicalize


class FingerTappingLGBM:

    MODEL_ID = "finger_tapping_lgbm_v1"

    def __init__(self):
        components = load_components()
        self._cal_model             = components["cal_model"]
        self._scaler                = components["scaler"]
        self._selected_feature_names = components["selected_feature_names"]
        self._metadata              = components["metadata"]
        self._hand_encoder          = components["hand_encoder"]
        self._gender_encoder        = components["gender_encoder"]
        self._explainer             = shap.TreeExplainer(self._cal_model.estimator)
        # FIX H1: read VALIDATION_AUC from metadata, don't hard-code 0.0
        self.VALIDATION_AUC = float(self._metadata.get("validation_auc", 0.75))

    def predict(self, raw_features: dict) -> ModelOutput:
        validation_auc    = self.VALIDATION_AUC
        decision_threshold = 0.5

        features = raw_features.copy()

        # Encode hand (L/R) and gender (F/M); handle already-encoded ints gracefully
        hand_val = features.get("hand", "R")
        if isinstance(hand_val, str):
            features["hand"] = self._hand_encoder.transform([hand_val])[0]

        gender_val = features.get("gender", "M")
        if isinstance(gender_val, str):
            features["gender"] = self._gender_encoder.transform([gender_val])[0]

        row = np.array(
            [[features.get(col, 0.0) for col in self._selected_feature_names]],
            dtype=np.float64,
        )
        row = self._scaler.transform(row)

        prob = float(self._cal_model.predict_proba(row)[0, 1])
        predicted_label = int(prob >= decision_threshold)

        eps = 1e-6
        p_clipped = min(max(prob, eps), 1 - eps)
        raw_logit = float(np.log(p_clipped / (1 - p_clipped)))

        shap_values = self._explainer.shap_values(row)
        if isinstance(shap_values, list):
            shap_vals = np.asarray(shap_values[1][0])
        else:
            arr = np.asarray(shap_values)
            shap_vals = arr[0, :, 1] if arr.ndim == 3 else arr[0]
        shap_vals = shap_vals.ravel()

        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
        shap_features = [
            SHAPFeature(
                name=canonicalize(self._selected_feature_names[idx]),
                value=float(shap_vals[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(top_idx, start=1)
        ]

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="finger_tapping",
            dataset="finger_tapping_severity",
            probability=prob,
            shap_features=shap_features,
            raw_logit=raw_logit,
            mc_samples=[],
            metadata={
                "n_features_used":    row.shape[1],
                "predicted_label":    predicted_label,
                "decision_threshold": float(decision_threshold),
                "validation_auc":     float(validation_auc),
            },
        )