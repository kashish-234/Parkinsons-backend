import numpy as np
import shap
from models.base.contracts import (ModelOutput, SHAPFeature)
from .model_ft import load_components
from .canonicalize import canonicalize

class FingerTappingLGBM:

    MODEL_ID = "finger_tapping_lgbm_v1"
    VALIDATION_AUC = 0.0

    def predict(self, raw_features: dict) -> ModelOutput:
        components = load_components()
        cal_model = components["cal_model"]
        scaler = components["scaler"]
        selected_feature_names = components["selected_feature_names"]
        metadata = components["metadata"]

        validation_auc = float(metadata["validation_auc"])
        decision_threshold = 0.5

        row = np.array(
            [[raw_features[col] for col in selected_feature_names]],
            dtype=np.float64
        )

        row = scaler.transform(row)

        prob = float(cal_model.predict_proba(row)[0, 1])
        predicted_label = int(prob >= decision_threshold)

        eps = 1e-6

        p_clipped = min(max(prob, eps), 1 - eps)

        raw_logit = float(
            np.log(
                p_clipped /
                (1 - p_clipped)
            )
        )

        explainer = shap.TreeExplainer(cal_model.estimator)

        shap_values = (explainer.shap_values(row))

        if isinstance(shap_values, list):
            shap_vals = np.asarray(shap_values[1][0])
        else:
            arr = np.asarray(shap_values)
            shap_vals = (
                arr[0, :, 1]
                if arr.ndim == 3
                else arr[0]
            )

        shap_vals = (shap_vals.ravel())

        top_idx = np.argsort(
            np.abs(shap_vals)
        )[::-1][:10]

        shap_features = []

        for rank, idx in enumerate(top_idx, start=1):
            raw_name = selected_feature_names[idx]
            shap_features.append(
                SHAPFeature(
                    name=canonicalize(raw_name),
                    value=float(shap_vals[idx]),
                    rank=rank,
                )
            )

        mc_samples = []

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="finger_tapping",
            dataset="finger_tapping_severity",
            probability=prob,
            shap_features=shap_features,
            raw_logit=raw_logit,
            mc_samples=mc_samples,
            metadata={
                "n_features_used": row.shape[1],
                "predicted_label": predicted_label,
                "decision_threshold": float(decision_threshold),
                "validation_auc": float(validation_auc),
            },
        )
