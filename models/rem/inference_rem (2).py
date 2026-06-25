import numpy as np
import pandas as pd
import shap
from models.base.contracts import (ModelOutput, SHAPFeature)
from .model_rem import load_components
from .canonicalize_rem import canonicalize

class REMEnsembleModel:

    MODEL_ID = "rem_ensemble_v1"
    VALIDATION_AUC = 0.0

    def predict(self, raw_features: dict) -> ModelOutput:
        components = load_components()
        ensemble = components["ensemble"]
        metadata = components["metadata"]

        ensemble_accuracy = float(metadata.get("ensemble_accuracy", 0.0))
        ensemble_f1 = float(metadata.get("ensemble_f1", 0.0))
        decision_threshold = 0.5

        feature_names = ensemble.feature_names

        row = pd.DataFrame(
            [[raw_features[col] for col in feature_names]],
            columns=feature_names
        )

        proba = ensemble.predict_proba(row)        # (1, n_classes)
        prob = float(proba[0, 1])
        predicted_label = int(ensemble.predict(row)[0])

        eps = 1e-6

        p_clipped = min(max(prob, eps), 1 - eps)

        raw_logit = float(
            np.log(
                p_clipped /
                (1 - p_clipped)
            )
        )

        # ── SHAP: run on each sub-model and average ───────────────────
        shap_accumulator = None
        n_models = 0

        for model_name, sub_model in ensemble.models.items():
            X_scaled = sub_model.scaler.transform(row)
            explainer = shap.TreeExplainer(sub_model.model)
            shap_values = explainer.shap_values(X_scaled)

            if isinstance(shap_values, list):
                sv = np.asarray(shap_values[1][0])
            else:
                arr = np.asarray(shap_values)
                sv = (
                    arr[0, :, 1]
                    if arr.ndim == 3
                    else arr[0]
                )

            sv = sv.ravel()

            if shap_accumulator is None:
                shap_accumulator = sv
            else:
                shap_accumulator = shap_accumulator + sv

            n_models += 1

        shap_vals = shap_accumulator / n_models

        top_idx = np.argsort(
            np.abs(shap_vals)
        )[::-1][:10]

        shap_features = []

        for rank, idx in enumerate(top_idx, start=1):
            raw_name = feature_names[idx]
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
            modality="rem",
            dataset="rem_sleep_disorder",
            probability=prob,
            shap_features=shap_features,
            raw_logit=raw_logit,
            mc_samples=mc_samples,
            metadata={
                "n_features_used": len(feature_names),
                "predicted_label": predicted_label,
                "decision_threshold": float(decision_threshold),
                "ensemble_accuracy": ensemble_accuracy,
                "ensemble_f1": ensemble_f1,
                "fusion_method": ensemble.fusion_method,
                "n_models": len(ensemble.models),
            },
        )
