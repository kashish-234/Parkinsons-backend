import numpy as np
import shap
from models.base.contracts import (ModelOutput,SHAPFeature)
from .model import load_components
from .canonicalize import canonicalize

class SpeechUCIRF:

    MODEL_ID = "speech_uci_rf_v1"
    VALIDATION_AUC = 0.924

    def predict(self,raw_features: dict) -> ModelOutput:
        components = load_components()
        model = components["model"]
        cal_model = components["cal_model"]
        selector = components["selector"]

        feature_cols_full = (components["feature_cols_full"])
        selected_feature_names = (components["selected_feature_names"])
        bootstrap_models = (components["bootstrap_models"])
        validation_auc = (components["validation_auc"])
        decision_threshold = (components["decision_threshold"])

        row = np.array([[raw_features.get(col, 0.0) for col in feature_cols_full]], dtype=np.float64)

        row_sel = selector.transform(row)
        prob = float(cal_model.predict_proba(row_sel)[0, 1])
        predicted_label = int(prob >= decision_threshold)

        eps = 1e-6

        p_clipped = min(max(prob, eps),1 - eps,)

        raw_logit = float(
            np.log(
                p_clipped /
                (1 - p_clipped)
            )
        )

        explainer = shap.TreeExplainer(model)

        shap_values = (explainer.shap_values(row_sel))

        if isinstance(shap_values,list):
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

        for rank, idx in enumerate(top_idx,start=1,):
            raw_name = (selected_feature_names[idx])
            shap_features.append(
                SHAPFeature(name=canonicalize(raw_name),
                    value=float(shap_vals[idx]),rank=rank))

        mc_samples = [
            float(
                bootstrap_model.predict_proba(row_sel)[0, 1]
            )
            for bootstrap_model in bootstrap_models
        ]

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="speech",
            dataset="uci_pd_speech",
            probability=prob,
            shap_features=shap_features,
            raw_logit=raw_logit,
            mc_samples=mc_samples,
            metadata={
                "n_features_used":row_sel.shape[1],
                "predicted_label":predicted_label,
                "decision_threshold":float(decision_threshold),
                "validation_auc": float(validation_auc),
            },
        )