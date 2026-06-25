import numpy as np
import torch
import shap
from models.base.contracts import (ModelOutput, SHAPFeature)
from .model import load_components
from .canonicalize import canonicalize


class GaitDaphnetCNNLSTM:

    MODEL_ID = "gait_cnn_lstm_v1"

    def predict(self, raw_features: dict) -> ModelOutput:
        components = load_components()
        model = components["model"]
        temperature_scaler = components["temperature_scaler"]
        scaler = components["scaler"]
        metadata = components["metadata"]
        background = components["background"]

        validation_auc = float(metadata["validation_auc"])
        window_size = int(metadata["window_size"])
        decision_threshold = float(metadata["decision_threshold"])

        # FIX 1: Read temperature from the loaded TemperatureScaler parameter,
        # not from metadata.json (which never writes a "temperature" key).
        temperature = float(temperature_scaler.temperature.item())

        window = np.asarray(
            raw_features["window"],
            dtype=np.float64
        )

        # Scale: fit was done per-sensor on (N*WINDOW, 9)
        window_flat = window.reshape(-1, window.shape[-1])
        window_flat = scaler.transform(window_flat)
        window_scaled = window_flat.reshape(window.shape)

        # CNN expects (N, channels, time) → (1, 9, WINDOW)
        x = torch.tensor(
            window_scaled.T[np.newaxis],  # (1, 9, WINDOW)
            dtype=torch.float32
        )

        model.eval()
        temperature_scaler.eval()

        with torch.no_grad():
            logits = model(x)                        # (1, 2)
            # FIX 2: Use the temperature_scaler module for scaling (single
            # source of truth), rather than manually dividing by temperature.
            scaled_logits = temperature_scaler(logits)
            probs = torch.softmax(scaled_logits, dim=1)

        prob = float(probs[0, 1].item())
        predicted_label = int(prob >= decision_threshold)

        eps = 1e-6
        p_clipped = min(max(prob, eps), 1 - eps)
        raw_logit = float(np.log(p_clipped / (1 - p_clipped)))

        # SHAP — GradientExplainer requires CPU tensors.
        # FIX 3: background is loaded from model components (saved during
        # training as real data batches), falling back to zeros only if
        # the artifact is absent. This matches the notebook's intent of using
        # actual training samples as the SHAP baseline.
        explainer = shap.GradientExplainer(model, background)

        # shap_values: list of 2 arrays (one per class), each (1, 9, WINDOW)
        shap_values = explainer.shap_values(x)

        # Take class-1 values, shape (9, WINDOW)
        shap_class1 = np.asarray(shap_values[1][0])  # (9, WINDOW)

        # Flatten to (9*WINDOW,) matching (sensor, timestep) order
        shap_vals = shap_class1.ravel()

        top_idx = np.argsort(np.abs(shap_vals))[::-1][:10]

        shap_features = []

        for rank, idx in enumerate(top_idx, start=1):
            sensor = int(idx) // window_size          # channel axis is first
            timestep = int(idx) % window_size
            raw_name = f"t{timestep}_s{sensor}"
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
            modality="gait",
            dataset="daphnet_fog",
            probability=prob,
            shap_features=shap_features,
            raw_logit=raw_logit,
            mc_samples=mc_samples,
            metadata={
                "n_features_used": int(x.shape[1] * x.shape[2]),
                "predicted_label": predicted_label,
                "decision_threshold": decision_threshold,
                "validation_auc": validation_auc,
                "temperature": temperature,
            },
        )