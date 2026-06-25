"""
handwriting/model.py
====================

Handwriting analysis model using DenseNet201 architecture.

TensorFlow is imported lazily (inside functions) so that the module can be
imported at application start without immediately allocating ~300 MB of RAM
for the TF runtime.  The actual TF import happens only when
get_handwriting_model() or build_handwriting_model() is first called,
which occurs on the first prediction request (or on startup if you
re-enable the warm-up block in main.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from models.base.contracts import ModelOutput, ModalityResult, SHAPFeature
from services.model_storage_service import model_storage_service

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HANDWRITING_ARTIFACT_CANDIDATES = (
    PROJECT_ROOT / "handwriting_model.keras",
    PROJECT_ROOT / "handwriting_model.h5",
    PROJECT_ROOT / "handwriting_model",
    PROJECT_ROOT / "handwriting" / "handwriting_model.keras",
    PROJECT_ROOT / "handwriting" / "handwriting_model.h5",
    PROJECT_ROOT / "handwriting" / "handwriting_model",
)


@dataclass(frozen=True)
class DenseNet201BinarySpec:
    modality: str
    model_id: str
    artifact_paths: tuple[Path, ...] = HANDWRITING_ARTIFACT_CANDIDATES
    input_size: int = 224
    class_names: dict[int, str] = field(
        default_factory=lambda: {0: "Healthy", 1: "Parkinson"}
    )


def _build_architecture(input_size: int = 224):
    """Build DenseNet201 fine-tuned architecture. TF imported here, not at module top."""
    import tensorflow as tf
    from tensorflow.keras.applications import DenseNet201
    from tensorflow.keras.layers import (
        BatchNormalization,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
    )
    from tensorflow.keras.models import Model

    base_model = DenseNet201(
        weights="imagenet", include_top=False, input_shape=(input_size, input_size, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs=base_model.input, outputs=outputs)


def _candidate_paths(paths: Iterable[Path]) -> list[Path]:
    """Deduplicate and return candidate paths."""
    resolved: list[Path] = []
    for path in paths:
        if path not in resolved:
            resolved.append(path)
    return resolved


class DenseNet201BinaryModel:
    """Handwriting classification model using DenseNet201."""

    spec: DenseNet201BinarySpec

    def __init__(self, spec: DenseNet201BinarySpec):
        self.spec = spec
        self.MODEL_ID = spec.model_id
        self._model = None  # loaded lazily on first predict()

    def _load_model(self):
        """Load TF model — called once on first predict(); lru_cache keeps it alive."""
        import tensorflow as tf

        # Try local candidate paths first (dev / Docker volume mounts)
        for path in _candidate_paths(self.spec.artifact_paths):
            if path.exists():
                try:
                    self._model = tf.keras.models.load_model(str(path))
                    return
                except Exception:
                    continue

        # FIX C9: HF repo stores at "Handwriting/model.keras" (capital H).
        artifact_path = model_storage_service.download_model(
            "Handwriting/model.keras"
        )
        try:
            self._model = tf.keras.models.load_model(artifact_path)
            return
        except Exception as e:
            pass

        # Last resort: build architecture and load weights
        try:
            self._model = _build_architecture(self.spec.input_size)
            self._model.load_weights(artifact_path)
            return
        except Exception as e:
            raise RuntimeError(
                f"Unable to load handwriting model artifact: {artifact_path}"
            ) from e

    @property
    def VALIDATION_AUC(self) -> float:
        return 0.85

    def _preprocess(self, input_data) -> Any:
        """Normalise input to a (1, H, W, 3) float32 tensor."""
        import tensorflow as tf

        input_size = self.spec.input_size

        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if path.is_dir():
                candidates = list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg"))
                if not candidates:
                    raise FileNotFoundError(
                        f"No image files found in handwriting directory: {path}"
                    )
                path = candidates[0]
            if not path.exists():
                raise FileNotFoundError(f"Handwriting image not found: {path}")
            img = Image.open(path).convert("RGB").resize((input_size, input_size))
            array = np.array(img, dtype=np.float32) / 255.0
            return tf.expand_dims(array, 0)

        array = np.asarray(input_data, dtype=np.float32)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] not in (1, 3):
            raise ValueError(
                f"Handwriting image must have 1 or 3 channels, got shape {array.shape}"
            )
        if array.shape[-1] == 1:
            array = np.concatenate([array] * 3, axis=-1)
        import tensorflow as tf
        resized = tf.image.resize(array, [input_size, input_size]).numpy()
        if resized.max() > 1.0:
            resized = resized / 255.0
        return tf.expand_dims(resized.astype(np.float32), 0)

    def predict(self, input_data) -> ModelOutput:
        """Run inference on handwriting image."""
        if self._model is None:
            self._load_model()

        tensor = self._preprocess(input_data)
        prob = float(self._model.predict(tensor, verbose=0)[0][0])

        eps = 1e-6
        p_c = min(max(prob, eps), 1 - eps)
        raw_logit = float(np.log(p_c / (1 - p_c)))

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="handwriting",
            dataset="handwriting",
            probability=prob,
            shap_features=[],
            raw_logit=raw_logit,
            mc_samples=[],
            metadata={"predicted_class": self.spec.class_names[int(prob >= 0.5)]},
        )


class InferenceFuser:
    """Fuse multiple handwriting models (if available)."""

    def __init__(
        self,
        modality: str,
        models: list[DenseNet201BinaryModel],
        weights: dict[str, float],
    ):
        self.modality = modality
        self.models = models

        total = sum(weights.values())
        if total <= 0:
            raise ValueError(
                f"InferenceFuser({modality}): weight values must sum > 0"
            )
        self.weights = {model_id: weight / total for model_id, weight in weights.items()}

    def fuse_one(self, input_data) -> ModalityResult:
        """Fuse predictions from all models for single sample."""
        outputs: list[ModelOutput] = []

        for model in self.models:
            try:
                outputs.append(model.predict(input_data))
            except Exception:
                continue

        if not outputs:
            return ModalityResult(
                modality=self.modality,
                available=False,
                probability=0.5,
                ci_low=0.0,
                ci_high=1.0,
                ci_width=1.0,
                shap_features=[],
                model_ids=[],
                metadata={"reason": "all_sub_models_failed"},
            )

        used_ids = [output.model_id for output in outputs]
        raw_weights = np.array(
            [self.weights.get(output.model_id, 1.0) for output in outputs],
            dtype=float,
        )
        used_weights = (
            raw_weights / raw_weights.sum()
            if raw_weights.sum() > 0
            else np.ones(len(outputs), dtype=float) / len(outputs)
        )
        probabilities = np.array([output.probability for output in outputs], dtype=float)
        fused_probability = float(np.dot(used_weights, probabilities))

        ci_low = (
            float(np.percentile(probabilities, 2.5))
            if len(probabilities) > 1
            else fused_probability
        )
        ci_high = (
            float(np.percentile(probabilities, 97.5))
            if len(probabilities) > 1
            else fused_probability
        )

        return ModalityResult(
            modality=self.modality,
            available=True,
            probability=fused_probability,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_width=ci_high - ci_low,
            shap_features=[],
            model_ids=used_ids,
            metadata={
                "ensemble_size": len(outputs),
                "ensemble_weights": {
                    output.model_id: float(weight)
                    for output, weight in zip(outputs, used_weights)
                },
            },
        )


def build_handwriting_model() -> DenseNet201BinaryModel:
    """Create handwriting model instance."""
    return DenseNet201BinaryModel(
        DenseNet201BinarySpec(modality="handwriting", model_id="handwriting_densenet201")
    )


@lru_cache(maxsize=1)
def get_handwriting_model() -> DenseNet201BinaryModel:
    """Singleton handwriting model."""
    return build_handwriting_model()


def build_handwriting_fuser() -> InferenceFuser:
    """Create handwriting fuser."""
    model = get_handwriting_model()
    weights = {model.MODEL_ID: float(getattr(model, "VALIDATION_AUC", 1.0))}
    return InferenceFuser("handwriting", [model], weights)


@lru_cache(maxsize=1)
def get_handwriting_fuser() -> InferenceFuser:
    """Singleton handwriting fuser."""
    return build_handwriting_fuser()


def predict_handwriting(input_data):
    """Run single handwriting sample through fuser."""
    return get_handwriting_fuser().fuse_one(input_data)


def predict_handwriting_batch(samples):
    """Run multiple handwriting samples and aggregate."""
    results = [predict_handwriting(sample) for sample in samples]
    if not results:
        return ModalityResult(
            modality="handwriting",
            available=False,
            probability=0.5,
            ci_low=0.0,
            ci_high=1.0,
            ci_width=1.0,
            shap_features=[],
            model_ids=[],
            metadata={"n_samples": 0},
        )

    probabilities = np.array([result.probability for result in results], dtype=float)
    return ModalityResult(
        modality="handwriting",
        available=any(result.available for result in results),
        probability=float(np.median(probabilities)),
        ci_low=float(np.min(probabilities)),
        ci_high=float(np.max(probabilities)),
        ci_width=float(np.max(probabilities) - np.min(probabilities)),
        shap_features=[],
        model_ids=[model_id for result in results for model_id in result.model_ids],
        metadata={
            "n_samples": len(results),
            "per_sample_probabilities": [
                round(float(p), 4) for p in probabilities
            ],
        },
    )


__all__ = [
    "DenseNet201BinarySpec",
    "DenseNet201BinaryModel",
    "InferenceFuser",
    "build_handwriting_model",
    "get_handwriting_model",
    "build_handwriting_fuser",
    "get_handwriting_fuser",
    "predict_handwriting",
    "predict_handwriting_batch",
]
