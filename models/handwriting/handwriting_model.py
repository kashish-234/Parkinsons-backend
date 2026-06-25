"""
handwriting/model.py
====================

Handwriting analysis model using DenseNet201 architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model

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


def _build_architecture(input_size: int = 224) -> Model:
    """Build DenseNet201 fine-tuned architecture."""
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


@dataclass
class DenseNet201BinaryModel:
    """Handwriting classification model using DenseNet201."""

    spec: DenseNet201BinarySpec
    model: Model | None = None
    model_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model, self.model_meta = self._load_model()

    @property
    def MODEL_ID(self) -> str:
        return self.spec.model_id

    @property
    def VALIDATION_AUC(self) -> float:
        return float(self.model_meta.get("best_val_auc", 1.0))

    def _find_artifact_path(self) -> Path:
        """Find model artifact among candidate paths."""
        for candidate in _candidate_paths(self.spec.artifact_paths):
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Handwriting model artifact not found. Looked for: "
            + ", ".join(str(candidate) for candidate in self.spec.artifact_paths)
        )

    def _load_model(self) -> tuple[Model, dict[str, Any]]:
        """Load model from artifact or download from HuggingFace."""
        try:
            # Try to download from HuggingFace
            artifact_path = model_storage_service.download_model(
                "handwriting/handwriting_model.keras"
            )
        except Exception:
            # Fallback to local candidates
            artifact_path = self._find_artifact_path()

        try:
            model = tf.keras.models.load_model(artifact_path)
            model_meta: dict[str, Any] = {}
        except Exception:
            model = _build_architecture(self.spec.input_size)
            try:
                model.load_weights(artifact_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to load handwriting model artifact: {artifact_path}"
                ) from exc
            model_meta = {}

        if "class_names" not in model_meta:
            model_meta["class_names"] = self.spec.class_names

        if "artifact_path" not in model_meta:
            model_meta["artifact_path"] = str(artifact_path)

        return model, model_meta

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert various types to numpy array."""
        if isinstance(value, tf.Tensor):
            return value.numpy()
        return np.asarray(value)

    @staticmethod
    def _load_image_from_path(image_path: Path) -> np.ndarray:
        """Load image from disk."""
        image = Image.open(image_path).convert("RGB")
        return np.asarray(image, dtype=np.float32)

    def _resolve_image(self, input_data: Any) -> np.ndarray:
        """Resolve various input formats to image array."""
        if isinstance(input_data, dict):
            if "image" in input_data:
                return self._resolve_image(input_data["image"])

            for key in ("image_path", "path", "file_path"):
                if key in input_data:
                    return self._resolve_image(input_data[key])

        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if path.is_dir():
                image_files = sorted(
                    candidate
                    for candidate in path.iterdir()
                    if candidate.is_file()
                    and candidate.suffix.lower()
                    in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                )
                if not image_files:
                    raise FileNotFoundError(
                        f"No image files found in handwriting directory: {path}"
                    )
                return self._load_image_from_path(image_files[0])

            if not path.exists():
                raise FileNotFoundError(f"Handwriting image not found: {path}")
            return self._load_image_from_path(path)

        array = self._to_numpy(input_data)
        if array.ndim == 2:
            return array.astype(np.float32)
        if array.ndim == 3 and array.shape[-1] in {1, 3}:
            if array.shape[-1] == 1:
                return array[..., 0].astype(np.float32)
            return array.astype(np.float32)

        raise ValueError(
            f"Handwriting input must be an image path or 2D/3D array, got shape {getattr(array, 'shape', None)}"
        )

    def _preprocess(self, image: np.ndarray) -> tf.Tensor:
        """Preprocess image to model input format."""
        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        if tensor.shape.rank == 2:
            tensor = tf.expand_dims(tensor, axis=-1)
        if tensor.shape.rank != 3:
            raise ValueError(
                f"Expected image tensor rank 2 or 3, got shape {tensor.shape}"
            )

        if tensor.shape[-1] == 1:
            tensor = tf.image.grayscale_to_rgb(tensor)
        elif tensor.shape[-1] != 3:
            raise ValueError(
                f"Handwriting image must have 1 or 3 channels, got shape {tensor.shape}"
            )

        tensor = tf.image.resize(
            tensor, (self.spec.input_size, self.spec.input_size), method="bilinear"
        )
        tensor = tensor / 255.0
        return tensor

    def predict(self, input_data) -> ModelOutput:
        """Run inference on handwriting image."""
        image = self._resolve_image(input_data)
        x = tf.expand_dims(self._preprocess(image), axis=0)

        probability = float(self.model.predict(x, verbose=0).reshape(-1)[0])
        prediction = int(probability >= 0.5)

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="handwriting",
            dataset="handwriting",
            probability=probability,
            shap_features=[],
            raw_logit=float(np.log(probability / (1 - probability + 1e-6))),
            mc_samples=[probability],
            metadata={
                "prediction": prediction,
                "artifact_path": self.model_meta.get("artifact_path", ""),
                "class_names": self.model_meta.get("class_names", self.spec.class_names),
                "input_shape": [self.spec.input_size, self.spec.input_size, 3],
                "validation_auc": self.VALIDATION_AUC,
            },
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
            except Exception as e:
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
                round(float(probability), 4) for probability in probabilities
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
