"""
imaging/mri/model.py
====================

MRI model wrapper using ResNet18 binary classification.
"""

from __future__ import annotations

from functools import lru_cache
import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18

from services.model_storage_service import model_storage_service
from models.base.contracts import ModelOutput, SHAPFeature

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MRIResNet18Model:
    """
    ResNet18-based model for MRI classification (PD vs HC).
    
    Loads pre-trained artifact from Hugging Face.
    """

    MODEL_ID = "mri_resnet18_v1"
    VALIDATION_AUC = 0.87  # Update based on your validation results

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model()
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> transforms.Compose:
        """MRI-specific preprocessing (no ImageNet normalization)."""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # MRI images are grayscale intensity; don't apply ImageNet norm
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def _load_model(self) -> torch.nn.Module:
        """Load ResNet18 from artifact."""
        try:
            artifact_path = model_storage_service.download_model(
                "mri/mri_resnet18.pt"
            )
            logger.info(f"Loading MRI model from: {artifact_path}")

            model = resnet18(pretrained=False, num_classes=2)
            # Adapt first layer to accept single-channel input
            model.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            checkpoint = torch.load(artifact_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(self.device)
            model.eval()

            logger.info("MRI model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load MRI model: {e}")
            raise

    def predict(self, input_data) -> ModelOutput:
        """
        Run MRI inference on input (image array or path).

        Args:
            input_data: Either:
                - numpy array (H, W) or (H, W, 1) grayscale image
                - Path-like object to MRI image file
                - dict with 'image' or 'path' key

        Returns:
            ModelOutput with probability and metadata
        """
        # Resolve input to numpy array
        image_array = self._resolve_input(input_data)

        # Preprocess: convert to tensor
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=-1)

        # Normalize to [0, 1]
        image_array = (image_array - image_array.min()) / (
            image_array.max() - image_array.min() + 1e-8
        )

        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(
            (image_array[:, :, 0] * 255).astype(np.uint8)
        ).convert("L")
        tensor_input = self.transforms(pil_image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor_input)  # (1, 2)
            probs = torch.softmax(logits, dim=1)

        prob_pd = float(probs[0, 1].item())
        predicted_label = int(prob_pd >= 0.5)

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="neuroimaging",
            dataset="mri",
            probability=prob_pd,
            shap_features=[],
            raw_logit=float(logits[0, 1].item()),
            mc_samples=[prob_pd],
            metadata={
                "predicted_label": predicted_label,
                "validation_auc": self.VALIDATION_AUC,
                "device": str(self.device),
                "input_shape": list(image_array.shape),
            },
        )

    def _resolve_input(self, input_data) -> np.ndarray:
        """Convert various input formats to numpy array."""
        if isinstance(input_data, dict):
            if "image" in input_data:
                return self._resolve_input(input_data["image"])
            if "path" in input_data:
                return self._resolve_input(input_data["path"])

        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if path.exists():
                img = Image.open(path).convert("L")
                return np.array(img, dtype=np.float32)
            else:
                raise FileNotFoundError(f"MRI image not found: {path}")

        if isinstance(input_data, np.ndarray):
            return input_data.astype(np.float32)

        # Try to convert to numpy
        array = np.asarray(input_data, dtype=np.float32)
        if array.ndim in (2, 3):
            return array

        raise ValueError(
            f"MRI input must be array, path, or dict, got {type(input_data)}"
        )


def build_mri_model() -> MRIResNet18Model:
    """Create MRI model instance."""
    return MRIResNet18Model()


@lru_cache(maxsize=1)
def get_mri_model() -> MRIResNet18Model:
    """Singleton MRI model."""
    return build_mri_model()


__all__ = ["MRIResNet18Model", "build_mri_model", "get_mri_model"]
