"""
imaging/spect/model.py
======================

SPECT model wrapper using ResNet18 binary classification.
"""

from __future__ import annotations

from functools import lru_cache
import logging
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

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SPECTResNet18Model:
    """
    ResNet18-based model for SPECT classification (PD vs HC).
    
    Uses ImageNet pre-training and normalization.
    Loads pre-trained artifact from Hugging Face.
    """

    MODEL_ID = "spect_resnet18_v1"
    VALIDATION_AUC = 0.89  # Update based on your validation results

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self._load_model()
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> transforms.Compose:
        """SPECT-specific preprocessing with ImageNet normalization."""
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # ImageNet normalization (appropriate for SPECT with pre-training)
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ]
        )

    def _load_model(self) -> torch.nn.Module:
        try:
            # FIX C7: correct HF path
            artifact_path = model_storage_service.download_model(
                "neuroimaging/dat-spect/model.pt"
            )
            logger.info(f"Loading SPECT model from: {artifact_path}")

            model = resnet18(pretrained=False, num_classes=2)
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

            # FIX H9: Load calibrator
            try:
                import joblib
                cal_path = model_storage_service.download_model(
                    "neuroimaging/dat-spect/calibrator.pkl"
                )
                self._calibrator = joblib.load(cal_path)
                logger.info("SPECT calibrator loaded.")
            except Exception as e:
                logger.warning(f"SPECT calibrator not loaded: {e}")
                self._calibrator = None

            logger.info("SPECT model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load SPECT model: {e}")
            raise

    def predict(self, input_data) -> ModelOutput:
        """
        Run SPECT inference on input (image array or path).

        Args:
            input_data: Either:
                - numpy array (H, W) or (H, W, 1) grayscale image
                - Path-like object to SPECT image file
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
        
        # Note: transforms expect 3-channel input (will convert L to RGB)
        pil_image_rgb = Image.fromarray(
            np.array([
                (image_array[:, :, 0] * 255).astype(np.uint8),
                (image_array[:, :, 0] * 255).astype(np.uint8),
                (image_array[:, :, 0] * 255).astype(np.uint8),
            ]).transpose(1, 2, 0)
        )
        
        tensor_input = self.transforms(pil_image_rgb).unsqueeze(0).to(
            self.device
        )

        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1)

        prob_pd_raw = float(probs[0, 1].item())

        # Apply calibrator if loaded
        if self._calibrator is not None:
            try:
                import numpy as np
                prob_pd = float(
                    self._calibrator.predict_proba(
                        np.array([[prob_pd_raw]])
                    )[0, 1]
                )
            except Exception:
                prob_pd = prob_pd_raw
        else:
            prob_pd = prob_pd_raw

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="neuroimaging",
            dataset="spect",
            probability=prob_pd,
            shap_features=[],
            raw_logit=float(logits[0, 1].item()),
            mc_samples=[prob_pd],
            metadata={
                "predicted_label": predicted_label,
                "validation_auc": self.VALIDATION_AUC,
                "device": str(self.device),
                "input_shape": list(image_array.shape),
                "imagenet_normalized": True,
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
                raise FileNotFoundError(f"SPECT image not found: {path}")

        if isinstance(input_data, np.ndarray):
            return input_data.astype(np.float32)

        # Try to convert to numpy
        array = np.asarray(input_data, dtype=np.float32)
        if array.ndim in (2, 3):
            return array

        raise ValueError(
            f"SPECT input must be array, path, or dict, got {type(input_data)}"
        )


def build_spect_model() -> SPECTResNet18Model:
    """Create SPECT model instance."""
    return SPECTResNet18Model()


@lru_cache(maxsize=1)
def get_spect_model() -> SPECTResNet18Model:
    """Singleton SPECT model."""
    return build_spect_model()


__all__ = ["SPECTResNet18Model", "build_spect_model", "get_spect_model"]
