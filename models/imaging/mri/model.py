from __future__ import annotations

from functools import lru_cache
import json
import logging
import pickle
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


class MRIResNet18Model:
    """
    ResNet18-based model for MRI binary classification (PD vs HC).

    Loads from the combined 'mri_model_artifact.pkl' which contains:
      {"state_dict": ..., "calibrator": ..., "metadata": ...}
    """

    MODEL_ID = "mri_resnet18_v1"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._metadata: dict = {}
        self._calibrator = None
        self.model = self._load_model()
        self.transforms = self._build_transforms()
        self.VALIDATION_AUC: float = float(
            self._metadata.get("validation_auc", 0.87)
        )

    def _build_transforms(self) -> transforms.Compose:
        """MRI-specific preprocessing — grayscale normalised to [-1, 1]."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_model(self) -> torch.nn.Module:
        """Load ResNet18 from the combined pkl artifact."""
        # Primary: combined artifact
        try:
            artifact_path = model_storage_service.download_model(
                "neuroimaging/mri/mri_model_artifact.pkl"
            )
            with open(artifact_path, "rb") as f:
                artifact = pickle.load(f)

            net = resnet18(num_classes=2)
            net.load_state_dict(artifact["state_dict"])
            net.to(self.device)
            net.eval()

            self._calibrator = artifact.get("calibrator")
            self._metadata = artifact.get("metadata", {})

            # Also try to load richer metadata from the separate JSON
            try:
                meta_path = model_storage_service.download_model(
                    "neuroimaging/mri/metadata.json"
                )
                with open(meta_path) as f:
                    self._metadata.update(json.load(f))
            except Exception:
                pass

            logger.info("MRI model loaded from mri_model_artifact.pkl")
            return net

        except Exception as e:
            logger.warning(f"mri_model_artifact.pkl failed ({e}), trying model.pt ...")

        # Fallback: raw state dict
        model_path = model_storage_service.download_model("neuroimaging/mri/model.pt")
        net = resnet18(num_classes=2)
        net.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        net.to(self.device)
        net.eval()

        try:
            cal_path = model_storage_service.download_model(
                "neuroimaging/mri/calibrator.pkl"
            )
            self._calibrator = joblib.load(cal_path)
        except Exception:
            pass

        try:
            meta_path = model_storage_service.download_model(
                "neuroimaging/mri/metadata.json"
            )
            with open(meta_path) as f:
                self._metadata = json.load(f)
        except Exception:
            pass

        logger.info("MRI model loaded from model.pt (fallback)")
        return net

    def predict(self, input_data) -> ModelOutput:
        """Run inference on an MRI image."""
        image_array = self._resolve_input(input_data)

        # Ensure 2-D or 3-D
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=-1)  # (H, W, 1)

        # Normalise intensity to [0, 1] before PIL conversion
        lo, hi = image_array.min(), image_array.max()
        image_array = (image_array - lo) / (hi - lo + 1e-8)

        # Convert to RGB PIL image
        pil_image = Image.fromarray(
            (image_array[:, :, 0] * 255).astype(np.uint8)
        ).convert("RGB")  # ResNet18 expects 3-channel input

        tensor_input = self.transforms(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_input)   # (1, 2)
            probs = torch.softmax(logits, dim=1)

        prob_pd_raw = float(probs[0, 1].item())

        # Post-hoc calibration
        if self._calibrator is not None:
            try:
                prob_pd = float(
                    self._calibrator.predict_proba(np.array([[prob_pd_raw]]))[0, 1]
                )
            except Exception:
                prob_pd = prob_pd_raw
        else:
            prob_pd = prob_pd_raw

        # FIX: assign predicted_label before using it in metadata
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
            for key in ("image", "path", "file_path"):
                if key in input_data:
                    return self._resolve_input(input_data[key])

        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if not path.exists():
                raise FileNotFoundError(f"MRI image not found: {path}")
            img = Image.open(path).convert("L")
            return np.array(img, dtype=np.float32)

        if isinstance(input_data, np.ndarray):
            return input_data.astype(np.float32)

        array = np.asarray(input_data, dtype=np.float32)
        if array.ndim in (2, 3):
            return array

        raise ValueError(
            f"MRI input must be array, path, or dict, got {type(input_data)}"
        )


def build_mri_model() -> MRIResNet18Model:
    return MRIResNet18Model()


@lru_cache(maxsize=1)
def get_mri_model() -> MRIResNet18Model:
    return build_mri_model()


__all__ = ["MRIResNet18Model", "build_mri_model", "get_mri_model"]