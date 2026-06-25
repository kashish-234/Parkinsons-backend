from __future__ import annotations

from functools import lru_cache
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18

from services.model_storage_service import model_storage_service
from models.base.contracts import ModelOutput, SHAPFeature

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class SPECTResNet18Model:
    """
    ResNet18-based model for SPECT classification (PD vs HC).

    Artifacts live under 'neuroimaging/dat-spect/' in the HF repo.
    """

    MODEL_ID = "spect_resnet18_v1"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._metadata: dict = {}
        self._calibrator = None
        self.model = self._load_model()
        self.transforms = self._build_transforms()
        self.VALIDATION_AUC: float = float(
            self._metadata.get("validation_auc", 0.89)
        )

    def _build_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_model(self) -> torch.nn.Module:
        """Load ResNet18 from the combined pkl artifact under dat-spect/."""
        # Primary: combined artifact
        try:
            artifact_path = model_storage_service.download_model(
                "neuroimaging/dat-spect/spect_model_artifact.pkl"
            )
            with open(artifact_path, "rb") as f:
                artifact = pickle.load(f)

            net = resnet18(num_classes=2)
            net.load_state_dict(artifact["state_dict"])
            net.to(self.device)
            net.eval()

            self._calibrator = artifact.get("calibrator")
            self._metadata = artifact.get("metadata", {})

            try:
                meta_path = model_storage_service.download_model(
                    "neuroimaging/dat-spect/metadata.json"
                )
                with open(meta_path) as f:
                    self._metadata.update(json.load(f))
            except Exception:
                pass

            logger.info("SPECT model loaded from spect_model_artifact.pkl")
            return net

        except Exception as e:
            logger.warning(f"spect_model_artifact.pkl failed ({e}), trying model.pt ...")

        # Fallback: raw state dict
        model_path = model_storage_service.download_model(
            "neuroimaging/dat-spect/model.pt"
        )
        net = resnet18(num_classes=2)
        net.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        net.to(self.device)
        net.eval()

        try:
            cal_path = model_storage_service.download_model(
                "neuroimaging/dat-spect/calibrator.pkl"
            )
            import joblib
            self._calibrator = joblib.load(cal_path)
        except Exception:
            pass

        try:
            meta_path = model_storage_service.download_model(
                "neuroimaging/dat-spect/metadata.json"
            )
            with open(meta_path) as f:
                self._metadata = json.load(f)
        except Exception:
            pass

        logger.info("SPECT model loaded from model.pt (fallback)")
        return net

    def predict(self, input_data) -> ModelOutput:
        """Run inference on a SPECT scan image."""
        image_array = self._resolve_input(input_data)

        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=-1)

        lo, hi = image_array.min(), image_array.max()
        image_array = (image_array - lo) / (hi - lo + 1e-8)

        # Convert to RGB for ResNet18
        pil_image = Image.fromarray(
            (image_array[:, :, 0] * 255).astype(np.uint8)
        ).convert("RGB")

        tensor_input = self.transforms(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1)

        prob_pd_raw = float(probs[0, 1].item())

        if self._calibrator is not None:
            try:
                prob_pd = float(
                    self._calibrator.predict_proba(np.array([[prob_pd_raw]]))[0, 1]
                )
            except Exception:
                prob_pd = prob_pd_raw
        else:
            prob_pd = prob_pd_raw

        # FIX: assign predicted_label before referencing in metadata
        predicted_label = int(prob_pd >= 0.5)

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="neuroimaging",
            dataset="dat-spect",
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
        if isinstance(input_data, dict):
            for key in ("image", "path", "file_path"):
                if key in input_data:
                    return self._resolve_input(input_data[key])

        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if not path.exists():
                raise FileNotFoundError(f"SPECT image not found: {path}")
            img = Image.open(path).convert("L")
            return np.array(img, dtype=np.float32)

        if isinstance(input_data, np.ndarray):
            return input_data.astype(np.float32)

        array = np.asarray(input_data, dtype=np.float32)
        if array.ndim in (2, 3):
            return array

        raise ValueError(
            f"SPECT input must be array, path, or dict, got {type(input_data)}"
        )


def build_spect_model() -> SPECTResNet18Model:
    return SPECTResNet18Model()


@lru_cache(maxsize=1)
def get_spect_model() -> SPECTResNet18Model:
    return build_spect_model()


__all__ = ["SPECTResNet18Model", "build_spect_model", "get_spect_model"]