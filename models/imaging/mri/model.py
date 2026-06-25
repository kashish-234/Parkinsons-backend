"""
imaging/mri/model.py

PyTorch is imported lazily inside load_mri_components() so that importing
this module at application startup does not immediately allocate PyTorch RAM.
"""
from __future__ import annotations

from functools import lru_cache
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from services.model_storage_service import model_storage_service
from models.base.contracts import ModelOutput, SHAPFeature

logger = logging.getLogger(__name__)


def _get_torch():
    import torch
    return torch

def _get_transforms():
    import torchvision.transforms as transforms
    return transforms

def _get_resnet18():
    from torchvision.models import resnet18
    return resnet18


class MRIResNet18Model:
    """ResNet18 fine-tuned for MRI PD classification."""

    MODEL_ID = "mri_resnet18_v1"

    def __init__(self):
        self._components = None

    def _load(self):
        if self._components is None:
            self._components = load_mri_components()

    def predict(self, input_data) -> ModelOutput:
        self._load()
        torch = _get_torch()
        transforms = _get_transforms()
        model      = self._components["model"]
        calibrator = self._components["calibrator"]
        metadata   = self._components["metadata"]
        shap_ref   = self._components.get("shap_reference_batch")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        path = input_data if isinstance(input_data, str) else input_data.get("mri")
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            prob_raw = torch.softmax(logits, dim=1)[0, 1].item()

        prob = float(calibrator.predict_proba([[prob_raw]])[0, 1]) \
            if calibrator is not None else float(prob_raw)

        eps = 1e-6
        p_c = min(max(prob, eps), 1 - eps)
        raw_logit = float(np.log(p_c / (1 - p_c)))

        shap_features = []
        if shap_ref is not None:
            try:
                import shap
                explainer = shap.GradientExplainer(model, shap_ref)
                sv = explainer.shap_values(tensor)
                vals = np.asarray(sv[1] if isinstance(sv, list) else sv)[0]
                flat = vals.ravel()
                top = np.argsort(np.abs(flat))[::-1][:10]
                shap_features = [
                    SHAPFeature(name=f"pixel_{i}", value=float(flat[i]), rank=r)
                    for r, i in enumerate(top, 1)
                ]
            except Exception as e:
                logger.warning(f"MRI SHAP failed: {e}")

        return ModelOutput(
            model_id=self.MODEL_ID,
            modality="neuroimaging",
            dataset="mri",
            probability=prob,
            shap_features=shap_features,
            raw_logit=raw_logit,
            mc_samples=[],
            metadata={"validation_auc": metadata.get("validation_auc", 0.0)},
        )


@lru_cache(maxsize=1)
def load_mri_components() -> dict:
    """Download and load all MRI model artifacts. PyTorch imported here."""
    torch = _get_torch()
    resnet18 = _get_resnet18()

    components: dict = {}

    # Try combined pkl artifact first
    try:
        artifact_path = model_storage_service.download_model(
            "neuroimaging/mri/mri_model_artifact.pkl"
        )
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)
        if hasattr(artifact, "state_dict") or isinstance(artifact, torch.nn.Module):
            components["model"] = artifact
        elif isinstance(artifact, dict) and "model" in artifact:
            components.update(artifact)
        try:
            meta_path = model_storage_service.download_model(
                "neuroimaging/mri/metadata.json"
            )
            with open(meta_path) as f:
                components["metadata"] = json.load(f)
        except Exception:
            components.setdefault("metadata", {})
        logger.info("MRI model loaded from mri_model_artifact.pkl")
    except Exception as e:
        logger.warning(f"mri_model_artifact.pkl failed ({e}), trying model.pt ...")
        model_path = model_storage_service.download_model("neuroimaging/mri/model.pt")
        net = resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, 2)
        net.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        net.eval()
        components["model"] = net

    # Calibrator (optional)
    try:
        cal_path = model_storage_service.download_model(
            "neuroimaging/mri/calibrator.pkl"
        )
        import joblib
        components["calibrator"] = joblib.load(cal_path)
    except Exception:
        components["calibrator"] = None

    # Metadata (optional)
    if "metadata" not in components:
        try:
            meta_path = model_storage_service.download_model(
                "neuroimaging/mri/metadata.json"
            )
            with open(meta_path) as f:
                components["metadata"] = json.load(f)
        except Exception:
            components["metadata"] = {}

    # SHAP reference batch (optional)
    try:
        shap_path = model_storage_service.download_model(
            "neuroimaging/mri/shap_reference_batch.pt"
        )
        components["shap_reference_batch"] = torch.load(
            shap_path, map_location="cpu", weights_only=True
        )
    except Exception:
        components["shap_reference_batch"] = None

    return components


@lru_cache(maxsize=1)
def get_mri_model() -> MRIResNet18Model:
    return MRIResNet18Model()
