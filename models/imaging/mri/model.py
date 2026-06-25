from __future__ import annotations

from functools import lru_cache

from common.intra_model import (
    PROJECT_ROOT,
    ResNet18BinaryModel,
    ResNet18BinarySpec,
    preprocess_to_3ch_tensor,
)

MRI_SPEC = ResNet18BinarySpec(
    modality="mri",
    model_id="mri_resnet18",
    artifact_path=PROJECT_ROOT / "mri_model_artifact.pkl",
    apply_imagenet_norm=False,
)


def build_mri_model() -> ResNet18BinaryModel:
    return ResNet18BinaryModel(MRI_SPEC)


@lru_cache(maxsize=1)
def get_mri_model() -> ResNet18BinaryModel:
    return build_mri_model()


def preprocess_mri_slice(img_2d, size: int = 224):
    return preprocess_to_3ch_tensor(img_2d, size=size, apply_imagenet_norm=False)


__all__ = [
    "MRI_SPEC",
    "build_mri_model",
    "get_mri_model",
    "preprocess_mri_slice",
]