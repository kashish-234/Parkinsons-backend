from __future__ import annotations

from functools import lru_cache

from common.intra_model import (
    PROJECT_ROOT,
    ResNet18BinaryModel,
    ResNet18BinarySpec,
    preprocess_to_3ch_tensor,
)

SPECT_SPEC = ResNet18BinarySpec(
    modality="spect",
    model_id="spect_resnet18",
    artifact_path=PROJECT_ROOT / "best_spect_resnet18.pt",
    apply_imagenet_norm=True,
)


def build_spect_model() -> ResNet18BinaryModel:
    return ResNet18BinaryModel(SPECT_SPEC)


@lru_cache(maxsize=1)
def get_spect_model() -> ResNet18BinaryModel:
    return build_spect_model()


def preprocess_spect_slice(img_2d, size: int = 224):
    return preprocess_to_3ch_tensor(img_2d, size=size, apply_imagenet_norm=True)


__all__ = [
    "SPECT_SPEC",
    "build_spect_model",
    "get_spect_model",
    "preprocess_spect_slice",
]