from __future__ import annotations

from functools import lru_cache

from common.intra_model import aggregate_modality_samples, build_modality_fuser

from mri.model import MRI_SPEC, build_mri_model, get_mri_model


def build_mri_fuser():
    return build_modality_fuser("mri", [MRI_SPEC])


@lru_cache(maxsize=1)
def get_mri_fuser():
    return build_mri_fuser()


def predict_mri(input_data):
    return get_mri_fuser().fuse_one(input_data)


def predict_mri_batch(samples):
    return aggregate_modality_samples([get_mri_fuser().fuse_one(sample) for sample in samples])


__all__ = [
    "build_mri_fuser",
    "get_mri_fuser",
    "predict_mri",
    "predict_mri_batch",
    "build_mri_model",
    "get_mri_model",
]