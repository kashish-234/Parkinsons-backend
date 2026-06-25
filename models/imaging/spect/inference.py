from __future__ import annotations

from functools import lru_cache

from common.intra_model import aggregate_modality_samples, build_modality_fuser

from spect.model import SPECT_SPEC, build_spect_model, get_spect_model


def build_spect_fuser():
    return build_modality_fuser("spect", [SPECT_SPEC])


@lru_cache(maxsize=1)
def get_spect_fuser():
    return build_spect_fuser()


def predict_spect(input_data):
    return get_spect_fuser().fuse_one(input_data)


def predict_spect_batch(samples):
    return aggregate_modality_samples([get_spect_fuser().fuse_one(sample) for sample in samples])


__all__ = [
    "build_spect_fuser",
    "get_spect_fuser",
    "predict_spect",
    "predict_spect_batch",
    "build_spect_model",
    "get_spect_model",
]