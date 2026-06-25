"""
imaging/spect/inference.py
==========================

SPECT inference wrapper that loads model and provides predict functions.
"""

from __future__ import annotations

from functools import lru_cache
import logging

from models.imaging.spect.model import build_spect_model, get_spect_model
from models.base.contracts import ModalityResult

logger = logging.getLogger(__name__)


def predict_spect(input_data) -> ModalityResult:
    """
    Run single SPECT scan through model and return ModalityResult.

    Args:
        input_data: SPECT image (array, path, or dict)

    Returns:
        ModalityResult with probability and metadata
    """
    try:
        model = get_spect_model()
        model_output = model.predict(input_data)

        # Wrap ModelOutput in ModalityResult
        return ModalityResult(
            modality="neuroimaging",
            available=True,
            probability=model_output.probability,
            ci_low=max(0.0, model_output.probability - 0.1),
            ci_high=min(1.0, model_output.probability + 0.1),
            ci_width=0.2,
            shap_features=model_output.shap_features,
            model_ids=[model_output.model_id],
            metadata=model_output.metadata,
        )
    except Exception as e:
        logger.error(f"SPECT inference failed: {e}", exc_info=True)
        return ModalityResult(
            modality="neuroimaging",
            available=False,
            probability=0.5,
            ci_low=0.0,
            ci_high=1.0,
            ci_width=1.0,
            shap_features=[],
            model_ids=[],
            metadata={"error": str(e)},
        )


def predict_spect_batch(samples) -> ModalityResult:
    """
    Run multiple SPECT images through model and aggregate results.

    Args:
        samples: List of SPECT images

    Returns:
        Aggregated ModalityResult
    """
    import numpy as np
    from models.speech.intra_model import aggregate_modality_samples

    results = [predict_spect(sample) for sample in samples]
    return aggregate_modality_samples(results)


__all__ = ["predict_spect", "predict_spect_batch"]
