"""imaging/spect/inference.py"""
from __future__ import annotations
import logging

from models.imaging.spect.model import get_spect_model
from models.base.contracts import ModalityResult

logger = logging.getLogger(__name__)


def predict_spect(input_data) -> ModalityResult:
    try:
        model = get_spect_model()
        model_output = model.predict(input_data)
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
    # FIX H3: use imaging's own aggregate function
    from models.imaging.intra_model import aggregate_modality_samples
    results = [predict_spect(sample) for sample in samples]
    return aggregate_modality_samples(results)


__all__ = ["predict_spect", "predict_spect_batch"]