"""
imaging/intra_model.py
======================

Intra-modal fusion for neuroimaging sub-modalities (MRI and SPECT).

Each imaging sub-modality (MRI, SPECT) can have:
  - Multiple models (e.g., different architectures, training runs)
  - Per-model validation AUC scores as weights

This module:
  1. Loads all sub-modality models (MRI, SPECT)
  2. Runs them on input imaging data
  3. Fuses their predictions using AUC-weighted averaging
  4. Returns a single ModalityResult for 'neuroimaging'
"""

import logging
import numpy as np
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

from models.base.contracts import ModelOutput, ModalityResult, SHAPFeature

logger = logging.getLogger(__name__)


class ImagingIntraFuser:
    """
    Fuses sub-modality predictions (e.g., MRI, SPECT) into one imaging ModalityResult.

    Args:
        modality: "neuroimaging"
        sub_modalities: Dict[str, list] mapping sub-modality name → model instances
                        e.g. {"mri": [mri_model_1], "spect": [spect_model_1, spect_model_2]}
        weights: Dict[str, float] mapping sub-modality → validation AUC weight
                 e.g. {"mri": 0.85, "spect": 0.92}
    """

    def __init__(
        self,
        modality: str,
        sub_modalities: Dict[str, List],
        weights: Dict[str, float],
    ):
        self.modality = modality
        self.sub_modalities = sub_modalities

        total = sum(weights.values())
        if total <= 0:
            raise ValueError("ImagingIntraFuser: weight values must sum > 0")

        # Normalize weights to sum to 1.0
        self.weights = {k: v / total for k, v in weights.items()}
        logger.info(
            f"ImagingIntraFuser initialized with sub-modalities: {list(sub_modalities.keys())} "
            f"and normalized weights: {self.weights}"
        )

    def fuse_one(self, input_data: Dict) -> ModalityResult:
        """
        Run all imaging sub-modality models on input_data and fuse results.

        Args:
            input_data: Dict with keys matching sub-modality names
                        e.g. {"mri": mri_tensor, "spect": spect_tensor}
                        or {"mri_path": "...", "spect_path": "..."}

        Returns:
            ModalityResult with fused probability, CI, and top SHAP features
        """
        sub_results: Dict[str, List[ModelOutput]] = {}

        # Run each sub-modality's models
        for sub_mod_name, models in self.sub_modalities.items():
            results = []

            # Get sub-modality specific data
            sub_input = input_data.get(sub_mod_name)
            if sub_input is None:
                logger.warning(
                    f"ImagingIntraFuser: no input data for sub-modality '{sub_mod_name}'. "
                    "Skipping."
                )
                continue

            for model in models:
                try:
                    out = model.predict(sub_input)
                    results.append(out)
                except Exception as e:
                    logger.warning(
                        f"ImagingIntraFuser: {sub_mod_name} model {model.MODEL_ID} failed: {e}"
                    )

            if results:
                sub_results[sub_mod_name] = results

        if not sub_results:
            logger.error(
                "ImagingIntraFuser: all sub-modality models failed. "
                "Returning unavailable ModalityResult."
            )
            return ModalityResult(
                modality=self.modality,
                available=False,
                probability=0.5,
                ci_low=0.0,
                ci_high=1.0,
                ci_width=1.0,
                shap_features=[],
                model_ids=[],
                metadata={"reason": "all_sub_modality_models_failed"},
            )

        # Fuse sub-modality predictions (average across sub-modalities)
        fused_prob, sub_mod_weights = self._fuse_sub_modalities(sub_results)

        # Propagate uncertainty
        ci_low, ci_high = self._propagate_uncertainty(sub_results, fused_prob)

        # Merge SHAP features
        merged_shap = self._merge_shap_features(sub_results)

        # Collect all model IDs
        all_model_ids = []
        for sub_mod_results in sub_results.values():
            for out in sub_mod_results:
                all_model_ids.append(out.model_id)

        return ModalityResult(
            modality=self.modality,
            available=True,
            probability=fused_prob,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_width=ci_high - ci_low,
            shap_features=merged_shap,
            model_ids=all_model_ids,
            metadata={
                "sub_modalities": list(sub_results.keys()),
                "sub_modality_weights": sub_mod_weights,
                "n_sub_modalities": len(sub_results),
                "n_models_per_sub": {
                    sub: len(outs) for sub, outs in sub_results.items()
                },
            },
        )

    def _fuse_sub_modalities(
        self, sub_results: Dict[str, List[ModelOutput]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Average predictions across sub-modalities using their weights.

        First, fuse within each sub-modality (simple average if multiple models).
        Then, fuse across sub-modalities using stored weights.

        Returns:
            (fused_probability, sub_modality_weights_used)
        """
        sub_mod_probs = {}

        for sub_mod_name, outputs in sub_results.items():
            # Within sub-modality fusion: simple average
            probs = np.array([o.probability for o in outputs])
            sub_mod_probs[sub_mod_name] = float(np.mean(probs))

        # Across sub-modality fusion: weighted average
        used_weights = {}
        total_weight = 0.0

        for sub_mod_name, prob in sub_mod_probs.items():
            weight = self.weights.get(sub_mod_name, 1.0)
            used_weights[sub_mod_name] = weight
            total_weight += weight

        if total_weight <= 0:
            total_weight = len(sub_mod_probs)
            used_weights = {k: 1.0 for k in sub_mod_probs.keys()}

        # Normalize and compute weighted average
        norm_weights = {k: v / total_weight for k, v in used_weights.items()}
        fused_prob = sum(
            norm_weights[sub] * sub_mod_probs[sub] for sub in sub_mod_probs.keys()
        )

        return float(fused_prob), norm_weights

    def _propagate_uncertainty(
        self,
        sub_results: Dict[str, List[ModelOutput]],
        fused_prob: float,
    ) -> Tuple[float, float]:
        """
        Pool confidence intervals across all sub-modality models.

        Returns:
            (ci_low, ci_high)
        """
        all_probs = []

        for outputs in sub_results.values():
            for out in outputs:
                all_probs.append(out.probability)

        if not all_probs:
            return max(0.0, fused_prob - 0.25), min(1.0, fused_prob + 0.25)

        # Consensus-based CI
        all_probs_arr = np.array(all_probs)
        prob_std = float(np.std(all_probs_arr))

        ci_low = max(0.0, min(np.min(all_probs_arr), fused_prob - prob_std))
        ci_high = min(1.0, max(np.max(all_probs_arr), fused_prob + prob_std))

        return float(ci_low), float(ci_high)

    def _merge_shap_features(
        self, sub_results: Dict[str, List[ModelOutput]], top_n: int = 10
    ) -> List[SHAPFeature]:
        """
        Merge SHAP features from all sub-modality models.

        Prefixes features with sub-modality name to avoid collisions:
        "mri::gray_matter_signal", "spect::striatal_binding", etc.

        Returns:
            Top N SHAP features sorted by absolute value
        """
        feature_map: Dict[str, float] = {}

        for sub_mod_name, outputs in sub_results.items():
            for out in outputs:
                if hasattr(out, "shap_features") and out.shap_features:
                    for shap_feat in out.shap_features:
                        prefixed_name = f"{sub_mod_name}::{shap_feat.name}"
                        feature_map[prefixed_name] = feature_map.get(
                            prefixed_name, 0.0
                        ) + shap_feat.value

        if not feature_map:
            return []

        # Rank by absolute value
        ranked = sorted(
            feature_map.items(), key=lambda x: abs(x[1]), reverse=True
        )

        shap_features = [
            SHAPFeature(
                name=name.split("::")[-1] + f" ({name.split('::')[0]})",
                value=float(value),
                rank=rank + 1,
            )
            for rank, (name, value) in enumerate(ranked[:top_n])
        ]

        return shap_features


def build_imaging_models() -> Dict[str, List]:
    """
    Build all imaging sub-modality model instances.

    Returns:
        Dict[str, list] mapping sub-modality name → list of model instances
    """
    models = {}

    # Try to load MRI models
    try:
        from models.imaging.mri.inference import get_mri_model

        mri_model = get_mri_model()
        models["mri"] = [mri_model]
        logger.info("MRI model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load MRI model: {e}")

    # Try to load SPECT models
    try:
        from models.imaging.spect.inference import get_spect_model

        spect_model = get_spect_model()
        models["spect"] = [spect_model]
        logger.info("SPECT model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load SPECT model: {e}")

    if not models:
        raise RuntimeError(
            "No imaging sub-modality models could be loaded. "
            "Check that MRI and SPECT artifacts are available."
        )

    return models


def build_imaging_fuser() -> ImagingIntraFuser:
    """
    Create imaging ensemble using sub-modality validation AUC as weights.

    Returns:
        ImagingIntraFuser instance
    """
    models = build_imaging_models()

    # Determine weights from validation AUC
    weights = {}
    for sub_mod_name, model_list in models.items():
        if model_list:
            # Use first model's validation AUC, or default to 1.0
            model = model_list[0]
            auc = float(getattr(model, "VALIDATION_AUC", 1.0))
            weights[sub_mod_name] = auc
            logger.info(f"{sub_mod_name} validation AUC: {auc}")

    return ImagingIntraFuser(
        modality="neuroimaging", sub_modalities=models, weights=weights
    )


@lru_cache(maxsize=1)
def get_imaging_fuser() -> ImagingIntraFuser:
    """
    Singleton imaging fuser.

    Models are loaded once per worker — artifacts cached in
    /tmp/pd_models on first call, then served from disk.
    """
    return build_imaging_fuser()


def predict_imaging(input_data: Dict) -> ModalityResult:
    """
    Run imaging sub-modalities through ensemble and fuse.

    Args:
        input_data: Dict with keys matching sub-modality names
                    e.g. {"mri": ..., "spect": ...}

    Returns:
        ModalityResult for 'neuroimaging' modality
    """
    return get_imaging_fuser().fuse_one(input_data)


def predict_imaging_batch(samples: List[Dict]) -> ModalityResult:
    """
    Run N imaging samples through fuser, then aggregate across samples.

    Args:
        samples: List of dicts, each with sub-modality data

    Returns:
        Aggregated ModalityResult
    """
    results = [predict_imaging(sample) for sample in samples]
    return aggregate_modality_samples(results)


def aggregate_modality_samples(file_results: List[ModalityResult]) -> ModalityResult:
    """
    Combine MULTIPLE per-file ModalityResults for neuroimaging
    (e.g., multiple image slices or scans) into one final result.

    Probability: MEDIAN across files (robust to one bad scan)
    CI: widened by inter-file spread
    SHAP: mean per feature across files
    Model IDs: union, de-duplicated

    Args:
        file_results: List of ModalityResults

    Returns:
        Aggregated ModalityResult
    """
    if not file_results:
        return ModalityResult(
            modality="neuroimaging",
            available=False,
            probability=0.5,
            ci_low=0.0,
            ci_high=1.0,
            ci_width=1.0,
            shap_features=[],
            model_ids=[],
            metadata={"n_samples": 0},
        )

    available = [r for r in file_results if r.available]
    modality = file_results[0].modality

    if not available:
        return ModalityResult(
            modality=modality,
            available=False,
            probability=0.5,
            ci_low=0.0,
            ci_high=1.0,
            ci_width=1.0,
            shap_features=[],
            model_ids=[],
            metadata={"n_samples": 0, "n_failed": len(file_results)},
        )

    # Aggregate probability via median
    probs = np.array([r.probability for r in available])
    fused_prob = float(np.median(probs))

    # Pool and widen CI
    pooled_low = float(min(r.ci_low for r in available))
    pooled_high = float(max(r.ci_high for r in available))

    if len(available) > 1:
        inter_std = float(np.std(probs))
        pooled_low = max(0.0, min(pooled_low, fused_prob - inter_std))
        pooled_high = min(1.0, max(pooled_high, fused_prob + inter_std))

    # Merge SHAP features
    feature_map: Dict[str, float] = {}
    n = len(available)
    for r in available:
        for f in r.shap_features:
            feature_map[f.name] = feature_map.get(f.name, 0.0) + f.value / n

    ranked = sorted(feature_map.items(), key=lambda x: abs(x[1]), reverse=True)
    shap_merged = [
        SHAPFeature(name=nm, value=v, rank=i + 1)
        for i, (nm, v) in enumerate(ranked[:10])
    ]

    # Collect model IDs
    all_model_ids: List[str] = []
    for r in available:
        for mid in r.model_ids:
            if mid not in all_model_ids:
                all_model_ids.append(mid)

    return ModalityResult(
        modality=modality,
        available=True,
        probability=fused_prob,
        ci_low=pooled_low,
        ci_high=pooled_high,
        ci_width=pooled_high - pooled_low,
        shap_features=shap_merged,
        model_ids=all_model_ids,
        metadata={
            "n_samples": len(available),
            "n_samples_failed": len(file_results) - len(available),
            "per_sample_probabilities": [round(float(p), 4) for p in probs],
        },
    )
