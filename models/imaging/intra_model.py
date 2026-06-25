"""
NOTES:
  - fuse_one() receives dicts like {"mri": "/tmp/.../scan.png"}  OR
    {"spect": "/tmp/.../scan.dcm"} — one sub-modality key at a time
    (as built by inference_pipeline._run_neuroimaging_inference).
  - predict_imaging_batch() handles a list of such single-key dicts and
    aggregates results across files via aggregate_modality_samples().
"""

import logging
import numpy as np
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

from models.base.contracts import ModelOutput, ModalityResult, SHAPFeature

logger = logging.getLogger(__name__)


class ImagingIntraFuser:
    """
    Fuses sub-modality predictions (MRI, SPECT) into one ModalityResult.

    Args:
        modality:      "neuroimaging"
        sub_modalities: Dict[str, list] mapping sub-modality → model instances
        weights:       Dict[str, float] mapping sub-modality → validation AUC
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

        self.weights = {k: v / total for k, v in weights.items()}
        logger.info(
            f"ImagingIntraFuser initialized — sub-modalities: "
            f"{list(sub_modalities.keys())}, normalized weights: {self.weights}"
        )

    def fuse_one(self, input_data: Dict) -> ModalityResult:
        """
        Run all imaging sub-modality models on input_data and fuse results.

        Args:
            input_data: Dict with sub-modality keys, e.g.
                        {"mri": "/path/to/scan.png"}  or
                        {"spect": "/path/to/scan.dcm"}  or
                        {"mri": arr, "spect": arr}   (numpy arrays)

        Returns:
            ModalityResult with fused probability and CI.
        """
        sub_results: Dict[str, List[ModelOutput]] = {}

        for sub_mod_name, models in self.sub_modalities.items():
            sub_input = input_data.get(sub_mod_name)
            if sub_input is None:
                logger.debug(
                    f"ImagingIntraFuser: no input for '{sub_mod_name}' — skipping."
                )
                continue

            results = []
            for model in models:
                try:
                    out = model.predict(sub_input)
                    results.append(out)
                except Exception as e:
                    logger.warning(
                        f"ImagingIntraFuser: {sub_mod_name}/{model.MODEL_ID} failed: {e}"
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

        fused_prob, sub_mod_weights = self._fuse_sub_modalities(sub_results)
        ci_low, ci_high = self._propagate_uncertainty(sub_results, fused_prob)
        merged_shap = self._merge_shap_features(sub_results)

        all_model_ids = [
            out.model_id
            for sub_mod_results in sub_results.values()
            for out in sub_mod_results
        ]

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
        sub_mod_probs = {
            sub: float(np.mean([o.probability for o in outputs]))
            for sub, outputs in sub_results.items()
        }

        used_weights = {
            sub: self.weights.get(sub, 1.0) for sub in sub_mod_probs
        }
        total_weight = sum(used_weights.values()) or len(sub_mod_probs)
        norm_weights = {k: v / total_weight for k, v in used_weights.items()}

        fused_prob = sum(
            norm_weights[sub] * sub_mod_probs[sub] for sub in sub_mod_probs
        )
        return float(fused_prob), norm_weights

    def _propagate_uncertainty(
        self,
        sub_results: Dict[str, List[ModelOutput]],
        fused_prob: float,
    ) -> Tuple[float, float]:
        all_probs = [
            out.probability
            for outputs in sub_results.values()
            for out in outputs
        ]

        if not all_probs:
            return max(0.0, fused_prob - 0.25), min(1.0, fused_prob + 0.25)

        arr = np.array(all_probs)
        prob_std = float(np.std(arr))
        ci_low  = max(0.0, min(float(np.min(arr)), fused_prob - prob_std))
        ci_high = min(1.0, max(float(np.max(arr)), fused_prob + prob_std))
        return ci_low, ci_high

    def _merge_shap_features(
        self,
        sub_results: Dict[str, List[ModelOutput]],
        top_n: int = 10,
    ) -> List[SHAPFeature]:
        feature_map: Dict[str, float] = {}

        for sub_mod_name, outputs in sub_results.items():
            for out in outputs:
                for shap_feat in getattr(out, "shap_features", []):
                    key = f"{sub_mod_name}::{shap_feat.name}"
                    feature_map[key] = feature_map.get(key, 0.0) + shap_feat.value

        if not feature_map:
            return []

        ranked = sorted(feature_map.items(), key=lambda x: abs(x[1]), reverse=True)
        return [
            SHAPFeature(
                name=name.split("::")[-1] + f" ({name.split('::')[0]})",
                value=float(value),
                rank=rank + 1,
            )
            for rank, (name, value) in enumerate(ranked[:top_n])
        ]


def build_imaging_models() -> Dict[str, List]:
    models: Dict[str, List] = {}

    try:
        from models.imaging.mri.inference import get_mri_model
        mri_model = get_mri_model()
        models["mri"] = [mri_model]
        logger.info("MRI model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load MRI model: {e}")

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
    models = build_imaging_models()

    weights = {}
    for sub_mod_name, model_list in models.items():
        if model_list:
            auc = float(getattr(model_list[0], "VALIDATION_AUC", 1.0))
            weights[sub_mod_name] = auc
            logger.info(f"{sub_mod_name} validation AUC: {auc}")

    return ImagingIntraFuser(modality="neuroimaging", sub_modalities=models, weights=weights)


@lru_cache(maxsize=1)
def get_imaging_fuser() -> ImagingIntraFuser:
    """Singleton imaging fuser — loaded once per worker."""
    return build_imaging_fuser()


def predict_imaging(input_data: Dict) -> ModalityResult:
    """Run a single imaging sample through the ensemble."""
    return get_imaging_fuser().fuse_one(input_data)


def predict_imaging_batch(samples: List[Dict]) -> ModalityResult:
    """
    Run N imaging samples (each a dict like {"mri": path} or {"spect": path})
    through the fuser, then aggregate across samples.
    """
    results = [predict_imaging(sample) for sample in samples]
    return aggregate_modality_samples(results)


def aggregate_modality_samples(file_results: List[ModalityResult]) -> ModalityResult:
    """
    Combine multiple per-file ModalityResults for neuroimaging
    (e.g. multiple MRI slices or scans) into one final result.

    Probability: MEDIAN across files (robust to one bad scan)
    CI:          widened by inter-file spread
    SHAP:        mean per feature across files
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
    modality  = file_results[0].modality

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

    probs      = np.array([r.probability for r in available])
    fused_prob = float(np.median(probs))

    pooled_low  = float(min(r.ci_low  for r in available))
    pooled_high = float(max(r.ci_high for r in available))

    if len(available) > 1:
        inter_std   = float(np.std(probs))
        pooled_low  = max(0.0, min(pooled_low,  fused_prob - inter_std))
        pooled_high = min(1.0, max(pooled_high, fused_prob + inter_std))

    n = len(available)
    feature_map: Dict[str, float] = {}
    for r in available:
        for f in r.shap_features:
            feature_map[f.name] = feature_map.get(f.name, 0.0) + f.value / n

    ranked = sorted(feature_map.items(), key=lambda x: abs(x[1]), reverse=True)
    shap_merged = [
        SHAPFeature(name=nm, value=v, rank=i + 1)
        for i, (nm, v) in enumerate(ranked[:10])
    ]

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