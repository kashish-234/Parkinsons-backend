from functools import lru_cache
from models.speech.oxford_pd.model import SpeechOxfordLGBM
from models.speech.uci.model import SpeechUCIRF
from models.speech.mdvr.model import SpeechMDVRSVC
from models.speech.italian.model import SpeechItalianLGBM

import logging
import numpy as np
from models.base.contracts import ModelOutput, ModalityResult, SHAPFeature
 
logger = logging.getLogger(__name__)
 
 
class IntraModalFuser:
    """
    Combines all sub-models for ONE input into a single ModalityResult.
 
    Args:
        modality: e.g. "speech"
        models:   list of sub-model instances, each with:
                    .MODEL_ID: str
                    .predict(input) -> ModelOutput
        weights:  dict {MODEL_ID: validation_auc}
                  normalised internally, so absolute scale doesn't matter.
    """
 
    def __init__(self, modality: str, models: list, weights: dict):
        self.modality = modality
        self.models   = models
 
        total = sum(weights.values())
        if total <= 0:
            raise ValueError(
                f"IntraModalFuser({modality}): weight values must sum > 0"
            )
        self.weights = {k: v / total for k, v in weights.items()}
 
    def fuse_one(self, input_data) -> ModalityResult:
        """
        Run all sub-models on input_data, return a fused ModalityResult.
 
        input_data is passed directly to each sub-model's .predict() method.
        The type depends on the modality:
          speech      → dict of feature_name: float
          gait        → file path string or numpy array
          tapping     → file path string or DataFrame
        Sub-models define their own input contract; IntraModalFuser does
        not inspect or transform input_data.
        """
        outputs: list[ModelOutput] = []
 
        for model in self.models:
            try:
                out = model.predict(input_data)
                outputs.append(out)
            except Exception as e:
                logger.warning(
                    f"IntraModalFuser({self.modality}): "
                    f"{model.MODEL_ID} failed: {e}"
                )
 
        if not outputs:
            logger.error(
                f"IntraModalFuser({self.modality}): all sub-models failed"
            )
            return ModalityResult(
                modality=self.modality, available=False,
                probability=0.5, ci_low=0.0, ci_high=1.0, ci_width=1.0,
                shap_features=[], model_ids=[],
                metadata={"reason": "all_sub_models_failed"},
            )
 
        # AUC-weighted average
        used_ids     = [o.model_id for o in outputs]
        raw_weights  = np.array([self.weights.get(o.model_id, 1.0) for o in outputs])
        w_sum        = raw_weights.sum()
        used_weights = raw_weights / w_sum if w_sum > 0 else np.ones(len(outputs)) / len(outputs)
 
        probs        = np.array([o.probability for o in outputs])
        fused_prob   = float(np.dot(used_weights, probs))
 
        # Pool MC samples → CI
        # Do NOT weight uncertainty samples.
        # Weights affect probability fusion, not uncertainty distribution.
        all_mc = []

        for o in outputs:
            if o.mc_samples:
                all_mc.extend(o.mc_samples)
 
        if all_mc:
            ci_low  = float(np.percentile(all_mc, 2.5))
            ci_high = float(np.percentile(all_mc, 97.5))
        else:
            logger.warning(
                f"IntraModalFuser({self.modality}): no mc_samples from any sub-model. "
                "Using fallback CI (fused_prob ± 0.25). Re-run training to fix."
            )
            ci_low  = max(0.0, fused_prob - 0.25)
            ci_high = min(1.0, fused_prob + 0.25)
 
        # Merge SHAP features
        shap_merged = self._merge_shap(outputs, used_weights)
 
        # Merge metadata (prefix by model_id on key collision)
        merged_meta: dict = {}
        for o in outputs:
            for k, v in o.metadata.items():
                merged_meta[f"{o.model_id}__{k}"] = v
 
        return ModalityResult(
            modality=self.modality,
            available=True,
            probability=fused_prob,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_width=ci_high - ci_low,
            shap_features=shap_merged,
            model_ids=used_ids,
            metadata={
                **merged_meta,
                "ensemble_size": len(outputs),
                "ensemble_weights": {
                    o.model_id: float(w)
                    for o, w in zip(outputs, used_weights)
                },
            },
        )
 
    def _merge_shap(self, outputs: list, weights: np.ndarray) -> list:
        """
        Weighted average of SHAP values across sub-models, merged by
        canonical feature name. Features unique to one sub-model are
        scaled by that model's weight.
        """
        feature_map: dict[str, float] = {}
        for o, w in zip(outputs, weights):
            for f in o.shap_features:
                feature_map[f.name] = feature_map.get(f.name, 0.0) + f.value * w
        ranked = sorted(feature_map.items(), key=lambda x: abs(x[1]), reverse=True)
        return [
            SHAPFeature(name=n, value=v, rank=i + 1)
            for i, (n, v) in enumerate(ranked[:10])
        ]
 
 
def aggregate_modality_samples(file_results: list) -> ModalityResult:
    """
    Combines MULTIPLE per-file ModalityResults for the SAME modality
    (e.g. patient submitted 6 speech recordings) into one final result.
 
    Strategy:
      probability  — MEDIAN across files (robust to one bad recording)
      ci_low/high  — widened by inter-file spread so disagreement between
                     files increases reported uncertainty
      shap_features — simple mean per canonical name across files
      model_ids    — union, de-duplicated
    """
    if not file_results:
        return ModalityResult(
            modality="unknown", available=False,
            probability=0.5, ci_low=0.0, ci_high=1.0, ci_width=1.0,
            shap_features=[], model_ids=[], metadata={"n_samples": 0},
        )
 
    available = [r for r in file_results if r.available]
    modality  = file_results[0].modality
 
    if not available:
        return ModalityResult(
            modality=modality, available=False,
            probability=0.5, ci_low=0.0, ci_high=1.0, ci_width=1.0,
            shap_features=[], model_ids=[],
            metadata={"n_samples": 0, "n_failed": len(file_results)},
        )
 
    probs      = np.array([r.probability for r in available])
    fused_prob = float(np.median(probs))
 
    # Pool CI bounds, then widen by inter-file std
    pooled_low  = float(min(r.ci_low  for r in available))
    pooled_high = float(max(r.ci_high for r in available))
    if len(available) > 1:
        inter_std   = float(np.std(probs))
        pooled_low  = max(0.0, min(pooled_low,  fused_prob - inter_std))
        pooled_high = min(1.0, max(pooled_high, fused_prob + inter_std))
 
    # Average SHAP across files (equal weight — each file is one observation)
    n = len(available)
    feature_map: dict[str, float] = {}
    for r in available:
        for f in r.shap_features:
            feature_map[f.name] = feature_map.get(f.name, 0.0) + f.value / n
    ranked = sorted(feature_map.items(), key=lambda x: abs(x[1]), reverse=True)
    shap_merged = [SHAPFeature(name=nm, value=v, rank=i+1) for i,(nm,v) in enumerate(ranked[:10])]
 
    all_model_ids: list[str] = []
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
            "n_samples":                   len(available),
            "n_samples_failed":            len(file_results) - len(available),
            "per_sample_probabilities":    [round(float(p), 4) for p in probs],
        },
    )
 
def build_speech_models():
    """
    Create all speech sub-models.
    """

    return [
        SpeechOxfordLGBM(),
        SpeechUCIRF(),
        SpeechMDVRSVC(),
        SpeechItalianLGBM(),
    ]


def build_speech_fuser():
    """
    Create speech ensemble using validation AUCs.
    """

    models = build_speech_models()

    weights = {}

    for model in models:

        auc = getattr(
            model,
            "VALIDATION_AUC",
            1.0
        )

        weights[model.MODEL_ID] = float(auc)

    return IntraModalFuser(
        modality="speech",
        models=models,
        weights=weights,
    )


@lru_cache(maxsize=1)
def get_speech_fuser():
    """
    Singleton speech ensemble.
    Models are loaded once per worker.
    """

    return build_speech_fuser()


def predict_speech(input_data):
    """
    Single speech sample.
    """

    return get_speech_fuser().fuse_one(
        input_data
    )


def predict_speech_batch(samples):
    """
    Multiple speech recordings.
    """

    results = [
        predict_speech(sample)
        for sample in samples
    ]

    return aggregate_modality_samples(
        results
    )