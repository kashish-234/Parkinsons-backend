"""
Multi-file support per patient:
  - Speech:       N CSV files  → each row is one recording → median-aggregate
  - Handwriting:  N image paths → per-image predict → median-aggregate
  - Gait:         N .npy files  → per-file predict  → median-aggregate
  - Tapping:      N CSV files   → per-row predict   → median-aggregate
  - Neuroimaging: files tagged with "mri" or "spect" in the filename are
                  routed to the correct sub-modality; or the caller may pass
                  sub-keys "mri" and "spect" inside data.files
  - REM:          N CSV files   → concat → single predict_proba call
"""

import os
import csv
import uuid
import shutil
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from models.base.contracts import FusedResult, ModalityResult
from models.speech.intra_model import predict_speech_batch, aggregate_modality_samples
from models.fusion.late_fusion import late_fusion_model

logger = logging.getLogger(__name__)

ALL_MODALITIES = ["neuroimaging", "speech", "tapping", "gait", "rem", "handwriting"]


# ============================================================================
# Singletons
# ============================================================================

@lru_cache(maxsize=1)
def get_gait_model():
    from models.gait.inference import GaitDaphnetCNNLSTM
    return GaitDaphnetCNNLSTM()


@lru_cache(maxsize=1)
def get_tapping_model():
    from models.finger_tapping.inference import FingerTappingLGBM
    return FingerTappingLGBM()


@lru_cache(maxsize=1)
def get_rem_pipeline():
    import joblib
    from models.rem.inference import REMInferencePipeline
    from services.model_storage_service import model_storage_service

    ensemble_path = model_storage_service.download_model("rem/rem_ensemble.pkl")
    metadata_path = model_storage_service.download_model("rem/modality_result.json")
    shap_path     = model_storage_service.download_model("rem/shap_explainer.pkl")

    return REMInferencePipeline.from_local(
        ensemble_path=ensemble_path,
        metadata_path=metadata_path,
        shap_path=shap_path,
    )


# ============================================================================
# Utility
# ============================================================================

def _make_unavailable(modality: str, reason: str) -> ModalityResult:
    return ModalityResult(
        modality=modality,
        available=False,
        probability=0.5,
        ci_low=0.0,
        ci_high=1.0,
        ci_width=1.0,
        shap_features=[],
        model_ids=[],
        metadata={"reason": reason},
    )


def _ci_from_mc(prob: float, mc_samples: list) -> tuple:
    """Derive CI from bootstrap / MC samples if available, else ±0.15 fallback."""
    import numpy as np
    if mc_samples and len(mc_samples) >= 5:
        arr = np.array(mc_samples, dtype=float)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return max(0.0, prob - 0.15), min(1.0, prob + 0.15)


# ============================================================================
# Speech
# ============================================================================

def _load_speech_file(path: str) -> List[dict]:
    samples = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = {}
                for k, v in row.items():
                    try:
                        sample[k.strip()] = float(v)
                    except (ValueError, TypeError):
                        pass
                if sample:
                    samples.append(sample)
    except Exception as e:
        logger.error(f"Failed to load speech file {path}: {e}")
    return samples


def _run_speech_inference(paths: List[str]) -> ModalityResult:
    try:
        all_samples: List[dict] = []
        for path in paths:
            samples = _load_speech_file(path)
            if not samples:
                logger.warning(f"Speech file produced no samples: {path}")
            all_samples.extend(samples)

        if not all_samples:
            return _make_unavailable("speech", "no_valid_feature_rows_in_files")

        result = predict_speech_batch(all_samples)
        logger.info(
            f"speech: prob={result.probability:.3f} "
            f"CI=[{result.ci_low:.3f},{result.ci_high:.3f}] "
            f"n_samples={result.metadata.get('n_samples', len(all_samples))}"
        )
        return result

    except Exception as e:
        logger.error(f"Speech inference failed: {e}", exc_info=True)
        return _make_unavailable("speech", f"inference_error: {e}")


# ============================================================================
# Neuroimaging (MRI + SPECT)
# ============================================================================

def _run_neuroimaging_inference(paths: List[str]) -> ModalityResult:
    """
    Accept a flat list of file paths and route them by filename substring.
    Convention: filenames must contain 'mri' or 'spect' (case-insensitive).

    If none are tagged, all are treated as MRI (best-effort fallback).
    """
    try:
        from models.imaging.intra_model import predict_imaging_batch

        if not paths:
            return _make_unavailable("neuroimaging", "no_files_provided")

        mri_paths   = [p for p in paths if "mri"   in Path(p).name.lower()]
        spect_paths = [p for p in paths if "spect" in Path(p).name.lower()]

        if not mri_paths and not spect_paths:
            logger.warning(
                "Neuroimaging files have no 'mri'/'spect' tag in filenames. "
                "Treating all as MRI."
            )
            mri_paths = paths

        imaging_samples = []
        for p in mri_paths:
            imaging_samples.append({"mri": p})
        for p in spect_paths:
            imaging_samples.append({"spect": p})

        result = predict_imaging_batch(imaging_samples)
        logger.info(
            f"neuroimaging: prob={result.probability:.3f} "
            f"CI=[{result.ci_low:.3f},{result.ci_high:.3f}]"
        )
        return result

    except Exception as e:
        logger.error(f"Neuroimaging inference failed: {e}", exc_info=True)
        return _make_unavailable("neuroimaging", f"inference_error: {e}")


# ============================================================================
# Finger Tapping
# ============================================================================

def _run_tapping_inference(paths: List[str]) -> ModalityResult:
    try:
        if not paths:
            return _make_unavailable("tapping", "no_files_provided")

        model = get_tapping_model()
        results: List[ModalityResult] = []

        for path in paths:
            try:
                samples = _load_speech_file(path)  # CSV loader reused
                for sample in samples:
                    model_output = model.predict(sample)
                    ci_low, ci_high = _ci_from_mc(
                        model_output.probability, model_output.mc_samples
                    )
                    result = ModalityResult(
                        modality="tapping",
                        available=True,
                        probability=model_output.probability,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        ci_width=ci_high - ci_low,
                        shap_features=model_output.shap_features,
                        model_ids=[model_output.model_id],
                        metadata=model_output.metadata,
                    )
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process tapping file {path}: {e}")

        if not results:
            return _make_unavailable("tapping", "inference_failed_for_all_files")

        return aggregate_modality_samples(results)

    except Exception as e:
        logger.error(f"Tapping inference failed: {e}", exc_info=True)
        return _make_unavailable("tapping", f"inference_error: {e}")


# ============================================================================
# Gait
# ============================================================================

def _run_gait_inference(paths: List[str]) -> ModalityResult:
    try:
        if not paths:
            return _make_unavailable("gait", "no_files_provided")

        import numpy as np
        model = get_gait_model()
        results: List[ModalityResult] = []

        for path in paths:
            try:
                # allow_pickle=False is safe for raw sensor windows
                window = np.load(path, allow_pickle=False)
                model_output = model.predict({"window": window})
                ci_low, ci_high = _ci_from_mc(
                    model_output.probability, model_output.mc_samples
                )
                result = ModalityResult(
                    modality="gait",
                    available=True,
                    probability=model_output.probability,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    ci_width=ci_high - ci_low,
                    shap_features=model_output.shap_features,
                    model_ids=[model_output.model_id],
                    metadata=model_output.metadata,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process gait file {path}: {e}")

        if not results:
            return _make_unavailable("gait", "inference_failed_for_all_files")

        return aggregate_modality_samples(results)

    except Exception as e:
        logger.error(f"Gait inference failed: {e}", exc_info=True)
        return _make_unavailable("gait", f"inference_error: {e}")


# ============================================================================
# REM
# ============================================================================

def _run_rem_inference(paths: List[str]) -> ModalityResult:
    """
    BUG FIXED: PD class is index 0 in LABEL_NAMES = {0: "PD", 1: "RB", 2: "HC"}.
    Original code correctly used proba[:, 0] — preserved here.
    Multiple CSV files are concatenated then passed as one batch.
    """
    try:
        import pandas as pd

        if not paths:
            return _make_unavailable("rem", "no_files_provided")

        dfs = []
        for path in paths:
            try:
                dfs.append(pd.read_csv(path))
            except Exception as e:
                logger.warning(f"Failed to load REM file {path}: {e}")

        if not dfs:
            return _make_unavailable("rem", "no_valid_rem_files")

        combined_df = pd.concat(dfs, ignore_index=True)
        pipeline = get_rem_pipeline()
        proba = pipeline.predict_proba(combined_df)  # shape (N, 3)

        prob_pd = float(proba[:, 0].mean())  # class 0 = PD
        logger.info(f"rem: prob_pd={prob_pd:.3f} from {len(combined_df)} rows")

        return ModalityResult(
            modality="rem",
            available=True,
            probability=prob_pd,
            ci_low=max(0.0, prob_pd - 0.15),
            ci_high=min(1.0, prob_pd + 0.15),
            ci_width=0.3,
            shap_features=[],
            model_ids=["rem_ensemble"],
            metadata={
                "n_rows": len(combined_df),
                "n_files": len(dfs),
                "per_class_proba": {
                    "PD": prob_pd,
                    "RB": float(proba[:, 1].mean()),
                    "HC": float(proba[:, 2].mean()),
                },
            },
        )

    except Exception as e:
        logger.error(f"REM inference failed: {e}", exc_info=True)
        return _make_unavailable("rem", f"inference_error: {e}")


# ============================================================================
# Handwriting
# ============================================================================

def _run_handwriting_inference(paths: List[str]) -> ModalityResult:
    """
    Each path is one handwriting image. predict_handwriting_batch accepts
    a list of paths and returns a median-aggregated ModalityResult.
    """
    try:
        from models.handwriting.model import predict_handwriting_batch

        if not paths:
            return _make_unavailable("handwriting", "no_files_provided")

        result = predict_handwriting_batch(paths)
        logger.info(
            f"handwriting: prob={result.probability:.3f} "
            f"CI=[{result.ci_low:.3f},{result.ci_high:.3f}] "
            f"n_images={len(paths)}"
        )
        return result

    except Exception as e:
        logger.error(f"Handwriting inference failed: {e}", exc_info=True)
        return _make_unavailable("handwriting", f"inference_error: {e}")


# ============================================================================
# Main pipeline
# ============================================================================

def run_inference(patient_id: str, input_paths: Dict[str, List[str]], job_id: str = None) -> FusedResult:
    job_id = job_id or str(uuid.uuid4())

    submitted = [m for m, paths in input_paths.items() if paths]
    logger.info(f"Job {job_id} | patient {patient_id} | submitted: {submitted}")

    modality_results: List[ModalityResult] = []

    for modality in ALL_MODALITIES:
        paths = input_paths.get(modality, [])

        if not paths:
            modality_results.append(_make_unavailable(modality, "not_submitted"))
            continue

        logger.info(f"Processing {modality} with {len(paths)} file(s)...")

        if modality == "speech":
            result = _run_speech_inference(paths)
        elif modality == "neuroimaging":
            result = _run_neuroimaging_inference(paths)
        elif modality == "tapping":
            result = _run_tapping_inference(paths)
        elif modality == "gait":
            result = _run_gait_inference(paths)
        elif modality == "rem":
            result = _run_rem_inference(paths)
        elif modality == "handwriting":
            result = _run_handwriting_inference(paths)
        else:
            result = _make_unavailable(modality, "unknown_modality")

        modality_results.append(result)

    available_count = sum(1 for r in modality_results if r.available)
    if available_count == 0:
        raise ValueError(
            "No modality data could be processed. "
            "Check input files and modality-specific requirements."
        )

    logger.info(f"Job {job_id} | available modalities: {available_count}/6")

    try:
        fused = late_fusion_model.fuse(modality_results, patient_id, job_id)
    except Exception as e:
        logger.error(f"Late fusion failed: {e}", exc_info=True)
        raise

    if available_count < 3:
        warning = (
            f"Only {available_count}/6 modalities submitted. "
            "Prediction confidence is reduced."
        )
        fused.risk_label = fused.risk_label + "_low_confidence"
        if fused.report_json is None:
            fused.report_json = {}
        fused.report_json["confidence_warning"] = warning
        logger.warning(f"Job {job_id}: {warning}")

    logger.info(
        f"Job {job_id} complete: prob={fused.probability:.4f} "
        f"risk={fused.risk_label} "
        f"modalities_used={[r.modality for r in modality_results if r.available]}"
    )

    # Cleanup tmp dir for this job
    tmp_dir = f"/tmp/{job_id}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return fused