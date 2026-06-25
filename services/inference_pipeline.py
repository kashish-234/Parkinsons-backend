"""
services/inference_pipeline.py
================================
Complete inference pipeline supporting all modalities.

Multi-file support:
  - Speech:       N CSV files  → each row is one recording → median-aggregate
  - Handwriting:  N images     → per-image fuse → median-aggregate
  - Gait:         N .npy files → per-file predict → median-aggregate
  - Tapping:      N CSV files  → per-row predict  → median-aggregate
  - Neuroimaging: files tagged "mri_" or "spect_" in the Supabase path
                  are routed to the correct sub-modality.
  - REM:          N CSV files  → concat → single predict_proba call
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
from models.speech.intra_model import predict_speech_batch
from models.fusion.late_fusion import late_fusion_model

logger = logging.getLogger(__name__)

ALL_MODALITIES = ["neuroimaging", "speech", "tapping", "gait", "rem", "handwriting"]


# ============================================================================
# Singletons  (FIX M1 — avoid re-instantiating expensive objects per request)
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
    """
    Download REM artifacts from HuggingFace and return a ready pipeline.
    FIX C4: artifacts come from HF, not from a local 'models/rem/artifacts' dir.
    """
    import joblib
    from models.rem.inference import REMInferencePipeline
    from services.model_storage_service import model_storage_service

    # Required artifacts
    ensemble_path  = model_storage_service.download_model("rem/rem_ensemble.pkl")
    metadata_path  = model_storage_service.download_model("rem/modality_result.json")
    shap_path      = model_storage_service.download_model("rem/shap_explainer.pkl")

    pipeline = REMInferencePipeline.from_local(
        ensemble_path=ensemble_path,
        metadata_path=metadata_path,
        shap_path=shap_path,
    )
    return pipeline


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
# FIX H7: Route files by filename prefix ("mri_" or "spect_") written by the
# predict route when it downloads from Supabase using the modality key.
# The predict route already prefixes files as f"{tmp_dir}/{modality}_{filename}",
# e.g. "neuroimaging_mri_scan1.nii". We further look at the original path to
# determine sub-modality. Convention: caller should pass "neuroimaging" files
# in sub-keys "mri" and "spect" inside data.files, which the predict route
# must flatten. See note in predict.py about updating the request schema.
# For backward compatibility we also accept the raw neuroimaging list and
# route by filename substring.
# ============================================================================

def _run_neuroimaging_inference(paths: List[str]) -> ModalityResult:
    try:
        from models.imaging.intra_model import predict_imaging_batch

        if not paths:
            return _make_unavailable("neuroimaging", "no_files_provided")

        # Route each file to MRI or SPECT based on filename substring.
        # The predict route writes files as: /tmp/{job_id}/neuroimaging_{original_path_basename}
        # The original Supabase paths should be named with "mri" or "spect" somewhere,
        # e.g. "patient_data/mri_scan_001.nii" or "patient_data/spect_scan_001.nii".
        mri_paths   = [p for p in paths if "mri"   in Path(p).name.lower()]
        spect_paths = [p for p in paths if "spect" in Path(p).name.lower()]

        # If no sub-modality tagging, treat all as MRI (best-effort fallback)
        if not mri_paths and not spect_paths:
            logger.warning(
                "Neuroimaging files have no 'mri'/'spect' tag in filenames. "
                "Treating all as MRI. Rename files to include 'mri' or 'spect'."
            )
            mri_paths = paths

        imaging_samples = []
        for mri_path in mri_paths:
            imaging_samples.append({"mri": mri_path})
        for spect_path in spect_paths:
            imaging_samples.append({"spect": spect_path})

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
# Finger Tapping  (FIX M1: use singleton)
# ============================================================================

def _run_tapping_inference(paths: List[str]) -> ModalityResult:
    try:
        if not paths:
            return _make_unavailable("tapping", "no_files_provided")

        model = get_tapping_model()
        results = []

        for path in paths:
            try:
                samples = _load_speech_file(path)  # CSV loader reused
                for sample in samples:
                    model_output = model.predict(sample)
                    result = ModalityResult(
                        modality="tapping",
                        available=True,
                        probability=model_output.probability,
                        ci_low=max(0.0, model_output.probability - 0.15),
                        ci_high=min(1.0, model_output.probability + 0.15),
                        ci_width=0.3,
                        shap_features=model_output.shap_features,
                        model_ids=[model_output.model_id],
                        metadata=model_output.metadata,
                    )
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process tapping file {path}: {e}")

        if not results:
            return _make_unavailable("tapping", "inference_failed_for_all_files")

        from models.speech.intra_model import aggregate_modality_samples
        return aggregate_modality_samples(results)

    except Exception as e:
        logger.error(f"Tapping inference failed: {e}", exc_info=True)
        return _make_unavailable("tapping", f"inference_error: {e}")


# ============================================================================
# Gait  (FIX M1: use singleton)
# ============================================================================

def _run_gait_inference(paths: List[str]) -> ModalityResult:
    try:
        if not paths:
            return _make_unavailable("gait", "no_files_provided")

        import numpy as np
        model = get_gait_model()
        results = []

        for path in paths:
            try:
                window = np.load(path)
                model_output = model.predict({"window": window})
                result = ModalityResult(
                    modality="gait",
                    available=True,
                    probability=model_output.probability,
                    ci_low=max(0.0, model_output.probability - 0.15),
                    ci_high=min(1.0, model_output.probability + 0.15),
                    ci_width=0.3,
                    shap_features=model_output.shap_features,
                    model_ids=[model_output.model_id],
                    metadata=model_output.metadata,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process gait file {path}: {e}")

        if not results:
            return _make_unavailable("gait", "inference_failed_for_all_files")

        from models.speech.intra_model import aggregate_modality_samples
        return aggregate_modality_samples(results)

    except Exception as e:
        logger.error(f"Gait inference failed: {e}", exc_info=True)
        return _make_unavailable("gait", f"inference_error: {e}")


# ============================================================================
# REM  (FIX C4: download artifacts from HF via get_rem_pipeline singleton)
# ============================================================================

def _run_rem_inference(paths: List[str]) -> ModalityResult:
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
        proba = pipeline.predict_proba(combined_df)

        prob_pd = float(proba[:, 0].mean())
        logger.info(f"rem: prob={prob_pd:.3f}")

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
                "n_samples": len(combined_df),
                "per_class_proba": {
                    "PD": float(proba[:, 0].mean()),
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
    try:
        from models.handwriting.model import predict_handwriting_batch

        if not paths:
            return _make_unavailable("handwriting", "no_files_provided")

        result = predict_handwriting_batch(paths)
        logger.info(
            f"handwriting: prob={result.probability:.3f} "
            f"CI=[{result.ci_low:.3f},{result.ci_high:.3f}]"
        )
        return result

    except Exception as e:
        logger.error(f"Handwriting inference failed: {e}", exc_info=True)
        return _make_unavailable("handwriting", f"inference_error: {e}")


# ============================================================================
# Main pipeline
# ============================================================================

def run_inference(patient_id: str, input_paths: Dict[str, List[str]]) -> FusedResult:
    job_id = str(uuid.uuid4())

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
        # FIX H5: Use "confidence_warning" consistently; predict.py now reads this key.
        fused.report_json["confidence_warning"] = warning
        logger.warning(f"Job {job_id}: {warning}")

    # Cleanup tmp dir
    tmp_dir = f"/tmp/{job_id}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        f"Job {job_id} complete: prob={fused.probability:.4f} "
        f"risk={fused.risk_label} "
        f"modalities_used={[r.modality for r in modality_results if r.available]}"
    )

    return fused