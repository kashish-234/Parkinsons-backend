"""
services/inference_pipeline.py
==============================

Complete inference pipeline supporting all modalities:
- speech (multi-dataset fusion)
- neuroimaging (MRI + SPECT fusion)
- finger tapping
- gait
- REM
- handwriting
"""

import os
import csv
import uuid
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

from models.base.contracts import FusedResult, ModalityResult
from models.speech.intra_model import predict_speech_batch
from models.fusion.late_fusion import late_fusion_model

logger = logging.getLogger(__name__)

ALL_MODALITIES = [
    "neuroimaging",
    "speech",
    "tapping",
    "gait",
    "rem",
    "handwriting",
]


# ============================================================================
# Utility Functions
# ============================================================================


def _make_unavailable(modality: str, reason: str) -> ModalityResult:
    """Create unavailable ModalityResult with all required fields."""
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
# Speech Modality
# ============================================================================


def _load_speech_file(path: str) -> List[dict]:
    """
    Load speech feature CSV file as list of feature dicts.

    Each row becomes one sample: dict[str, float] → one recording.
    Supports single-row and multi-row CSVs.

    Args:
        path: Path to CSV file

    Returns:
        List of feature dicts, one per row
    """
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
                        pass  # Skip non-numeric columns
                if sample:
                    samples.append(sample)
    except Exception as e:
        logger.error(f"Failed to load speech file {path}: {e}")
    return samples


def _run_speech_inference(paths: List[str]) -> ModalityResult:
    """
    Run speech inference on uploaded files.

    Args:
        paths: List of local file paths to speech CSVs

    Returns:
        ModalityResult for speech modality
    """
    try:
        all_samples: List[dict] = []
        for path in paths:
            samples = _load_speech_file(path)
            if not samples:
                logger.warning(f"Speech file produced no samples: {path}")
            all_samples.extend(samples)

        if not all_samples:
            return _make_unavailable(
                "speech", "no_valid_feature_rows_in_files"
            )

        # Run through speech ensemble (fuses sub-models per sample,
        # then aggregates across N recordings via median)
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
# Neuroimaging Modality (MRI + SPECT fusion)
# ============================================================================


def _run_neuroimaging_inference(paths: List[str]) -> ModalityResult:
    """
    Run neuroimaging inference (MRI + SPECT fusion) on uploaded files.

    Args:
        paths: List of local file paths to imaging files

    Returns:
        ModalityResult for neuroimaging modality
    """
    try:
        from models.imaging.intra_model import predict_imaging_batch

        if not paths:
            return _make_unavailable("neuroimaging", "no_files_provided")

        # Load imaging samples from files
        # Assumes files are named with modality prefix or in sub-directories:
        # e.g., "mri_scan_001.nii", "spect_scan_001.nii", or "mri/scan_001.nii"
        imaging_samples = []
        for path in paths:
            try:
                # Simple approach: treat each file as one sample
                # More sophisticated: group by modality
                imaging_samples.append({"mri": path} if "mri" in path.lower() else {"spect": path})
            except Exception as e:
                logger.warning(f"Failed to load imaging file {path}: {e}")

        if not imaging_samples:
            return _make_unavailable("neuroimaging", "no_valid_imaging_files")

        result = predict_imaging_batch(imaging_samples)
        logger.info(
            f"neuroimaging: prob={result.probability:.3f} "
            f"CI=[{result.ci_low:.3f},{result.ci_high:.3f}]"
        )
        return result

    except Exception as e:
        logger.error(f"Neuroimaging inference failed: {e}", exc_info=True)
        return _make_unavailable(
            "neuroimaging", f"inference_error: {e}"
        )


# ============================================================================
# Finger Tapping Modality
# ============================================================================


def _run_tapping_inference(paths: List[str]) -> ModalityResult:
    """
    Run finger tapping inference on uploaded files.

    Args:
        paths: List of local file paths to tapping data

    Returns:
        ModalityResult for tapping modality
    """
    try:
        from models.finger_tapping.inference import FingerTappingLGBM

        if not paths:
            return _make_unavailable("tapping", "no_files_provided")

        model = FingerTappingLGBM()
        results = []

        for path in paths:
            try:
                # Load tapping features from CSV
                samples = _load_speech_file(path)  # Reuse CSV loading
                for sample in samples:
                    model_output = model.predict(sample)

                    # Convert ModelOutput to ModalityResult
                    result = ModalityResult(
                        modality="tapping",
                        available=True,
                        probability=model_output.probability,
                        ci_low=max(0.0, model_output.probability - 0.1),
                        ci_high=min(1.0, model_output.probability + 0.1),
                        ci_width=0.2,
                        shap_features=model_output.shap_features,
                        model_ids=[model_output.model_id],
                        metadata=model_output.metadata,
                    )
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process tapping file {path}: {e}")

        if not results:
            return _make_unavailable("tapping", "inference_failed_for_all_files")

        # Aggregate results
        from models.speech.intra_model import aggregate_modality_samples
        return aggregate_modality_samples(results)

    except Exception as e:
        logger.error(f"Tapping inference failed: {e}", exc_info=True)
        return _make_unavailable("tapping", f"inference_error: {e}")


# ============================================================================
# Gait Modality
# ============================================================================


def _run_gait_inference(paths: List[str]) -> ModalityResult:
    """
    Run gait inference on uploaded files.

    Args:
        paths: List of local file paths to gait sensor data

    Returns:
        ModalityResult for gait modality
    """
    try:
        from models.gait.inference import GaitDaphnetCNNLSTM

        if not paths:
            return _make_unavailable("gait", "no_files_provided")

        model = GaitDaphnetCNNLSTM()
        results = []

        for path in paths:
            try:
                # Load gait sensor window from file (assumes numpy-serialized format)
                import numpy as np
                window = np.load(path)  # or load from CSV/JSON as needed

                model_output = model.predict({"window": window})

                # Convert ModelOutput to ModalityResult
                result = ModalityResult(
                    modality="gait",
                    available=True,
                    probability=model_output.probability,
                    ci_low=max(0.0, model_output.probability - 0.1),
                    ci_high=min(1.0, model_output.probability + 0.1),
                    ci_width=0.2,
                    shap_features=model_output.shap_features,
                    model_ids=[model_output.model_id],
                    metadata=model_output.metadata,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process gait file {path}: {e}")

        if not results:
            return _make_unavailable("gait", "inference_failed_for_all_files")

        # Aggregate results
        from models.speech.intra_model import aggregate_modality_samples
        return aggregate_modality_samples(results)

    except Exception as e:
        logger.error(f"Gait inference failed: {e}", exc_info=True)
        return _make_unavailable("gait", f"inference_error: {e}")


# ============================================================================
# REM Modality
# ============================================================================


def _run_rem_inference(paths: List[str]) -> ModalityResult:
    """
    Run REM sleep behaviour disorder inference on uploaded files.

    Args:
        paths: List of local file paths to REM data (CSV)

    Returns:
        ModalityResult for rem modality
    """
    try:
        import pandas as pd
        from models.rem.inference import REMInferencePipeline

        if not paths:
            return _make_unavailable("rem", "no_files_provided")

        # Load and aggregate all REM data
        dfs = []
        for path in paths:
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load REM file {path}: {e}")

        if not dfs:
            return _make_unavailable("rem", "no_valid_rem_files")

        combined_df = pd.concat(dfs, ignore_index=True)

        # Load REM pipeline and run inference
        pipeline = REMInferencePipeline.from_artifacts(
            artifact_dir="models/rem/artifacts"
        )
        proba = pipeline.predict_proba(combined_df)

        # Extract P(PD) = probability of class 0
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
# Handwriting Modality
# ============================================================================


def _run_handwriting_inference(paths: List[str]) -> ModalityResult:
    """
    Run handwriting analysis inference on uploaded image files.

    Args:
        paths: List of local file paths to handwriting images

    Returns:
        ModalityResult for handwriting modality
    """
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
        return _make_unavailable(
            "handwriting", f"inference_error: {e}"
        )


# ============================================================================
# Main Inference Pipeline
# ============================================================================


def run_inference(
    patient_id: str, input_paths: Dict[str, List[str]]
) -> FusedResult:
    """
    Run complete inference pipeline across all modalities.

    Args:
        patient_id: Patient identifier
        input_paths: Dict mapping modality name → list of local file paths.
                     Files are already downloaded from Supabase storage
                     by the predict route.

    Returns:
        FusedResult with final probability, risk label, CI, and per-modality
        contributions.

    Raises:
        ValueError: If no modality data could be processed
    """
    job_id = str(uuid.uuid4())

    submitted = [m for m, paths in input_paths.items() if paths]
    logger.info(
        f"Job {job_id} | patient {patient_id} | submitted: {submitted}"
    )

    modality_results: List[ModalityResult] = []

    # Run inference for each modality
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

    # ========================================================================
    # Validate and Fuse
    # ========================================================================

    available_count = sum(1 for r in modality_results if r.available)
    if available_count == 0:
        raise ValueError(
            "No modality data could be processed. "
            "Check input files and modality-specific requirements."
        )

    logger.info(
        f"Job {job_id} | available modalities: {available_count}/6"
    )

    # Late fusion
    try:
        fused = late_fusion_model.fuse(modality_results, patient_id, job_id)
    except Exception as e:
        logger.error(f"Late fusion failed: {e}", exc_info=True)
        raise

    # Append confidence warning if few modalities available
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

    # ========================================================================
    # Cleanup
    # ========================================================================

    tmp_dir = f"/tmp/{job_id}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(
        f"Job {job_id} complete: prob={fused.probability:.4f} "
        f"risk={fused.risk_label} "
        f"modalities_used={[r.modality for r in modality_results if r.available]}"
    )

    return fused