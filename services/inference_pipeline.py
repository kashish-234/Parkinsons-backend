import os
import csv
import uuid
import shutil
import logging
 
from models.base.contracts import FusedResult, ModalityResult
from models.speech.intra_model import predict_speech_batch
from models.fusion.late_fusion import late_fusion_model
 
logger = logging.getLogger(__name__)
 
ALL_MODALITIES = ["neuroimaging", "speech", "tapping", "gait", "rem", "handwriting"]
 
# ---------------------------------------------------------------------------
# Speech file loading
# ---------------------------------------------------------------------------
 
def _load_speech_file(path: str) -> list[dict]:
    """
    Load a speech feature file (CSV) as a list of dicts.
    Each row in the CSV becomes one dict[str, float] — one recording sample.
    Supports both single-row (one recording per file) and multi-row CSVs.
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
                        pass  # skip non-numeric columns (e.g. filename, label)
                if sample:
                    samples.append(sample)
    except Exception as e:
        logger.error(f"Failed to load speech file {path}: {e}")
    return samples
 
 
def _make_unavailable(modality: str, reason: str) -> ModalityResult:
    """Helper to build a correct unavailable ModalityResult (all fields required)."""
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
 
 
# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------
 
def run_inference(patient_id: str, input_paths: dict[str, list[str]]) -> FusedResult:
    """
    Run full inference for a patient across all submitted modalities.
 
    Args:
        patient_id:  Patient identifier string.
        input_paths: Dict mapping modality name → list of local file paths.
                     Files are already downloaded from Supabase storage
                     by the predict route before this is called.
 
    Returns:
        FusedResult with final probability, risk label, CI, and per-modality
        contributions. Persisted to Supabase by the route as a background task.
    """
    job_id = str(uuid.uuid4())
 
    submitted = [m for m, paths in input_paths.items() if paths]
    logger.info(f"Job {job_id} | patient {patient_id} | submitted: {submitted}")
 
    modality_results: list[ModalityResult] = []
 
    for modality in ALL_MODALITIES:
        paths = input_paths.get(modality, [])
 
        if not paths:
            modality_results.append(_make_unavailable(modality, "not_submitted"))
            continue
 
        # ----------------------------------------------------------------
        # Speech — the only fully deployed modality
        # ----------------------------------------------------------------
        if modality == "speech":
            try:
                # Load every uploaded speech file into feature dicts
                all_samples: list[dict] = []
                for path in paths:
                    samples = _load_speech_file(path)
                    if not samples:
                        logger.warning(f"Speech file produced no samples: {path}")
                    all_samples.extend(samples)
 
                if not all_samples:
                    modality_results.append(
                        _make_unavailable("speech", "no_valid_feature_rows_in_files")
                    )
                    continue
 
                # Run all samples through the speech ensemble (fuses sub-models
                # per sample, then aggregates across N recordings via median)
                result = predict_speech_batch(all_samples)
                modality_results.append(result)
                logger.info(
                    f"speech: prob={result.probability:.3f} "
                    f"CI=[{result.ci_low:.3f},{result.ci_high:.3f}] "
                    f"n_samples={result.metadata.get('n_samples', len(all_samples))}"
                )
 
            except Exception as e:
                logger.error(f"Speech inference failed: {e}", exc_info=True)
                modality_results.append(_make_unavailable("speech", f"inference_error: {e}"))
 
        # ----------------------------------------------------------------
        # Other modalities — not yet deployed
        # Add modality-specific inference logic here as each is trained:
        #
        # elif modality == "gait":
        #     from models.gait.inference import GaitModel
        #     result = GaitModel().predict_batch(paths)
        #     modality_results.append(result)
        #
        # elif modality == "tapping":
        #     from models.finger_tapping.inference import TappingModel
        #     result = TappingModel().predict_batch(paths)
        #     modality_results.append(result)
        # ----------------------------------------------------------------
        else:
            modality_results.append(
                _make_unavailable(modality, "modality_not_yet_deployed")
            )
 
    # ----------------------------------------------------------------
    # Validate at least one modality is available
    # ----------------------------------------------------------------
    available_count = sum(1 for r in modality_results if r.available)
    if available_count == 0:
        raise ValueError(
            "No modality data could be processed. "
            "Check that speech CSV files contain valid acoustic feature columns."
        )
 
    # ----------------------------------------------------------------
    # Late fusion — heuristic weighted average until LightGBM meta-learner
    # is trained on multi-modal labeled data (see models/fusion/late_fusion.py)
    # ----------------------------------------------------------------
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
        fused.risk_label = fused.risk_label.rstrip("_low_confidence") + "_low_confidence"
        fused.report_json = {"warning": warning}
        logger.warning(f"Job {job_id}: {warning}")
 
    # ----------------------------------------------------------------
    # Cleanup temp files for this job
    # ----------------------------------------------------------------
    tmp_dir = f"/tmp/{job_id}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
 
    logger.info(
        f"Job {job_id} complete: prob={fused.probability:.4f} "
        f"risk={fused.risk_label} "
        f"modalities_used={[r.modality for r in modality_results if r.available]}"
    )
    return fused