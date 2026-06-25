import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from core.auth import verify_user
from api.schemas import PredictRequest, PredictResponse
from services.inference_pipeline import run_inference
from services.supabase_service import persist_fused_result
from services.storage_service import download_file

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    data: PredictRequest,
    background_tasks: BackgroundTasks,
    user=Depends(verify_user),
):
    user_id = data.patient_id  # patient_id from body; user_id from JWT
    user_id = user["user_id"]

    # Use a staging dir keyed by the client-supplied job_id (just for
    # the Supabase Storage downloads; the pipeline uses its own uuid).
    staging_dir = f"/tmp/staging_{data.job_id}"
    os.makedirs(staging_dir, exist_ok=True)

    input_paths: dict = {}

    for modality, paths in data.files.items():
        modality_files = []
        for path in paths:
            filename = os.path.basename(path)
            local_path = f"{staging_dir}/{modality}_{filename}"
            try:
                download_file(path, local_path)
                modality_files.append(local_path)
            except Exception as e:
                # Log and skip; modality will be marked unavailable downstream
                import logging
                logging.getLogger(__name__).warning(
                    f"Could not download {path}: {e}"
                )
                continue
        input_paths[modality] = modality_files

    try:
        fused = run_inference(data.patient_id, input_paths)
    except ValueError as e:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up staging dir
        shutil.rmtree(staging_dir, ignore_errors=True)

    background_tasks.add_task(persist_fused_result, fused, user_id)

    warning = None
    if fused.report_json:
        warning = (
            fused.report_json.get("confidence_warning")
            or fused.report_json.get("warning")
        )

    # FIX: use fused.job_id (the one written to DB) not data.job_id
    return PredictResponse(
        job_id=fused.job_id,
        patient_id=fused.patient_id,
        probability=fused.probability,
        risk_label=fused.risk_label,
        ci_low=fused.ci_low,
        ci_high=fused.ci_high,
        modality_weights=fused.modality_weights,
        available_modalities=[r.modality for r in fused.modality_results if r.available],
        warning=warning,
    )