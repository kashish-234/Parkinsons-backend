import os
import uuid
import shutil
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from typing import Optional

from core.auth import verify_user
from api.schemas import PredictResponse
from services.inference_pipeline import run_inference
from services.supabase_service import persist_fused_result

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    background_tasks: BackgroundTasks,
    user=Depends(verify_user),
    # ── form fields ──────────────────────────────────────────────────────────
    patient_id: str = Form(...),          # human-readable patient code
    patient_uuid: Optional[str] = Form(None),  # UUID FK to patients table
    job_id: Optional[str] = Form(None),
    # ── files ────────────────────────────────────────────────────────────────
    # Each modality is a separate multi-file field.
    # FastAPI collects all files with the same field name into a list.
    speech: list[UploadFile] = File(default=[]),
    handwriting: list[UploadFile] = File(default=[]),
    tapping: list[UploadFile] = File(default=[]),
    gait: list[UploadFile] = File(default=[]),
    rem: list[UploadFile] = File(default=[]),
    neuroimaging: list[UploadFile] = File(default=[]),
):
    user_id = user["user_id"]
    job_id = job_id or str(uuid.uuid4())

    staging_dir = f"/tmp/staging_{job_id}_{os.getpid()}"
    os.makedirs(staging_dir, exist_ok=True)

    input_paths: dict[str, list[str]] = {}

    modality_uploads = {
        "speech": speech,
        "handwriting": handwriting,
        "tapping": tapping,
        "gait": gait,
        "rem": rem,
        "neuroimaging": neuroimaging,
    }

    try:
        # Save uploaded bytes to /tmp staging dir
        for modality, uploads in modality_uploads.items():
            if not uploads:
                continue
            modal_files = []
            for upload in uploads:
                if not upload.filename:
                    continue
                safe_name = os.path.basename(upload.filename)
                local_path = os.path.join(staging_dir, f"{modality}_{safe_name}")
                content = await upload.read()
                with open(local_path, "wb") as f:
                    f.write(content)
                modal_files.append(local_path)
                logger.info("Staged %s (%d bytes) → %s", upload.filename, len(content), local_path)
            if modal_files:
                input_paths[modality] = modal_files

        if not input_paths:
            raise HTTPException(status_code=400, detail="No files received")

        fused = run_inference(patient_id=patient_id, input_paths=input_paths, job_id=job_id)

        if patient_uuid:
            fused.patient_uuid = patient_uuid

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always delete the temp files — no storage needed
        shutil.rmtree(staging_dir, ignore_errors=True)

    background_tasks.add_task(persist_fused_result, fused=fused, user_id=user_id)

    warning = None
    if fused.report_json:
        warning = (
            fused.report_json.get("confidence_warning")
            or fused.report_json.get("warning")
        )

    return PredictResponse(
        job_id=fused.job_id,
        patient_id=fused.patient_id,
        probability=fused.probability,
        risk_label=fused.risk_label,
        ci_low=fused.ci_low,
        ci_high=fused.ci_high,
        modality_weights=fused.modality_weights,
        available_modalities=[r.modality for r in fused.modality_results if r.available],
        fusion_model_version=fused.fusion_model_version,
        warning=warning,
    )
