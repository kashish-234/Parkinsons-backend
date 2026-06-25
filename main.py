"""
main.py — FastAPI application entry point.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os

from api.routes import predict, results, report, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("PD Detection Backend - Startup")
    logger.info("=" * 60)

    # Warm speech fuser
    logger.info("Startup: warming speech ensemble...")
    try:
        from models.speech.intra_model import get_speech_fuser
        fuser = get_speech_fuser()
        logger.info(f"✓ Speech ensemble ready. Sub-models: {[m.MODEL_ID for m in fuser.models]}")
    except Exception as e:
        logger.error(f"✗ Speech ensemble warm-up failed: {e}.")

    # Warm imaging fuser
    logger.info("Startup: warming imaging ensemble...")
    try:
        from models.imaging.intra_model import get_imaging_fuser
        get_imaging_fuser()
        logger.info("✓ Imaging ensemble ready.")
    except Exception as e:
        logger.warning(f"Imaging ensemble not available: {e}.")

    # Warm handwriting model
    logger.info("Startup: warming handwriting model...")
    try:
        from models.handwriting.model import get_handwriting_model
        model = get_handwriting_model()
        logger.info(f"✓ Handwriting model ready: {model.MODEL_ID}")
    except Exception as e:
        logger.warning(f"Handwriting model not available: {e}.")

    # Warm gait model
    logger.info("Startup: warming gait model...")
    try:
        from services.inference_pipeline import get_gait_model
        get_gait_model()
        logger.info("✓ Gait model ready.")
    except Exception as e:
        logger.warning(f"Gait model not available: {e}.")

    # Warm tapping model
    logger.info("Startup: warming finger-tapping model...")
    try:
        from services.inference_pipeline import get_tapping_model
        get_tapping_model()
        logger.info("✓ Finger-tapping model ready.")
    except Exception as e:
        logger.warning(f"Finger-tapping model not available: {e}.")

    # Warm REM pipeline
    logger.info("Startup: warming REM pipeline...")
    try:
        from services.inference_pipeline import get_rem_pipeline
        get_rem_pipeline()
        logger.info("✓ REM pipeline ready.")
    except Exception as e:
        logger.warning(f"REM pipeline not available: {e}.")

    logger.info("=" * 60)
    logger.info("Backend Ready - Accepting Requests")
    logger.info("=" * 60)

    yield

    logger.info("Shutdown: cleaning up resources...")
    logger.info("Goodbye!")


app = FastAPI(
    title="PD Detection API",
    version="1.0.0",
    description="Multi-modal Parkinson's Disease detection system",
    lifespan=lifespan,
)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZIPMiddleware, minimum_size=1000)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(results.router, prefix="/api", tags=["results"])
app.include_router(report.router, prefix="/api", tags=["report"])


@app.get("/", tags=["health"])
def root():
    return {"status": "running", "service": "PD Detection Backend", "version": "1.0.0"}


# FIX C1: Must return a JSONResponse, not a bare (dict, int) tuple.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload, log_level="info")