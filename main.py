"""
main.py
=======

FastAPI application entry point with production configuration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from contextlib import asynccontextmanager
import logging
import os
import sys

from api.routes import predict, results, report, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    
    Startup:
    - Warm up critical models (speech, imaging) to cache HuggingFace artifacts
      in /tmp/pd_models before first request (reduces cold-start latency)
    
    Shutdown:
    - Clean up resources
    """
    logger.info("=" * 60)
    logger.info("PD Detection Backend - Startup")
    logger.info("=" * 60)

    # Warm speech fuser
    logger.info("Startup: warming speech ensemble...")
    try:
        from models.speech.intra_model import get_speech_fuser

        fuser = get_speech_fuser()
        logger.info(
            f"✓ Speech ensemble ready. Sub-models: "
            f"{[m.MODEL_ID for m in fuser.models]}"
        )
    except Exception as e:
        logger.error(
            f"✗ Speech ensemble warm-up failed: {e}. "
            "First speech request will trigger download (slower)."
        )

    # Warm imaging fuser (if available)
    logger.info("Startup: warming imaging ensemble...")
    try:
        from models.imaging.intra_model import get_imaging_fuser

        fuser = get_imaging_fuser()
        logger.info(f"✓ Imaging ensemble ready.")
    except Exception as e:
        logger.warning(
            f"Imaging ensemble not available: {e}. "
            "Imaging modality will be unavailable."
        )

    # Warm handwriting model (if available)
    logger.info("Startup: warming handwriting model...")
    try:
        from models.handwriting.model import get_handwriting_model

        model = get_handwriting_model()
        logger.info(f"✓ Handwriting model ready: {model.MODEL_ID}")
    except Exception as e:
        logger.warning(
            f"Handwriting model not available: {e}. "
            "Handwriting modality will be unavailable."
        )

    logger.info("=" * 60)
    logger.info("Backend Ready - Accepting Requests")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutdown: cleaning up resources...")
    # Add any cleanup logic here if needed
    logger.info("Goodbye!")


# ============================================================================
# FastAPI App Configuration
# ============================================================================

app = FastAPI(
    title="PD Detection API",
    version="1.0.0",
    description="Multi-modal Parkinson's Disease detection system",
    lifespan=lifespan,
)

# ============================================================================
# CORS Configuration
# ============================================================================

# Get allowed origins from environment or use default
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Compression Middleware
# ============================================================================

app.add_middleware(
    GZIPMiddleware,
    minimum_size=1000,
)

# ============================================================================
# Include Routers
# ============================================================================

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(results.router, prefix="/api", tags=["results"])
app.include_router(report.router, prefix="/api", tags=["report"])

# ============================================================================
# Root Endpoint
# ============================================================================


@app.get("/", tags=["health"])
def root():
    """
    Root endpoint - service status check.
    """
    return {
        "status": "running",
        "service": "PD Detection Backend",
        "version": "1.0.0",
    }


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    logger.error(
        f"Unhandled exception: {type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return {
        "error": "Internal server error",
        "detail": str(exc),
    }, 500


if __name__ == "__main__":
    # Local development server (use 'uvicorn main:app' for production)
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    logger.info(f"Starting development server on port {port}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info",
    )