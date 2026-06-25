from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

from api.routes import predict, results, report, health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the speech fuser so the first request doesn't bear the
    # HuggingFace download latency (artifacts cached to /tmp/pd_models)
    logger.info("Startup: warming speech fuser...")
    try:
        from models.speech.intra_model import get_speech_fuser
        get_speech_fuser()
        logger.info("Speech fuser ready.")
    except Exception as e:
        logger.error(
            f"Speech fuser warm-up failed: {e}. "
            "First speech request will trigger download."
        )
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="PD Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

# Set in Render environment variables:
#   ALLOWED_ORIGINS=https://your-app.vercel.app,https://staging.your-app.com
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")
app.include_router(results.router, prefix="/api")
app.include_router(report.router,  prefix="/api")
app.include_router(health.router,  prefix="/api")


@app.get("/")
def root():
    return {"status": "running", "service": "PD Detection Backend"}