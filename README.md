# Parkinson's Disease Detection Backend

A FastAPI backend for multi-modal Parkinson's Disease prediction and report generation.

It combines several specialized models across speech, gait, finger tapping, REM, neuroimaging, and handwriting modalities, stores fused results in Supabase, and generates clinical-style reports with Gemini.

## Key Features

- FastAPI service with OpenAPI docs available at `/docs` and `/redoc`
- Supabase JWT authentication for protected endpoints
- Multi-modal prediction pipeline with model fusion
- Result persistence to Supabase tables
- Report generation using Gemini and RAG-style retrieval from SHAP embeddings
- Lazy model loading to conserve memory on low-tier deployments
- Ready for deployment on Render with `render.yaml`

## Repository Structure
- `main.py` - FastAPI application entrypoint
- `api/` - REST route definitions and request/response schemas
- `core/` - configuration and authentication logic
- `services/` - inference pipeline, storage, Supabase access, report/RAG services
- `models/` - model wrappers and inference code for each modality
- `requirements.txt` - pinned Python dependencies
- `render.yaml` - Render deployment configuration

## Requirements

- Python 3.11.x
- Supabase project with:
  - `supabase_url`
  - `supabase_service_key`
  - A `patient-data` storage bucket for patient files
  - Tables: `fused_results`, `modality_results`, `clinical_reports`, `shap_embeddings`
- Gemini API key for report generation
- Optional Hugging Face token for private model access

## Environment Variables

Create a `.env` file at the repository root or set the environment variables directly.

Required:

- `SUPABASE_URL` - Supabase API URL
- `SUPABASE_SERVICE_KEY` - Supabase service role key
- `GEMINI_API_KEY` - Gemini / Google Cloud GenAI API key

Optional:

- `HF_TOKEN` - Hugging Face token (if using private model repo)
- `HF_MODEL_REPO` - Hugging Face model repository name
- `ENVIRONMENT` - `development` or `production` (default: `development`)
- `DEBUG` - `true` or `false` (default: `false`)
- `MODEL_CACHE_DIR` - model cache directory (default: `/tmp/pd_models`)
- `ALLOWED_ORIGINS` - comma-separated CORS origins (default: `http://localhost:3000`)
- `PORT` - server port (default: `8000`)

## Setup

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` with at least:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
GEMINI_API_KEY=your-gemini-api-key
HF_TOKEN=your-hf-token
ENVIRONMENT=development
DEBUG=true
MODEL_CACHE_DIR=/tmp/pd_models
```

## Run Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Visit:

- `http://localhost:8000/docs` - Swagger UI
- `http://localhost:8000/redoc` - ReDoc UI
- `http://localhost:8000/api/health` - health status

## API Endpoints

### `GET /api/health`

Returns service health and basic integration status.

Example response:

```json
{
  "status": "ok",
  "environment": "development",
  "supabase": "connected",
  "gemini": "configured"
}
```

### `POST /api/predict`

Protected endpoint. Requires `Authorization: Bearer <token>`.

Request body:

```json
{
  "patient_id": "patient_123",
  "job_id": "client_job_001",
  "files": {
    "speech": ["session/123/speech.csv"],
    "gait": ["session/123/gait.npy"],
    "handwriting": ["session/123/handwriting.png"]
  }
}
```

Response:

```json
{
  "job_id": "...",
  "patient_id": "patient_123",
  "probability": 0.82,
  "risk_label": "high",
  "ci_low": 0.75,
  "ci_high": 0.89,
  "modality_weights": {
    "speech": 0.25,
    "gait": 0.30,
    "handwriting": 0.20
  },
  "available_modalities": ["speech", "gait", "handwriting"],
  "warning": null
}
```

### `GET /api/results/{job_id}`

Protected endpoint. Returns persisted fused prediction results for the authenticated user.

Example response:

```json
{
  "job_id": "...",
  "patient_id": "patient_123",
  "probability": 0.82,
  "risk_label": "high",
  "ci_low": 0.75,
  "ci_high": 0.89,
  "modality_weights": {"speech": 0.25, "gait": 0.30},
  "modality_results": [...],
  "report_json": {...}
}
```

### `GET /api/report/{job_id}`

Protected endpoint. Returns an existing generated report.

### `POST /api/report/{job_id}`

Protected endpoint. Generates a report if one does not already exist.

The report uses:

- persisted predictions from `fused_results`
- SHAP embeddings for retrieval context
- Gemini-based report generation

## Authentication

The backend uses Supabase JWT authentication.

Clients must send:

```http
Authorization: Bearer <access_token>
```

The service validates tokens against Supabase JWKS and requires the audience `authenticated`.

## Deployment

This project is configured for Render with `render.yaml`.

Key deployment notes:

- Uses Python `3.11.10`
- Installs Torch CPU wheels from the PyTorch CPU index
- Starts with `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health endpoint: `/api/health`

## Notes

- The service warms lightweight sklearn/joblib models at startup.
- Heavy PyTorch/TensorFlow models are lazy-loaded on first request to conserve memory.
- `MODEL_CACHE_DIR` should be set for persistent model caching on hosted environments.

## Contributing

1. Fork the repo
2. Create a feature branch
3. Update tests and documentation
4. Open a pull request

## License

See `LICENSE`.
 