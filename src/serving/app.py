"""
Model Serving API - HeartCare MLOps Pipeline
Task 6: FastAPI service exposing /predict (accepts JSON patient data, returns
prediction + confidence) running the packaged preprocessing+model pipeline.
Task 8: request logging + Prometheus metrics for monitoring.

Run locally:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
or:
    python src/serving/app.py
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# Make the project importable (so the pickled pipeline can resolve its
# preprocessing class, and so we can reuse the predict helper).
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE_DIR)
from src.packaging.package_model import load_pipeline, predict as pipeline_predict

LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'api.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

# Monitoring (Task 8) - Prometheus metrics + structured access logging.
# Centralized in src/monitoring/metrics.py; degrades to no-ops if
# prometheus_client is not installed.
from src.monitoring import metrics as mon


# ---- Pydantic schemas ----
class PatientFeatures(BaseModel):
    """Raw clinical features for a single patient."""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="0 = female, 1 = male")
    cp: int = Field(..., ge=0, le=4, description="Chest pain type")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=3, description="Slope of peak exercise ST")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels")
    thal: int = Field(..., ge=0, le=7, description="Thalassemia")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float


class BatchRequest(BaseModel):
    patients: List[PatientFeatures]


# ---- App + model lifecycle ----
_pipeline = None


def get_pipeline():
    """Lazy-load the packaged pipeline once per process."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading model pipeline...")
        _pipeline = load_pipeline()
        logger.info("Model pipeline loaded")
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; report not-ready via /health if it fails."""
    try:
        get_pipeline()
        mon.set_model_ready(True)
    except Exception as e:
        mon.set_model_ready(False)
        logger.error(f"Model load failed at startup: {e}")
    yield


app = FastAPI(
    title="HeartCare Risk Prediction API",
    description="Predicts heart disease risk from patient clinical data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with latency and status (Task 8)."""
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    client = request.client.host if request.client else '-'
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"({elapsed*1000:.1f} ms) client={client}"
    )
    mon.record_request(
        request.url.path, request.method, response.status_code, elapsed, client
    )
    return response


@app.get("/")
def root():
    return {
        "service": "HeartCare Risk Prediction API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/predict/batch", "/health", "/metrics"],
    }


@app.get("/health")
def health():
    """Readiness probe - reports whether the model is loaded."""
    ready = _pipeline is not None
    status = "healthy" if ready else "model_not_loaded"
    return {"status": status, "model_loaded": ready,
            "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
def predict_one(features: PatientFeatures):
    """Predict heart disease risk for a single patient."""
    try:
        pipeline = get_pipeline()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")

    try:
        result = pipeline_predict(features.model_dump(), pipeline=pipeline)[0]
    except Exception as e:
        mon.record_prediction_error()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    mon.record_prediction(result['label'])
    logger.info(f"Prediction: {result}")
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    """Predict heart disease risk for multiple patients."""
    try:
        pipeline = get_pipeline()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")

    records = [p.model_dump() for p in req.patients]
    try:
        results = pipeline_predict(records, pipeline=pipeline)
    except Exception as e:
        mon.record_prediction_error()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    for r in results:
        mon.record_prediction(r['label'])
    return {"count": len(results), "predictions": results}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint (Task 8)."""
    payload, content_type = mon.metrics_payload()
    return PlainTextResponse(payload, media_type=content_type)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
