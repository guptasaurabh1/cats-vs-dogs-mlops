"""
FastAPI inference service for Cats vs Dogs classification.
Endpoints:
  - GET /health: Health check
  - POST /predict: Predict image class (accepts file upload)
  - GET /metrics: Request count and latency metrics
"""

import io
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import HealthResponse, PredictionResponse
from src.predict import CatDogClassifier
from monitoring.metrics import MetricsCollector
from monitoring.performance_tracker import PerformanceTracker


# ---------------------------------------------------------------------------
# Globals (set during lifespan)
# ---------------------------------------------------------------------------
class ModelState:
    classifier: CatDogClassifier = None
    metrics: MetricsCollector = None
    performance: PerformanceTracker = None


state = ModelState()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("uvicorn.api")
log_fmt = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_fmt,
    handlers=[
        logging.FileHandler("api/api.log"),
        logging.StreamHandler(),
    ],
)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and initialise metrics on startup."""
    os.makedirs("api", exist_ok=True)
    logger.info("Starting Cats vs Dogs inference service...")

    model_path = os.getenv("MODEL_PATH", "models/model.pt")
    device = os.getenv("DEVICE", "cpu")

    try:
        state.classifier = CatDogClassifier(model_path=model_path, device=device)
        state.metrics = MetricsCollector()
        state.performance = PerformanceTracker()
        logger.info(f"Model loaded from {model_path} on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        state.classifier = None

    yield

    logger.info("Shutting down inference service.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="MLOps pipeline inference service for pet adoption platform",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if state.metrics is not None:
        state.metrics.record_request("/health")
    return HealthResponse(
        status="healthy",
        model_loaded=state.classifier is not None,
        device=os.getenv("DEVICE", "cpu"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an uploaded image is a cat or a dog.
    Accepts JPEG/PNG files.
    """
    start_time = time.time()

    if state.classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. Use JPEG or PNG.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        result = state.classifier.predict(image)

        latency = time.time() - start_time
        state.metrics.record_request("/predict", latency=latency)

        # Record prediction for post-deployment performance monitoring
        # (true_label is unknown at request time; filled in later via
        #  performance_tracker.py update)
        if state.performance is not None:
            state.performance.record_prediction(
                input_id=file.filename or str(time.time()),
                true_label=None,
                predicted_label=result["prediction"],
                confidence=result["confidence"],
                latency_ms=latency * 1000,
            )

        # Log request (excluding raw image data)
        logger.info(
            f"Prediction | file={file.filename} | "
            f"prediction={result['prediction']} | "
            f"confidence={result['confidence']:.4f} | "
            f"latency={latency:.3f}s"
        )

        return PredictionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        if state.metrics is not None:
            state.metrics.record_error()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict classes for multiple uploaded images in one request.
    Exceeds baseline: supports batching for throughput-sensitive clients.
    """
    start_time = time.time()

    if state.classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for file in files:
        if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {file.content_type}. Use JPEG or PNG.",
            )
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = state.classifier.predict(image)
        result["file"] = file.filename
        results.append(result)

    latency = time.time() - start_time
    state.metrics.record_request("/predict_batch", latency=latency)

    logger.info(
        f"Batch prediction | {len(files)} images | "
        f"latency={latency:.3f}s"
    )

    return {"count": len(results), "predictions": results}


@app.get("/metrics")
async def get_metrics():
    """Return request metrics: counts, latency stats, and error count."""
    if state.metrics is None:
        return JSONResponse({"error": "Metrics not initialized"}, status_code=503)
    return JSONResponse(state.metrics.get_summary())


@app.get("/monitor/performance")
async def monitor_performance():
    """Post-deployment accuracy + drift status (M5)."""
    if state.performance is None:
        return JSONResponse({"error": "Performance tracker not initialized"}, status_code=503)
    accuracy = state.performance.get_accuracy()
    drift = state.performance.get_drift_status()
    return JSONResponse({"accuracy": accuracy, "drift": drift})


@app.get("/")
async def root():
    return {"message": "Cats vs Dogs Classifier API", "docs": "/docs"}
